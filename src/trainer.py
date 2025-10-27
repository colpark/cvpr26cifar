import os
import math
import numpy as np
from .dataset import dataset_wrapper
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torch.optim import Adam
# from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from multiprocessing import cpu_count
from functools import partial
from tqdm import tqdm
import datetime
from termcolor import colored
from .utils import *
from torchvision.utils import make_grid
from .sparsity import SparsityController
# from .perceiver import *   # ← removed
import torch.nn.functional as F

def cycle_with_label(dl):
    while True:
        for data in dl:
            img, label = data
            yield img

def cycle(dl):
    while True:
        for data in dl:
            yield data

class Trainer:
    def __init__(self, diffusion_model, dataset, batch_size=32, lr=2e-5, total_step=100000, ddim_samplers=None,
                 save_and_sample_every=1000, num_samples=25, result_folder='./results', cpu_percentage=0,
                 fid_estimate_batch_size=None, ddpm_fid_score_estimate_every=None, ddpm_num_fid_samples=None,
                 max_grad_norm=1., tensorboard=False, exp_name=None, clip=True,
                 sparsity_mode='random_epoch', sparsity_pattern='random', sparsity_level=0.2, block_size=5, num_blocks=5, **dataset_kwargs):

        now = datetime.datetime.now()
        self.cur_time = now.strftime('%Y-%m-%d_%Hh%Mm')
        if exp_name is None:
            exp_name = os.path.basename(dataset)
            if exp_name == '':
                exp_name = os.path.basename(os.path.dirname(dataset))
        self.exp_name = exp_name
        self.diffusion_model = diffusion_model
        self.ddim_samplers = ddim_samplers
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.nrow = int(math.sqrt(self.num_samples))
        assert (self.nrow ** 2) == self.num_samples, 'num_samples must be a square number. ex) 25, 36, 49, ...'
        self.save_and_sample_every = save_and_sample_every
        self.image_size = self.diffusion_model.image_size
        self.max_grad_norm = max_grad_norm
        self.result_folder = os.path.join(result_folder, exp_name, self.cur_time)
        self.ddpm_result_folder = os.path.join(self.result_folder, 'DDPM')
        self.device = self.diffusion_model.device
        self.clip = clip
        self.ddpm_fid_flag = True if ddpm_fid_score_estimate_every is not None else False
        self.ddpm_fid_score_estimate_every = ddpm_fid_score_estimate_every
        self.cal_fid = True if self.ddpm_fid_flag else False
        self.tqdm_sampler_name = None
        self.tensorboard = tensorboard
        self.tensorboard_name = None
        self.writer = None
        self.global_step = 0
        self.total_step = total_step
        self.fid_score_log = dict()
        assert clip in [True, False, 'both'], "clip must be one of [True, False, 'both']"
        if clip is True or clip == 'both':
            os.makedirs(os.path.join(self.ddpm_result_folder, 'clip'), exist_ok=True)
        if clip is False or clip == 'both':
            os.makedirs(os.path.join(self.ddpm_result_folder, 'no_clip'), exist_ok=True)

        # ===== Dataset & DataLoader & Optimizer =====
        notification = make_notification('Dataset', color='green')
        print(notification)
        dataSet = dataset_wrapper(dataset, self.image_size, **dataset_kwargs)
        assert len(dataSet) >= 100, 'you should have at least 100 images in your folder.at least 10k images recommended'
        print(colored('Dataset Length: {}\n'.format(len(dataSet)), 'green'))
        CPU_cnt = cpu_count()
        num_workers = int(CPU_cnt * cpu_percentage)
        assert num_workers <= CPU_cnt, "cpu_percentage must be [0.0, 1.0]"
        dataLoader = DataLoader(dataSet, batch_size=self.batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=True)
        self.dataLoader = cycle(dataLoader) if os.path.isdir(dataset) else cycle_with_label(dataLoader)
        self.optimizer = Adam(self.diffusion_model.parameters(), lr=lr)

        self.sparsity_controller = SparsityController(
            image_size=self.image_size,
            mode=sparsity_mode,
            pattern=sparsity_pattern,
            sparsity=sparsity_level,
            block_size=block_size,
            num_blocks=num_blocks
        )

        # ===== DDIM sampler setting =====
        self.ddim_sampling_schedule = list()
        for idx, sampler in enumerate(self.ddim_samplers):
            sampler.sampler_name = 'DDIM_{}_steps{}_eta{}'.format(idx + 1, sampler.ddim_steps, sampler.eta)
            self.ddim_sampling_schedule.append(sampler.sample_every)
            save_path = os.path.join(self.result_folder, sampler.sampler_name)
            sampler.save_path = save_path
            if sampler.save:
                os.makedirs(save_path, exist_ok=True)
            if sampler.generate_image:
                if sampler.clip is True or sampler.clip == 'both':
                    os.makedirs(os.path.join(save_path, 'clip'), exist_ok=True)
                if sampler.clip is False or sampler.clip == 'both':
                    os.makedirs(os.path.join(save_path, 'no_clip'), exist_ok=True)
            if sampler.calculate_fid:
                self.cal_fid = True
                if self.tqdm_sampler_name is None:
                    self.tqdm_sampler_name = sampler.sampler_name
                sampler.num_fid_sample = sampler.num_fid_sample if sampler.num_fid_sample is not None else len(dataSet)
                self.fid_score_log[sampler.sampler_name] = list()
            if sampler.fixed_noise:
                sampler.register_buffer('noise', torch.randn([self.num_samples, sampler.channel,
                                                              sampler.image_size, sampler.image_size]))

        # ===== Image generation log =====
        notification = make_notification('Image Generation', color='cyan')
        print(notification)
        print(colored('Image will be generated with the following sampler(s)', 'cyan'))
        print(colored('-> DDPM Sampler / Image generation every {} steps'.format(self.save_and_sample_every),
                      'cyan'))
        for sampler in self.ddim_samplers:
            if sampler.generate_image:
                print(colored('-> {} / Image generation every {} steps / Fixed Noise : {}'
                              .format(sampler.sampler_name, sampler.sample_every, sampler.fixed_noise), 'cyan'))
        print('\n')

        # ===== FID score =====
        notification = make_notification('FID', color='magenta')
        print(notification)
        if not self.cal_fid or dataset.lower() == 'navierstokes':
            print(colored('No FID evaluation will be executed!\n'
                          'If you want FID evaluation consider using DDIM sampler.', 'magenta'))
        else:
            self.fid_batch_size = fid_estimate_batch_size if fid_estimate_batch_size is not None else self.batch_size
            dataSet_fid = dataset_wrapper(dataset, self.image_size,
                                          augment_horizontal_flip=False, info_color='magenta', min1to1=False,
                                          **dataset_kwargs)
            dataLoader_fid = DataLoader(dataSet_fid, batch_size=self.fid_batch_size, num_workers=num_workers)

            self.fid_scorer = FID(self.fid_batch_size, dataLoader_fid, dataset_name=exp_name, device=self.device,
                                  no_label=os.path.isdir(dataset))

            print(colored('FID score will be calculated with the following sampler(s)', 'magenta'))
            if self.ddpm_fid_flag:
                self.ddpm_num_fid_samples = ddpm_num_fid_samples if ddpm_num_fid_samples is not None else len(dataSet)
                print(colored('-> DDPM Sampler / FID calculation every {} steps with {} generated samples'
                              .format(self.ddpm_fid_score_estimate_every, self.ddpm_num_fid_samples), 'magenta'))
            for sampler in self.ddim_samplers:
                if sampler.calculate_fid:
                    print(colored('-> {} / FID calculation every {} steps with {} generated samples'
                                  .format(sampler.sampler_name, sampler.sample_every,
                                          sampler.num_fid_sample), 'magenta'))
            print('\n')
            if self.ddpm_fid_flag:
                self.tqdm_sampler_name = 'DDPM'
                self.fid_score_log['DDPM'] = list()
                notification = make_notification('WARNING', color='red', boundary='*')
                print(notification)
                msg = """
                FID computation witm DDPM sampler requires a lot of generated samples and can therefore be very time 
                consuming.\nTo accelerate sampling, only using DDIM sampling is recommended. To disable DDPM sampling,
                set [ddpm_fid_score_estimate_every] parameter to None while instantiating Trainer.\n
                """
                print(colored(msg, 'red'))
            del dataLoader_fid
            del dataSet_fid

    def train(self):
        notification = make_notification('Training', color='yellow', boundary='+')
        print(notification)
        cur_fid = 'NAN'
        ddpm_best_fid = 1e10
        stepTQDM = tqdm(range(self.global_step, self.total_step))

        running_loss = 0.0
        running_count = 0
        last_avg_loss = None
        for cur_step in stepTQDM:
            self.diffusion_model.train()
            self.optimizer.zero_grad()

            batch = next(self.dataLoader)

            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                image, sample_id = batch                  # (B,1,H,W), (B,)
                image = image.to(self.device, non_blocking=True).float()
                sample_ids = [int(s.item()) for s in sample_id]
            else:
                image = batch.to(self.device, non_blocking=True).float()
                B = image.shape[0]
                sample_ids = [
                    int(hash(image[i].detach().cpu().numpy().tobytes()) & ((1 << 63) - 1))
                    for i in range(B)
                ]

            B, C, H, W = image.shape
            assert C == 1, f"Expected single-channel NS image, got {C}"

            cond_masks, target_masks = self.sparsity_controller.get_masks(B, C, sample_ids)
            cond_mask = torch.stack(cond_masks).to(self.device)
            target_mask = torch.stack(target_masks).to(self.device)
            sparse_input = (image * cond_mask).to(self.device)

            # No Perceiver reconstruction anymore
            recon = None

            if cur_step < 3:
                print(f"[Step {cur_step}] Mode = {self.sparsity_controller.mode}, Pattern = {self.sparsity_controller.pattern}")
                print(f"Mean cond_mask: {cond_mask[0,0].mean():.4f}, target_mask: {target_mask[0,0].mean():.4f}")

                os.makedirs('./debug', exist_ok=True)
                viz = torch.cat([
                    (image[:8] + 1) / 2,            # GT
                    (sparse_input[:8] + 1) / 2,     # masked input
                    cond_mask[:8],                   # cond mask
                    target_mask[:8],                 # target mask
                ], dim=0)
                grid = make_grid(viz, nrow=8)
                save_image(grid, f'./debug/composite_step{cur_step}.png')

            loss = self.diffusion_model(
                image,
                sparse_input=sparse_input,
                perceiver_input=None,      # ← removed Perceiver guidance
                mask=cond_mask,
                loss_mask=target_mask
            )

            running_loss += loss.item()
            running_count += 1

            loss.backward()
            nn.utils.clip_grad_norm_(self.diffusion_model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            if running_count % 750 == 0:
                last_avg_loss = running_loss / running_count
                running_loss = 0.0
                running_count = 0

            vis_fid = cur_fid if isinstance(cur_fid, str) else f'{cur_fid:.04f}'
            postfix = {
                'loss': f'{loss.item():.04f}',
                'FID': vis_fid,
                'step': self.global_step
            }
            if last_avg_loss is not None:
                postfix['avg_loss'] = f'{last_avg_loss:.6f}'
            stepTQDM.set_postfix(postfix)

            self.diffusion_model.eval()
            # ===== DDPM Sampler for generating images =====
            if cur_step != 0 and (cur_step % self.save_and_sample_every) == 0:
                if self.writer is not None:
                    self.writer.add_scalar('Loss', loss.item(), cur_step)
                with torch.inference_mode():
                    batches = num_to_groups(self.num_samples, self.batch_size)
                    for i, j in zip([True, False], ['clip', 'no_clip']):
                        if self.clip not in [i, 'both']:
                            continue

                        imgs = []
                        start_idx = 0
                        for n in batches:
                            mask_batch   = cond_mask[start_idx:start_idx+n]
                            image_batch  = image[start_idx:start_idx+n]
                            sparse_batch = mask_batch * image_batch

                            sampled = self.diffusion_model.sample(
                                batch_size=n,
                                sparse_input=sparse_batch,
                                perceiver_input=None,   # ← removed Perceiver guidance
                                mask=mask_batch,
                                clip=i
                            )
                            imgs.append(sampled)
                            start_idx += n

                        imgs = torch.cat(imgs, dim=0)
                        save_image(imgs, nrow=self.nrow,
                                   fp=os.path.join(self.ddpm_result_folder, j, f'sample_{cur_step}.png'))
                        if self.writer is not None:
                            self.writer.add_images(f'DDPM sampling result ({j})', imgs, cur_step)

                        # Simple colormapped composite (no perceiver_vis)
                        def apply_colormap(tensor, cmap_name='viridis'):
                            import matplotlib.pyplot as plt
                            cmap = plt.get_cmap(cmap_name)
                            tensor_np = tensor.detach().cpu().numpy()
                            tensor_np = (tensor_np - tensor_np.min()) / (tensor_np.max() - tensor_np.min() + 1e-8)
                            rgb_list = []
                            for img in tensor_np:
                                img_2d = img[0]
                                rgba_img = cmap(img_2d)
                                rgb_img = rgba_img[..., :3]
                                rgb_list.append(torch.from_numpy(rgb_img).permute(2, 0, 1))
                            return torch.stack(rgb_list, dim=0)

                        image_vis  = apply_colormap(image[:8])
                        sparse_vis = apply_colormap(sparse_input[:8])
                        mask_vis   = apply_colormap(cond_mask[:8])
                        imgs_vis   = apply_colormap(imgs[:8])
                        target_vis = apply_colormap(target_mask[:8])

                        viz = torch.cat([
                            image_vis,
                            sparse_vis,
                            mask_vis,
                            imgs_vis,
                            target_vis,
                        ], dim=0)

                        grid = make_grid(viz, nrow=self.nrow)
                        save_image(grid, os.path.join(self.ddpm_result_folder, j, f'composite_{cur_step}_viridis.png'))

                self.save('latest')

            # ===== DDPM Sampler for FID score evaluation =====
            if self.ddpm_fid_flag and cur_step != 0 and (cur_step % self.ddpm_fid_score_estimate_every) == 0:
                ddpm_cur_fid, _ = self.fid_scorer.fid_score(self.diffusion_model.sample, self.ddpm_num_fid_samples)
                if ddpm_best_fid > ddpm_cur_fid:
                    ddpm_best_fid = ddpm_cur_fid
                    self.save('best_fid_ddpm')
                if self.writer is not None:
                    self.writer.add_scalars('FID', {'DDPM': ddpm_cur_fid}, cur_step)
                cur_fid = ddpm_cur_fid
                self.fid_score_log['DDPM'].append((self.global_step, ddpm_cur_fid))

            # ===== DDIM Sampler =====
            for sampler in self.ddim_samplers:
                if cur_step != 0 and (cur_step % sampler.sample_every) == 0:
                    # Image generation
                    if sampler.generate_image:
                        with torch.inference_mode():
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            c_batch = np.insert(np.cumsum(np.array(batches)), 0, 0)
                            for i, j in zip([True, False], ['clip', 'no_clip']):
                                if sampler.clip not in [i, 'both']:
                                    continue
                                if sampler.fixed_noise:
                                    imgs = []
                                    for b in range(len(batches)):
                                        imgs.append(sampler.sample(self.diffusion_model, batch_size=None, clip=i,
                                                                   noise=sampler.noise[c_batch[b]:c_batch[b+1]]))
                                else:
                                    imgs = list(map(lambda n: self.diffusion_model.sample(
                                        batch_size=n,
                                        sparse_input=sparse_input[:n],
                                        perceiver_input=None,  # ← removed Perceiver guidance
                                        mask=cond_mask[:n],
                                        clip=i
                                    ), batches))
                                imgs = torch.cat(imgs, dim=0)
                                save_image(imgs, nrow=self.nrow,
                                           fp=os.path.join(sampler.save_path, j, f'sample_{cur_step}.png'))
                                if self.writer is not None:
                                    self.writer.add_images('{} sampling result ({})'
                                                           .format(sampler.sampler_name, j), imgs, cur_step)

                    # FID evaluation
                    if sampler.calculate_fid:
                        sample_ = lambda batch_size, clip=True, min1to1=False: sampler.sample(
                            self.diffusion_model,
                            batch_size=batch_size,
                            clip=clip,
                            min1to1=min1to1,
                            sparse_input=torch.zeros(batch_size, self.diffusion_model.channel, self.image_size, self.image_size, device=self.device),
                            perceiver_input=None,  # ← removed Perceiver guidance
                            mask=torch.zeros(batch_size, self.diffusion_model.channel, self.image_size, self.image_size, device=self.device)
                        )
                        ddim_cur_fid, _ = self.fid_scorer.fid_score(sample_, sampler.num_fid_sample)
                        if sampler.best_fid[0] > ddim_cur_fid:
                            sampler.best_fid[0] = ddim_cur_fid
                            if sampler.save:
                                self.save('best_fid_{}'.format(sampler.sampler_name))
                        if sampler.sampler_name == self.tqdm_sampler_name:
                            cur_fid = ddim_cur_fid
                        if self.writer is not None:
                            self.writer.add_scalars('FID', {sampler.sampler_name: ddim_cur_fid}, cur_step)
                        self.fid_score_log[sampler.sampler_name].append((self.global_step, ddim_cur_fid))

            self.global_step += 1

        print(colored('Training Finished!', 'yellow'))
        if self.writer is not None:
            self.writer.close()

    def save(self, name):
        data = {
            'global_step': self.global_step,
            'model': self.diffusion_model.state_dict(),
            'opt': self.optimizer.state_dict(),
            'fid_logger': self.fid_score_log,
            'tensorboard': self.tensorboard_name
        }
        for sampler in self.ddim_samplers:
            data[sampler.sampler_name] = sampler.state_dict()
        torch.save(data, os.path.join(self.result_folder, 'model_{}.pt'.format(name)))

    def load(self, path, tensorboard_path=None, no_prev_ddim_setting=False):
        if not os.path.exists(path):
            print(make_notification('ERROR', color='red', boundary='*'))
            print(colored('No saved checkpoint is detected. Please check you gave existing path!', 'red'))
            exit()
        if tensorboard_path is not None and not os.path.exists(tensorboard_path):
            print(make_notification('ERROR', color='red', boundary='*'))
            print(colored('No tensorboard is detected. Please check you gave existing path!', 'red'))
            exit()
        print(make_notification('Loading Checkpoint', color='green'))
        data = torch.load(path, map_location=self.device)
        self.diffusion_model.load_state_dict(data['model'])
        self.global_step = data['global_step']
        self.optimizer.load_state_dict(data['opt'])
        fid_score_log = data['fid_logger']
        if no_prev_ddim_setting:
            for key, val in self.fid_score_log.items():
                if key not in fid_score_log:
                    fid_score_log[key] = val
        else:
            for sampler in self.ddim_samplers:
                sampler.load_state_dict(data[sampler.sampler_name])
        self.fid_score_log = fid_score_log
        if tensorboard_path is not None:
            self.tensorboard_name = data['tensorboard']
        print(colored('Successfully loaded checkpoint!\n', 'green'))
