"""
Master test runner for flowmatch package.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py shapes       # Run only shape tests
    python run_tests.py sampling     # Run only sampling tests
    python run_tests.py training     # Run only training tests
"""
import sys
import subprocess


def run_test_file(test_name):
    """Run a specific test file."""
    print(f"\n{'='*70}")
    print(f"Running {test_name}...")
    print(f"{'='*70}\n")

    result = subprocess.run(
        [sys.executable, f"tests/test_{test_name}.py"],
        cwd=".",
        capture_output=False
    )

    return result.returncode == 0


def main():
    test_files = {
        'shapes': 'shapes',
        'sampling': 'sampling_api',
        'training': 'train_step'
    }

    if len(sys.argv) > 1:
        # Run specific test
        test_key = sys.argv[1]
        if test_key not in test_files:
            print(f"Unknown test: {test_key}")
            print(f"Available tests: {', '.join(test_files.keys())}")
            sys.exit(1)

        success = run_test_file(test_files[test_key])
        sys.exit(0 if success else 1)

    else:
        # Run all tests
        print("\n" + "="*70)
        print("RUNNING ALL TESTS FOR FLOWMATCH PACKAGE")
        print("="*70)

        results = {}
        for name, filename in test_files.items():
            results[name] = run_test_file(filename)

        # Summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)

        for name, passed in results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{name.upper():.<40} {status}")

        all_passed = all(results.values())

        print("="*70)
        if all_passed:
            print("🎉 ALL TESTS PASSED! 🎉")
        else:
            print("❌ SOME TESTS FAILED")
        print("="*70 + "\n")

        sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
