#!/bin/bash

# Script to push only Python, notebooks, and helper files to GitHub
# Repository: https://github.com/colpark/cvpr26cifar
# Maximum total size: 100MB
#
# Usage: ./push_to_github.sh [commit message]
# Example: ./push_to_github.sh "Initial commit"

set -e  # Exit on error

REPO_URL="https://github.com/colpark/cvpr26cifar.git"
BRANCH="main"
MAX_SIZE_MB=100

# Check for required commands
for cmd in git awk bc; do
    if ! command -v $cmd &> /dev/null; then
        echo "❌ ERROR: '$cmd' is required but not installed."
        echo "Please install it and try again."
        exit 1
    fi
done

echo "🚀 Pushing to GitHub repository: $REPO_URL"
echo "📁 Current directory: $(pwd)"
echo "📏 Size limit: ${MAX_SIZE_MB}MB"
echo ""

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📦 Initializing git repository..."
    git init
    git branch -M $BRANCH
else
    echo "✅ Git repository already initialized"
fi

# Check if remote exists
if git remote get-url origin &>/dev/null; then
    echo "✅ Remote 'origin' already configured"
    CURRENT_REMOTE=$(git remote get-url origin)
    if [ "$CURRENT_REMOTE" != "$REPO_URL" ]; then
        echo "⚠️  Current remote: $CURRENT_REMOTE"
        echo "⚠️  Expected remote: $REPO_URL"
        echo "Updating remote URL..."
        git remote set-url origin $REPO_URL
    fi
else
    echo "🔗 Adding remote 'origin'..."
    git remote add origin $REPO_URL
fi

# Remove all currently staged files and re-add with updated .gitignore
echo ""
echo "🔄 Removing all previously added files from git..."
git rm -r --cached . 2>/dev/null || true
git reset 2>/dev/null || true

echo ""
echo "📋 Adding files (only .py, .ipynb, and helpers)..."
git add -A

# Show what will be committed
echo ""
echo "📊 Files to be committed:"
git status --short

# Calculate total size
echo ""
echo "📏 Calculating total size..."
TOTAL_SIZE_BYTES=$(git ls-files -s | awk '{sum+=$4} END {print sum}')
TOTAL_SIZE_MB=$(echo "scale=2; $TOTAL_SIZE_BYTES/1048576" | bc)

echo "Total size: ${TOTAL_SIZE_MB}MB / ${MAX_SIZE_MB}MB"

# Check if size exceeds limit
if (( $(echo "$TOTAL_SIZE_MB > $MAX_SIZE_MB" | bc -l) )); then
    echo ""
    echo "❌ ERROR: Total size (${TOTAL_SIZE_MB}MB) exceeds ${MAX_SIZE_MB}MB limit!"
    echo ""
    echo "Largest files:"
    git ls-files -s | awk '{print $4, $2}' | sort -rn | head -20 | while read size hash; do
        file=$(git ls-files -s | grep "$hash" | awk '{print $4}')
        size_mb=$(echo "scale=2; $size/1048576" | bc)
        echo "  ${size_mb}MB - $file"
    done
    echo ""
    echo "💡 Suggestions:"
    echo "  - Remove large Python/notebook files"
    echo "  - Check if any .py/.ipynb files contain embedded data"
    echo "  - Review .gitignore to exclude more file types"
    exit 1
fi

echo "✅ Size check passed!"

# Show file type breakdown
echo ""
echo "📊 File breakdown:"
echo "Python files:"
git ls-files | grep "\.py$" | wc -l | xargs echo "  "
echo "Jupyter notebooks:"
git ls-files | grep "\.ipynb$" | wc -l | xargs echo "  "
echo "Helper files:"
git ls-files | grep -E "\.(sh|txt|md|json|yaml|yml|toml|cfg|ini|conf)$" | wc -l | xargs echo "  "

# Commit
echo ""
if [ $# -ge 1 ]; then
    # Use command line argument as commit message
    COMMIT_MSG="$*"
    echo "Using commit message from argument: $COMMIT_MSG"
else
    # Prompt for commit message
    read -p "Enter commit message (or press Enter for default): " COMMIT_MSG
    if [ -z "$COMMIT_MSG" ]; then
        COMMIT_MSG="Update project files - $(date +'%Y-%m-%d %H:%M:%S')"
    fi
fi

if git diff --cached --quiet; then
    echo "⚠️  No changes to commit - repository is up to date"
    echo "If you expect changes, check your .gitignore file"
    exit 0
fi

git commit -m "$COMMIT_MSG"

# Push to remote
echo ""
echo "⬆️  Pushing to $BRANCH branch..."

if git push -u origin $BRANCH; then
    echo ""
    echo "✅ Successfully pushed to GitHub!"
    echo "🌐 View at: https://github.com/colpark/cvpr26cifar"
    echo "📦 Total size pushed: ${TOTAL_SIZE_MB}MB"
    echo "📊 Files pushed:"
    echo "   - Python files: $(git ls-files | grep '\.py$' | wc -l | xargs)"
    echo "   - Notebooks: $(git ls-files | grep '\.ipynb$' | wc -l | xargs)"
    echo "   - Configs: $(git ls-files | grep -E '\.(yaml|yml|json|toml)$' | wc -l | xargs)"
else
    echo ""
    echo "❌ Push failed!"
    echo ""
    echo "Common issues:"
    echo "  1. Authentication: Set up SSH keys or GitHub token"
    echo "  2. Remote exists: Repository may need to be created first"
    echo "  3. Force push needed: Try 'git push -f origin $BRANCH' (CAUTION!)"
    echo "  4. Branch protected: Check repository settings"
    echo ""
    echo "To push manually:"
    echo "  git push -u origin $BRANCH"
    exit 1
fi
