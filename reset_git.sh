#!/bin/bash

# Script to remove all git tracked files and re-add only Python, notebooks, and helpers

set -e

echo "🔄 Resetting Git staging area..."
echo "This will keep ONLY .py, .ipynb, and helper files"
echo ""

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "❌ Not a git repository. Run push_to_github.sh first."
    exit 1
fi

# Remove all files from git cache (keeps files on disk)
echo "📤 Removing all files from git tracking..."
git rm -r --cached . 2>/dev/null || true

# Reset staging area
echo "🔄 Resetting staging area..."
git reset 2>/dev/null || true

# Re-add files with whitelist .gitignore rules
echo "📥 Re-adding only Python, notebooks, and helper files..."
git add -A

# Calculate total size
echo ""
echo "📏 Calculating total size..."
TOTAL_SIZE_BYTES=$(git ls-files -s | awk '{sum+=$4} END {print sum}')
TOTAL_SIZE_MB=$(echo "scale=2; $TOTAL_SIZE_BYTES/1048576" | bc)

echo "Total size: ${TOTAL_SIZE_MB}MB / 100MB"

if (( $(echo "$TOTAL_SIZE_MB > 100" | bc -l) )); then
    echo "⚠️  WARNING: Total size exceeds 100MB!"
else
    echo "✅ Size check passed!"
fi

# Show what's staged
echo ""
echo "✅ Files now staged:"
git status --short

echo ""
echo "📊 File breakdown:"
echo "Python files:"
git ls-files | grep "\.py$" | wc -l | xargs echo "  "
echo "Jupyter notebooks:"
git ls-files | grep "\.ipynb$" | wc -l | xargs echo "  "
echo "Helper files:"
git ls-files | grep -E "\.(sh|txt|md|json|yaml|yml|toml|cfg|ini|conf)$" | wc -l | xargs echo "  "

echo ""
echo "📊 Summary:"
git diff --cached --stat
