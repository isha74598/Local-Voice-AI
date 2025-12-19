# GitHub Upload Guide for Cursor

This guide walks you through uploading your project to GitHub using Cursor's integrated terminal.

## Step 1: Update .gitignore ✅

The `.gitignore` file has been updated to exclude:
- Virtual environments (`.venv/`, `venv/`)
- Python cache files (`__pycache__/`, `*.pyc`)
- Environment files (`.env`)
- Node.js files (`node_modules/`, `.next/`)
- IDE files (`.vscode/`, `.idea/`)
- OS files (`.DS_Store`, `Thumbs.db`)

## Step 2: Check What Will Be Committed

```bash
git status
```

This shows:
- **M** = Modified files
- **A** = Added (new) files
- **??** = Untracked files

## Step 3: Add Files to Staging

### Option A: Add All Changes (Recommended for first commit)
```bash
git add .
```

### Option B: Add Specific Files
```bash
# Add specific files
git add agent/myagent.py
git add agent/camera_vision.py
git add agent/vision_processor.py

# Add all files in a directory
git add agent/

# Add the updated .gitignore
git add .gitignore
```

### Option C: Add All Changes Except .env Files
```bash
git add .
git restore --staged .env agent/.env
```

## Step 4: Review What's Staged

```bash
git status
```

Make sure you're not committing:
- `.env` files (sensitive data)
- `.venv/` directory (too large)
- `__pycache__/` files (generated files)

## Step 5: Commit Your Changes

```bash
git commit -m "Add camera vision capabilities to voice agent"
```

Or with a more detailed message:
```bash
git commit -m "Add camera vision capabilities

- Add CameraVision class for capturing frames
- Integrate vision processor with Ollama qwen2.5vl model
- Update agent to handle vision queries
- Change camera device from /dev/video2 to /dev/video0
- Add comprehensive .gitignore"
```

## Step 6: Push to GitHub

### First Time Setup (if not already done)
```bash
# Check if remote is set
git remote -v

# If not set, add remote:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
```

### Push to GitHub
```bash
# Push to main branch
git push origin main

# Or if your default branch is master:
git push origin master
```

### If You Get Authentication Errors
```bash
# Use GitHub CLI (if installed)
gh auth login

# Or use SSH instead of HTTPS
git remote set-url origin git@github.com:YOUR_USERNAME/YOUR_REPO.git
```

## Step 7: Verify on GitHub

1. Go to your repository on GitHub
2. Check that your files are uploaded
3. Verify `.env` files are NOT visible (they should be ignored)

## Common Commands Reference

```bash
# Check status
git status

# See what changed
git diff

# Unstage a file
git restore --staged <file>

# Discard local changes (careful!)
git restore <file>

# View commit history
git log --oneline

# Pull latest changes
git pull origin main

# Create a new branch
git checkout -b feature-name

# Switch branches
git checkout main
```

## Troubleshooting

### "Everything up-to-date" but you have changes
```bash
# Make sure you've added and committed
git add .
git commit -m "Your message"
git push
```

### Large files warning
If you get warnings about large files:
```bash
# Remove large files from git history (if needed)
git rm --cached large-file.mp4
git commit -m "Remove large file"
```

### Merge conflicts
```bash
# Pull latest changes first
git pull origin main

# Resolve conflicts, then:
git add .
git commit -m "Resolve merge conflicts"
git push
```

## Quick Workflow Summary

```bash
# 1. Check status
git status

# 2. Add files
git add .

# 3. Commit
git commit -m "Your commit message"

# 4. Push
git push origin main
```

That's it! Your code is now on GitHub. 🚀
