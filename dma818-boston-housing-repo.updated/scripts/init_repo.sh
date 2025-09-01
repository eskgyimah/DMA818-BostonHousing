#!/usr/bin/env bash
set -euo pipefail

REMOTE="${1:-https://github.com/eskgyimah/DMA818-BostonHousing.git}"
BRANCH="${2:-main}"

if ! command -v git &>/dev/null; then
  echo "git not found. Install Git." >&2
  exit 1
fi

echo "==> Initializing repo"
git init
git add .
git commit -m "Initial commit: DMA818 Boston Housing (report, pipeline, app, CI/CD)" || true

# Optional: Git LFS
if command -v git-lfs &>/dev/null; then
  git lfs install
  git lfs track "*.pdf" "*.docx" "*.pptx" "*.png" "*.zip"
  git add .gitattributes || true
  git commit -m "chore: enable Git LFS" || true
fi

git branch -M "$BRANCH"
git remote add origin "$REMOTE" || true
echo "==> Pushing to $REMOTE ($BRANCH)"
git push -u origin "$BRANCH"
