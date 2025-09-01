#!/usr/bin/env bash
set -euo pipefail
NAME="${1:-DMA818-BostonHousing}"
VIS="${2:-private}"
if ! command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI (gh) not found. See https://cli.github.com/" >&2
  exit 1
fi
if [ "$VIS" = "public" ]; then PRIV="--public"; else PRIV="--private"; fi
gh repo create "$NAME" $PRIV --source "." --remote "origin" --push
