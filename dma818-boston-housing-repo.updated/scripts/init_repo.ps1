Param(
  [string]$Remote = "https://github.com/eskgyimah/DMA818-BostonHousing.git",
  [string]$Branch = "main"
)

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
  Write-Error "git not found. Install Git first."
  exit 1
}

Write-Host "==> Initializing repo"
git init
git config branch.$Branch.mergeOptions "--no-ff"
git add .
git commit -m "Initial commit: DMA818 Boston Housing (report, pipeline, app, CI/CD)"

# Optional: Git LFS
if (Get-Command git -ErrorAction SilentlyContinue) {
  git lfs install 2>$null
  git lfs track "*.pdf" "*.docx" "*.pptx" "*.png" "*.zip" 2>$null
  git add .gitattributes 2>$null
  git commit -m "chore: enable Git LFS" 2>$null
}

git branch -M $Branch
git remote add origin $Remote

Write-Host "==> Pushing to $Remote ($Branch)"
git push -u origin $Branch
