Param(
  [string]$ZipPath = "C:\Users\UCCSMS\Documents\GitHub\RedWavesTech\ACADEMIC\MPhil Data Science (M&A)\DMA818 MACHINE LEARNING\DMA818 ProjectWork\dma818-boston-housing-repo.darkbadge.zip",
  [string]$DestPath = "C:\Users\UCCSMS\Documents\GitHub\RedWavesTech\ACADEMIC\MPhil Data Science (M&A)\DMA818 MACHINE LEARNING\DMA818 ProjectWork",
  [string]$Remote = "https://github.com/eskgyimah/DMA818-BostonHousing.git",
  [string]$Branch = "main",
  [string]$HerokuApp = "dma818-boston-demo",
  [switch]$RunStreamlit = $true
)

[Console]::OutputEncoding = [Text.UTF8Encoding]::UTF8
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

Write-Host "==> Unzipping repo..."
if (!(Test-Path -LiteralPath $DestPath)) { New-Item -ItemType Directory -Path $DestPath -Force | Out-Null }
Expand-Archive -LiteralPath $ZipPath -DestinationPath $DestPath -Force

$RepoRoot = Join-Path -Path $DestPath -ChildPath "dma818-boston-housing-repo"
if (!(Test-Path -LiteralPath $RepoRoot)) {
  $RepoRoot = (Get-ChildItem -LiteralPath $DestPath -Directory | Where-Object { $_.Name -like "dma818-boston-housing-repo*" } | Select-Object -First 1).FullName
}

Write-Host "==> Repo root: $RepoRoot"
Set-Location -LiteralPath $RepoRoot

# --- Patch workflows: pin Heroku app name ---
$DeployWf = ".github\workflows\deploy-heroku.yml"
if (Test-Path $DeployWf) {
  $content = Get-Content -Raw -LiteralPath $DeployWf
  $content = [Regex]::Replace($content, "heroku_app_name:\s*\${.*?}", "heroku_app_name: $HerokuApp")
  $content = $content -replace "Deploy to Heroku", "Deploy Streamlit to Heroku"
  Set-Content -LiteralPath $DeployWf -Value $content -Encoding UTF8
} else {
  New-Item -ItemType Directory -Force -Path ".github\workflows" | Out-Null
  Set-Content -LiteralPath $DeployWf -Encoding UTF8 -Value @"
name: Deploy Streamlit to Heroku
on: [push]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - name: Prepare requirements.txt
        run: |
          if [ -f requirements_streamlit.txt ]; then cp requirements_streamlit.txt requirements.txt; fi
      - name: Deploy Streamlit to Heroku
        uses: akhileshns/heroku-deploy@v4.1.7
        with:
          heroku_api_key: ${{ secrets.HEROKU_API_KEY }}
          heroku_app_name: dma818-boston-demo
          heroku_email: ${{ secrets.HEROKU_EMAIL }}
          usedocker: false
"@
}

$PreviewWf = ".github\workflows\deploy-preview.yml"
if (Test-Path $PreviewWf) {
  $content = Get-Content -Raw -LiteralPath $PreviewWf
  $content = [Regex]::Replace($content, "PREVIEW_APP:\s*\${.*}", "PREVIEW_APP: $HerokuApp-pr-${ github.event.number }")
  Set-Content -LiteralPath $PreviewWf -Value $content -Encoding UTF8
}

# --- Add app.json for Heroku Deploy button ---
$appJson = @"
{
  "name": "dma818-boston-demo",
  "description": "DMA818 — Boston Housing Streamlit app",
  "repository": "https://github.com/eskgyimah/DMA818-BostonHousing",
  "keywords": ["streamlit", "scikit-learn", "python", "ml"],
  "buildpacks": [{ "url": "heroku/python" }],
  "env": { "PYTHON_VERSION": { "description": "Python runtime", "value": "3.11.x" } },
  "stack": "heroku-22"
}
"@
Set-Content -LiteralPath "app.json" -Value $appJson -Encoding UTF8

# --- Ensure README has Deploy button ---
$readme = "README.md"
$buttonLine = "[![Deploy to Heroku](https://www.herokucdn.com/deploy/button.svg)](https://heroku.com/deploy?template=https://github.com/eskgyimah/DMA818-BostonHousing)"
if (Test-Path $readme) {
  $md = Get-Content -Raw -LiteralPath $readme
  if ($md -notmatch "herokucdn\.com/deploy/button\.svg") {
    $md = $buttonLine + "`n" + $md
    Set-Content -LiteralPath $readme -Value $md -Encoding UTF8
  }
}

# --- Init git & push if needed ---
if (!(Test-Path .git)) {
  git init
  git add .
  git commit -m "Initial commit: DMA818 Boston Housing (report, pipeline, app, CI/CD)"
  git branch -M $Branch
  git remote add origin $Remote
  git push -u origin $Branch
}

# --- Python env & pipeline ---
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
if (Test-Path requirements_streamlit.txt) { pip install -r requirements_streamlit.txt }

python dma818_pipeline.py --data BostonHousing.csv --out out

if ($RunStreamlit) {
  streamlit run streamlit_app.py
}
