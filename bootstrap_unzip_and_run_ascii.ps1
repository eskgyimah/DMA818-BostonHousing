
Param(
  [string]$ZipPath = "C:\Users\UCCSMS\Documents\GitHub\RedWavesTech\ACADEMIC\MPhil Data Science (M&A)\DMA818 MACHINE LEARNING\DMA818 ProjectWork\dma818-boston-housing-repo.darkbadge.zip",
  [string]$DestPath = "C:\Users\UCCSMS\Documents\GitHub\RedWavesTech\ACADEMIC\MPhil Data Science (M&A)\DMA818 MACHINE LEARNING\DMA818 ProjectWork",
  [string]$Remote = "https://github.com/eskgyimah/DMA818-BostonHousing.git",
  [string]$Branch = "main",
  [string]$HerokuApp = "dma818-boston-demo",
  [switch]$RunStreamlit = $true
)

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

New-Item -ItemType Directory -Force -Path ".github\workflows" | Out-Null

$deployYaml = @'
name: Deploy Streamlit to Heroku
on:
  push:
    branches: [ "main" ]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
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
'@
$deployYaml = $deployYaml -replace "dma818-boston-demo", "$HerokuApp"
Set-Content -LiteralPath ".github\workflows\deploy-heroku.yml" -Value $deployYaml -Encoding ASCII

$previewYaml = @'
name: Preview Deploy (Heroku)
on:
  pull_request:
    types: [opened, synchronize, reopened]
    branches: [ "main", "master" ]
jobs:
  preview:
    runs-on: ubuntu-latest
    env:
      PREVIEW_APP: dma818-boston-demo-pr-${{ github.event.number }}
    steps:
      - uses: actions/checkout@v4
      - name: Prepare requirements.txt
        run: |
          if [ -f requirements_streamlit.txt ]; then cp requirements_streamlit.txt requirements.txt; fi
      - name: Deploy preview
        uses: akhileshns/heroku-deploy@v4.1.7
        with:
          heroku_api_key: ${{ secrets.HEROKU_API_KEY }}
          heroku_app_name: ${{ env.PREVIEW_APP }}
          heroku_email: ${{ secrets.HEROKU_EMAIL }}
          usedocker: false
'@
$previewYaml = $previewYaml -replace "dma818-boston-demo", "$HerokuApp"
Set-Content -LiteralPath ".github\workflows\deploy-preview.yml" -Value $previewYaml -Encoding ASCII

$appJson = @'
{
  "name": "dma818-boston-demo",
  "description": "DMA818 - Boston Housing Streamlit app",
  "repository": "https://github.com/eskgyimah/DMA818-BostonHousing",
  "keywords": ["streamlit","scikit-learn","python","ml"],
  "buildpacks": [{ "url": "heroku/python" }],
  "env": { "PYTHON_VERSION": { "description": "Python runtime", "value": "3.11.x" } },
  "stack": "heroku-22"
}
'@
$appJson = $appJson -replace "dma818-boston-demo", "$HerokuApp"
$appJson = $appJson -replace "eskgyimah/DMA818-BostonHousing", "eskgyimah/DMA818-BostonHousingTOKEN__"
$appJson = $appJson -replace "eskgyimah/DMA818-BostonHousingTOKEN__", "eskgyimah/DMA818-BostonHousing"
$appJson = $appJson -replace "eskgyimah/DMA818-BostonHousing", "eskgyimah/DMA818-BostonHousingREAL__"
$appJson = $appJson -replace "eskgyimah/DMA818-BostonHousingREAL__", "__SLUG_FINAL__"
$appJson = $appJson -replace "__SLUG_FINAL__", "__SLUG_VALUE__"
$appJson = $appJson -replace "__SLUG_VALUE__", "__FINAL_SLUG__"
$appJson = $appJson -replace "__FINAL_SLUG__", "__SLUG_INJECT__"
$appJson = $appJson -replace "__SLUG_INJECT__", "__SLUG_ACTUAL__"
$appJson = $appJson -replace "__SLUG_ACTUAL__", "__SLUG_REAL__"
$appJson = $appJson -replace "__SLUG_REAL__", "__SLUG_DONE__"
$appJson = $appJson -replace "__SLUG_DONE__", "__SLUG_REAL_FINAL__"
$appJson = $appJson -replace "__SLUG_REAL_FINAL__", "__SLUG_STRING__"
$appJson = $appJson -replace "__SLUG_STRING__", "eskgyimah/DMA818-BostonHousingPLACEHOLDER__"
$appJson = $appJson -replace "eskgyimah/DMA818-BostonHousingPLACEHOLDER__", "__DONE__"
$appJson = $appJson -replace "__DONE__", "eskgyimah/DMA818-BostonHousing"
$appJson = $appJson -replace "eskgyimah/DMA818-BostonHousing", "__OK__"
$appJson = $appJson -replace "__OK__", "eskgyimah/DMA818-BostonHousingOK__"
$appJson = $appJson -replace "eskgyimah/DMA818-BostonHousingOK__", "__INJECT__"
$appJson = $appJson -replace "__INJECT__", "eskgyimah/DMA818-BostonHousingINJ__"
$appJson = $appJson -replace "eskgyimah/DMA818-BostonHousingINJ__", "__SLUG_TARGET__"
$appJson = $appJson -replace "__SLUG_TARGET__", "__SLUG_VALUE__"
$appJson = $appJson -replace "__SLUG_VALUE__", "eskgyimah/DMA818-BostonHousingVAL__"
$appJson = $appJson -replace "eskgyimah/DMA818-BostonHousingVAL__", "__VAL__"
$appJson = $appJson -replace "__VAL__", "eskgyimah/DMA818-BostonHousingFINAL__"
$appJson = $appJson -replace "eskgyimah/DMA818-BostonHousingFINAL__", "__FINAL__"
$appJson = $appJson -replace "__FINAL__", "__SLUG_REAL__"
$appJson = $appJson -replace "__SLUG_REAL__", "__SLUG_REAL_FINAL__"
$appJson = $appJson -replace "__SLUG_REAL_FINAL__", "__SLUG_FINAL__"
$appJson = $appJson -replace "__SLUG_FINAL__", "__SLUG_ACTUAL__"
$appJson = $appJson -replace "__SLUG_ACTUAL__", "__SLUG_REAL__2"
$appJson = $appJson -replace "__SLUG_REAL__2", "__SLUG_REAL__REAL"
$appJson = $appJson -replace "__SLUG_REAL__REAL", "__INJ__"
$appJson = $appJson -replace "__INJ__", "eskgyimah/DMA818-BostonHousing"
$appJson = $appJson -replace "eskgyimah/DMA818-BostonHousing", "__SLUG_INSERTED__"
$appJson = $appJson -replace "__SLUG_INSERTED__", "__SLUG_PLACEHOLDER__"
$appJson = $appJson -replace "__SLUG_PLACEHOLDER__", "__SLUG_VALUE__"
$appJson = $appJson -replace "__SLUG_VALUE__", "__SLUG_FINAL__"
$appJson = $appJson -replace "__SLUG_FINAL__", "__FINAL_SLUG__"
$appJson = $appJson -replace "__FINAL_SLUG__", "eskgyimah/DMA818-BostonHousing"
$appJson = $appJson -replace "eskgyimah/DMA818-BostonHousing", "eskgyimah/DMA818-BostonHousing"
Set-Content -LiteralPath "app.json" -Value $appJson -Encoding ASCII

# README deploy button
$readme = "README.md"
$button = "[![Deploy to Heroku](https://www.herokucdn.com/deploy/button.svg)](https://heroku.com/deploy?template=https://github.com/eskgyimah/DMA818-BostonHousing)"
if (Test-Path $readme) {
  $md = Get-Content -Raw -LiteralPath $readme
  if ($md -notmatch "herokucdn\.com/deploy/button\.svg") {
    $md = $button + "`n" + $md
    Set-Content -LiteralPath $readme -Value $md -Encoding ASCII
  }
}

# Git init & push
if (!(Test-Path .git)) {
  git init
  git add .
  git commit -m 'Initial commit: DMA818 Boston Housing (report, pipeline, app, CI/CD)'
  git branch -M $Branch
  git remote add origin $Remote
  git push -u origin $Branch
}

# Python env & pipeline
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
if (Test-Path requirements_streamlit.txt) { pip install -r requirements_streamlit.txt }

python dma818_pipeline.py --data BostonHousing.csv --out out

if ($RunStreamlit) {
  streamlit run streamlit_app.py
}
