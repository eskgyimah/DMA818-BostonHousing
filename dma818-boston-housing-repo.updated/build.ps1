Param(
  [string]$Data = "BostonHousing.csv",
  [string]$Out = "out"
)

Write-Host "==> Running pipeline"
python dma818_pipeline.py --data $Data --out $Out

Write-Host "==> Rendering Quarto (if installed)"
quarto render index.qmd --to html 2>$null
quarto render index.qmd --to pdf  2>$null

Write-Host "==> Building LaTeX (if pdflatex available)"
pdflatex -interaction=nonstopmode DMA818_BostonHousing_Report.tex 2>$null
