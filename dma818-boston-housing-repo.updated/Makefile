.PHONY: all pipeline quarto latex clean

all: pipeline quarto

pipeline:
	python dma818_pipeline.py --data BostonHousing.csv --out out

quarto:
	@echo "Rendering Quarto (requires Quarto installed)"
	quarto render index.qmd --to html || true
	quarto render index.qmd --to pdf || true

latex:
	@echo "Building LaTeX PDF (requires pdflatex)"
	pdflatex -interaction=nonstopmode DMA818_BostonHousing_Report.tex || true

clean:
	rm -rf out _output *.log *.aux *.out *.toc
