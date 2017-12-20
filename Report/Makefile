PDFs = report.pdf

all: $(PDFs)

clean:
	rm *.log *.aux

realclean:
	rm $(PDFs)

%.pdf: %.tex
	pdflatex $<
	pdflatex $<
