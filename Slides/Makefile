all: slides.pdf

clean:
	rm `ls slides.* | grep -e pdf -e tex -v`

realclean:
	rm slides.pdf

%.pdf: %.tex
	pdflatex $<
	pdflatex $<
