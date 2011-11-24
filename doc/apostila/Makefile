NAME=apostila

all:
	cd latex; latex ${NAME}.tex; dvips ${NAME}.dvi; ps2pdf ${NAME}.ps

clean:
	cd latex; rm *.log *.toc *.dvi *.ps *.pdf
