NAME=apostila

${NAME}.pdf: ${NAME}.tex ${NAME}.bib latex/*.tex
	pdflatex ${NAME}
	bibtex ${NAME}
	pdflatex ${NAME}
	pdflatex ${NAME}

clean:
	rm ${NAME}.log ${NAME}.toc ${NAME}.pdf ${NAME}.aux ${NAME}.out ${NAME}.blg ${NAME}.bbl 
