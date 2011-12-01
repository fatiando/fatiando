NAME=apostila

${NAME}.pdf: ${NAME}.tex latex/*.tex
	pdflatex ${NAME}.tex
	pdflatex ${NAME}.tex

clean:
	rm ${NAME}.log ${NAME}.toc ${NAME}.pdf ${NAME}.aux ${NAME}.out
