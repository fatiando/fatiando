NAME=apostila
FILES=latex/apendice.tex latex/introducao.tex latex/leitura.tex latex/prefacio.tex latex/prob_inverso.tex latex/prob_inverso_linear.tex latex/prob_inverso_naolinear.tex latex/procedimentos_praticos.tex latex/referencias.tex latex/regularizacao.tex
${NAME}.pdf: apostila.tex ${FILES}
	pdflatex ${NAME}.tex
	pdflatex ${NAME}.tex

clean:
	rm ${NAME}.log ${NAME}.toc ${NAME}.pdf ${NAME}.aux ${NAME}.out
