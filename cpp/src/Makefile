################################################################################
# Makefile for project Fatiando O Mundo.
# $Revision $
# Last edited: $Date $
# Edited by: $Author $
################################################################################


# Environment
COMP_OPT=-Wall -ansi -g2


# Build for Linux
build: log.o


log.o: log.cpp headers/log.h
	g++ -o log.o ${COMP_OPT} -c log.cpp

# Build for Windows
win:


# Testing programs
test: test_log
	./test_log

test_log: test_log.o log.o
	g++ -o test_log ${COMP_OPT} test_log.o log.o

test_log.o: test_log.cpp
	g++ -o test_log.o ${COMP_OPT} -c test_log.cpp


# clean
clean:
	rm *.o  *.log
		
# all
all:

# help
help: 