CC = gcc
COPT = -lm -fPIC
PYPATH = /usr/include/python2.6
PROJ_PATH = /home/leo/src/inversion/trunk
EXTMOD_PATH = sloot
BUILD_PATH = build
TEST_PATH = test
UNITTESTS = \
	${TEST_PATH}/test_lu_decomp.py \
	${TEST_PATH}/test_lu_decomp_pivot.py \
	${TEST_PATH}/test_solve_linsys_lu.py
TESTDATA = ${TEST_PATH}/linsys-data/data1.txt


all: ${TEST_PATH}/regression.py build_ext_mods
	@echo "\n**** RUNNING TESTS ****"
	python $< -v
	
	
# BUILD THE EXTENTION MODULES
build_ext_mods: setup.py c/prismgrav_wrap.c c/linalg_wrap.c
	@echo "\n**** BUILDING EXTENTION MODULES ****"
	python setup.py build_ext --inplace

# WRAP THE C-CODED LIBS
c/prismgrav_wrap.c: c/prismgrav.i c/prismgrav.c
	@echo "\n**** WRAPPING PRISMGRAV.C ****"
	swig -python -outdir ${EXTMOD_PATH} $< 
	
c/linalg_wrap.c: c/linalg.i c/linalg.c
	@echo "\n**** WRAPPING LINALG.C ****"
	swig -python -outdir ${EXTMOD_PATH} $< 
	
	
# CLEAN UP
clean: clean-build clean-wraps clean-extmods

clean-build:	
	@echo "Removing build path..."
	rm -f -r ${BUILD_PATH}
	
clean-wraps:
	@echo "Removing SWIG wraps..."
	rm -f c/prismgrav_wrap.c \
		c/linalg_wrap.c
	
clean-extmods:
	@echo "Removing extention modules..."
	rm -f ${EXTMOD_PATH}/*linalg.* \
		${EXTMOD_PATH}/*prismgrav.* 
	
# RUN TESTS
test:  ${TESTDATA}
	@echo "\n**** RUNNING TESTS ****"
	python ${TEST_PATH}/regression.py -v
	
# GENERATE THE TEST DATA
${TESTDATA}: ${TEST_PATH}/linsys-gen.py
	@echo "\n**** GENERATING TEST DATA ****"
	@echo "Clear the old data..."
	rm -f ${TESTDATA}
	@if [ ! -d ${TEST_PATH}/linsys-data ]; then echo "\nCreate linsys-data dir...";	mkdir ${TEST_PATH}/linsys-data; fi
	cd ${TEST_PATH} && python linsys-gen.py





#${EXTMOD_PATH}/_prismgrav.so: ${BUILD_PATH}/prismgrav_wrap.o ${BUILD_PATH}/prismgrav.o
#	ld -shared $^ -o $@

#${BUILD_PATH}/prismgrav.o: c/prismgrav.c
#	${CC} -c $< ${COPT}	-o $@
	
#${BUILD_PATH}/prismgrav_wrap.o: c/prismgrav_wrap.c
#	${CC} -c $< -I${PYPATH} ${COPT} -o $@
