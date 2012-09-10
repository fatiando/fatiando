SCRIPT=../../../scripts

all: data.txt Harvestfile.py
	python $(SCRIPT)/harvester -v

data.txt: datagen.py
	python datagen.py

clean:
	@rm -vf data.txt predicted.txt result.den result.msh result.pickle
