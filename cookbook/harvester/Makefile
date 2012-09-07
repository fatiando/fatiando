SCRIPT=../../scripts

all: data.txt Harvestfile.py
	python $(SCRIPT)/harvester -v

data.txt: datagen.py
	python datagen.py

clean:
	rm -f data.txt predicted.txt result.den result.msh result.pickle
