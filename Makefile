.PHONY: install clean experiments

install:
	pip install -r requirements.txt && \
	cd voxelmorph && \
	pip install -e . && \
	cd .. && \
	mkdir figures && \
	mkdir experiments && \
	mkdir models

clean:
	if [ -d "experiments/debug" ]; then \
		rm -r experiments/debug ; \
	fi
	
experiment: experiments
	mkdir experiments/debug
	python registration/main.py --output experiments/debug --debug ; 

experiment-%: experiments
	mkdir experiments/$*
	if [-z "$(DEBUG)"]; then \
		python registration/main.py --output experiments/$* --debug ; \
	else \
		python registration/main.py --output experiments/$* ; \
	fi