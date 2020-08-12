init:
	conda env create -f environment.yml

lock:
	conda env export > environment.yml

activate:
	conda activate caldera

check:
	echo "Python environment"
	which python
	python check.py