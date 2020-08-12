install:
	echo "Make sure your activated in your conda environment"
	poetry export --dev -f requirements.txt > requirements.txt
	pip install -r requirements.txt > requirements.txt
	conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
	CUDA="cu101"
	pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.6.0.html
	rm requirements.txt