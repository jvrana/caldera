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

docs:
	@echo "Updating docs"

	rm -rf docs
	cd docsrc && poetry run make html
	find docs -type f -exec chmod 444 {} \;
	@echo "\033[95m\n\nBuild successful! View the docs homepage at docs/html/index.html.\n\033[0m"

	touch docs/.nojekyll
	open ./docs/index.html
