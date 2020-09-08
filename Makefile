#.PHONY: docs  # necessary so it doesn't look for 'docs/makefile html'

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

.PHONY: docs
docs:
	make -C docs html


#docs:
#	@echo "Updating documentation..."
#
#	@echo "(1) Update version from pyproject.toml and pkg/__version__.py"
#	poetry run keats version up
#	poetry run keats changelog up
#	#cp .keats/changelog.md docs/source/changelog.md
#
#	@echo "(2) Building documentation"
#	rm -rf ./docs/builds
#	cd docs && poetry run make html
#	find docs -type f -exec chmod 444 {} \;
#	@echo "\033[95m\n\nBuild successful! View the docs homepage at docs/html/index.html.\n\033[0m"
#
#	@echo "(3) Clean up"
#	touch docs/build/.nojekyll
#	open ./docs/build/index.html
