# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) s.

# Put it first so that "make" without argument is like "make help".
.PHONY: help Makefile
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: clean
clean:
	@echo "Cleaning"
	rm -rf $(BUILDDIR)/*
	keats version up
	keats changelog up
	rm -rf $(SOURCEDIR)/generated

# execute each notebook and generate files for each notebook
.PHONY: nbexamples
nbexamples:
	python _tools/nb_to_doc.py -f ../examples/traversals/shortest_path.ipynb -o source/narratives

# execute python based tests
.PHONY: tests
tests:
	@cd _tools && python -m pytest

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@touch $(BUILDDIR)/html/.nojekyll

