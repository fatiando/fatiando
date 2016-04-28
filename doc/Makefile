# Makefile for Sphinx documentation

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
BUILDDIR      = _build
CBDIR         = _static/cookbook

# User-friendly check for sphinx-build
ifeq ($(shell which $(SPHINXBUILD) >/dev/null 2>&1; echo $$?), 1)
$(error The '$(SPHINXBUILD)' command was not found. Make sure you have Sphinx installed, then set the SPHINXBUILD environment variable to point to the full path of the '$(SPHINXBUILD)' executable. Alternatively you can add the directory with the executable to your PATH. If you don't have Sphinx installed, grab it from http://sphinx-doc.org/)
endif

# Internal variables.
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees  $(SPHINXOPTS) .

.PHONY: help clean html linkcheck doctest cookbook userguide

all: userguide cookbook html

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  all        generate the complete webpage"
	@echo "  html       make only the HTML files from the existing rst sources"
	@echo "  cookbook   make the cookbook recipes .rst files from the .py files"
	@echo "  linkcheck  check all external links for integrity"
	@echo "  doctest    run all doctests embedded in the documentation (if enabled)"

cookbook:
	@echo
	@echo "Building cookbook rst files."
	@echo
	mkdir -p $(CBDIR)
	cp ../cookbook/*.py $(CBDIR)
	cd $(CBDIR); python ../../tools/make_cookbook.py `ls *.py`
	rm -rf cookbook
	mkdir -p cookbook
	mv $(CBDIR)/*.rst cookbook
	@echo
	@echo "Build finished."

userguide:
	@echo
	@echo "Building userguide rst files."
	@echo
	bash tools/nb2rst.sh userguide/

clean:
	rm -rf $(BUILDDIR)/html/*
	rm -rf $(CBDIR) cookbook
	rm -rf $(BUILDDIR)/plot_directive
	rm -rf api/*.rst
	rm -rf userguide/*_files/ userguide/*.rst userguide/*.png

html:
	@echo
	@echo "Building HTML files."
	@echo
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

linkcheck:
	$(SPHINXBUILD) -b linkcheck $(ALLSPHINXOPTS) $(BUILDDIR)/linkcheck
	@echo
	@echo "Link check complete; look for any errors in the above output " \
	      "or in $(BUILDDIR)/linkcheck/output.txt."

doctest:
	$(SPHINXBUILD) -b doctest $(ALLSPHINXOPTS) $(BUILDDIR)/doctest
	@echo "Testing of doctests in the sources finished, look at the " \
	      "results in $(BUILDDIR)/doctest/output.txt."

serve:
	cd $(BUILDDIR)/html && python -m SimpleHTTPServer 8008
