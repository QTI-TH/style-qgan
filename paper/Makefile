output = qgan.pdf
paper = qgan.tex

bibfile = qgan.bib

figures = $(wildcard plots/*/*)

LC = pdflatex
FPFLAGS = ""
LPFLAGS = ""
BC = bibtex
arxivtar = qgan.tar


.PHONY: all clean clean_latex arxiv pdf

all: $(output) # $(arxivtar)

arxiv: $(arxivtar)

pdf: $(output)

%.pdf : %.tex $(bibfile) | %.bbl
	@$(LC) $(FPFLAGS) $< > /dev/null
	@$(LC) $(LPFLAGS) $<


%.bbl : %.aux $(bibfile)
	@echo "Generating bibliography"
	@echo $<
	@$(BC) $<

%.aux : %.tex
	@echo "First pass"
	@$(LC) $(FPFLAGS) $< > /dev/null

%.tar: %.tex %.bbl
	@echo "Creating arxiv tar"
	tar -cf $@ $^ plots

clean_latex:
	@echo "Removing all latex byproducts"
	@rm -f *.{log,aux,bbl,bcf,blg,ilg,toc,tdo,fls,fdb_latexmk,lof,lot,idx,ind,snm,out,nav,synctex.gz,bak,xml,dvi,spl}

clean: clean_latex
	@echo "Removing latex pdf and arxiv tar"
	@rm -f $(output) $(arxivtar)
