# Define variables
SLIDEFACTORY = ../../../tmp/test/slidefactory_3.1.0-beta.6.sif
THEME = ../../../tmp/test/slidefactory-themes/theme/csc-2019

# Define the markdown files and their corresponding PDF output
FILES = 2-gpu-ecosystem.md 4-gpu-concepts.md 9-non-portable.md 11-gpu-porting.md
PDFS = $(FILES:.md=.pdf)

# Default target
all: $(PDFS)

# Rule to convert .md to .pdf
%.pdf: %.md
	$(SLIDEFACTORY) slides -f pdf --theme $(THEME) $<

# Clean target to remove generated PDF files
.PHONY: clean 
clean:
	rm -f $(PDFS)
