# Report

Compile the report (if a TeX toolchain is installed):

```bash
cd report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Place generated performance figures in `report/figures/` (for example `ecdf.pdf`) so they are embedded automatically.
