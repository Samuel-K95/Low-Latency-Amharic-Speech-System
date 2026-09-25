#!/usr/bin/env bash
# Build the report PDF (LuaLaTeX, two passes for references/TOC) and copy it next to this folder.
set -euo pipefail
cd "$(dirname "$0")"
for pass in 1 2; do lualatex -interaction=nonstopmode -halt-on-error main.tex > build.log 2>&1 || { tail -40 build.log; exit 1; }; done
cp main.pdf ../Week6_Low_Latency_Amharic_S2S_Report.pdf
echo "Built: $(pdfinfo main.pdf | awk '/Pages/{print $2}') pages -> Week6_Low_Latency_Amharic_S2S_Report.pdf"
