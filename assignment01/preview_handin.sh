#!/usr/bin/env bash
# Rebuild assignment01/handin.pdf without leaving LaTeX artifacts behind.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source_tex="$script_dir/handout/assignment01.tex"
output_pdf="$script_dir/handin.pdf"

if ! command -v latexmk >/dev/null; then
  echo "latexmk is required but was not found." >&2
  exit 1
fi

build_dir="$(mktemp -d "${TMPDIR:-/tmp}/assignment01-handin.XXXXXX")"
trap 'rm -rf "$build_dir"' EXIT

cp "$source_tex" "$build_dir/assignment01.tex"
(
  cd "$build_dir"
  latexmk -xelatex -interaction=nonstopmode -halt-on-error \
    -jobname=handin assignment01.tex
)
install -m 664 "$build_dir/handin.pdf" "$output_pdf"

echo
echo "Done: $output_pdf"
if command -v pdfinfo >/dev/null; then
  pdfinfo "$output_pdf" | grep -E '^(Pages|Page size|PDF version)'
fi

# Open it automatically only in a graphical local session. On a headless server,
# use the displayed path in your remote editor's PDF preview.
if [[ -n "${DISPLAY:-}" ]] && command -v xdg-open >/dev/null; then
  xdg-open "$output_pdf" >/dev/null 2>&1 &
fi
