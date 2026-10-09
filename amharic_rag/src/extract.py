#!/usr/bin/env python3
"""
Step 1: Extract Amharic text from PDF (two-column bilingual layout).
Extracts the Amharic left column using bounding box clip rect (0, 0, 306, height),
marking page boundaries with ===PAGE n===.
"""
import sys
from pathlib import Path
import fitz  # PyMuPDF

def run(force=False):
    base_dir = Path(__file__).resolve().parent.parent
    pdf_path = base_dir / 'data' / 'labour_proclamation_1156_2019.pdf'
    raw_path = base_dir / 'data' / 'raw.txt'
    report_path = base_dir / 'data' / 'cleaning_report.txt'
    
    if raw_path.exists() and not force:
        print(f"raw.txt already exists at {raw_path}")
    else:
        if not pdf_path.exists():
            print(f"Error: PDF not found at {pdf_path}")
            sys.exit(1)
        print(f"Extracting Amharic column from {pdf_path}...")
        text = extract_amharic_column(pdf_path)
        raw_path.write_text(text, encoding='utf-8')
        print(f"Wrote {len(text)} chars to {raw_path}")
    
    # Write report section
    write_report(report_path, raw_path, pdf_path)

def extract_amharic_column(pdf_path):
    """
    Extract only the Amharic left column (x <= 306) page by page,
    prepending ===PAGE n=== before each page.
    """
    doc = fitz.open(str(pdf_path))
    pages_output = []
    for i in range(len(doc)):
        page = doc[i]
        page_num = i + 1
        # Left column clip bounding box (width is 612, split is at 306)
        rect = fitz.Rect(0, 0, 306, page.rect.height)
        page_text = page.get_text('text', clip=rect)
        pages_output.append(f"===PAGE {page_num}===\n{page_text}")
    doc.close()
    return '\n'.join(pages_output)

def write_report(report_path, raw_path, pdf_path=None):
    import datetime
    raw = raw_path.read_text(encoding='utf-8') if raw_path.exists() else ''
    lines = raw.splitlines()
    non_empty = [l for l in lines if l.strip()]
    
    # Count lines per page
    page_sections = raw.split('===PAGE ')
    lines_per_page = []
    short_pages = []
    for sec in page_sections:
        if not sec.strip():
            continue
        p_header, *p_lines = sec.splitlines()
        p_num = p_header.split('===')[0].strip()
        non_empty_p = [l for l in p_lines if l.strip()]
        lines_per_page.append((p_num, len(p_lines), len(non_empty_p)))
        if len(non_empty_p) < 15:
            short_pages.append((p_num, len(non_empty_p)))

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"=== STAGE 1: EXTRACT ===\n")
        f.write(f"Date: {datetime.datetime.now().isoformat()}\n")
        f.write(f"PDF page count: {len(lines_per_page)}\n")
        f.write(f"raw.txt size: {len(raw)} chars\n")
        f.write(f"raw.txt lines: {len(lines)}\n")
        f.write(f"raw.txt non-empty lines: {len(non_empty)}\n")
        f.write(f"Suspiciously short or empty pages (< 15 lines): {short_pages if short_pages else 'None'}\n\n")
        f.write(f"Lines per page (Page: total_lines / non_empty_lines):\n")
        for p, tot, non_e in lines_per_page:
            f.write(f"  Page {p:>3}: {tot:>3} total lines, {non_e:>3} non-empty lines\n")

if __name__ == '__main__':
    force_run = '--force' in sys.argv
    run(force=force_run)

