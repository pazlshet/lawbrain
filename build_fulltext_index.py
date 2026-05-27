#!/usr/bin/env python3
"""
build_fulltext_index.py — Extract full text from PDF and text files into a flat index.

Mirrors the source document folder structure, producing one .txt file per source document.
The resulting index is consumed by semantic_index.py to build the vector store.

Two-stage PDF extraction:
  1. PyMuPDF (fast) — text-layer PDFs are extracted immediately
  2. Docling (OCR)  — image-only scans are processed with AI layout + OCR

Usage:
    python build_fulltext_index.py --base /path/to/documents
    python build_fulltext_index.py --base ./data/documents --force
    python build_fulltext_index.py --base ./data/documents --no-docling   # skip OCR
    python build_fulltext_index.py --base ./data/documents --docling-all  # docling for all PDFs
    python build_fulltext_index.py --base ./data/documents --ocr-only     # re-run scans_no_ocr.txt

Dependencies:
    pip install pymupdf docling
"""

import sys
import time
import argparse
from pathlib import Path

BASE_DEFAULT    = Path("./data/documents")
INDEX_DIR_NAME  = "_INDEX"
FULLTEXT_SUBDIR = "fulltext"
SKIP_DIRS       = {INDEX_DIR_NAME, "_UNSORTED", "_TRASH"}
MIN_TEXT_BYTES  = 50

_docling_converter = None


def _get_docling_converter():
    global _docling_converter
    if _docling_converter is not None:
        return _docling_converter
    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            EasyOcrOptions,
        )

        ocr_opts = EasyOcrOptions(
            force_full_page_ocr=True,
            lang=["ru", "en"],  # Russian (covers Kazakh Cyrillic) + English
        )
        opts = PdfPipelineOptions()
        opts.do_ocr = True
        opts.do_table_structure = True
        opts.ocr_options = ocr_opts

        _docling_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=opts)
            }
        )
        return _docling_converter
    except ImportError:
        return None
    except Exception as e:
        print(f"  [docling] init error: {e}", file=sys.stderr)
        return None


def extract_pdf_docling(pdf_path: Path) -> tuple[str, str]:
    """Extract text from any PDF via docling (OCR + layout). Returns (markdown_text, status)."""
    converter = _get_docling_converter()
    if converter is None:
        return "", "error:docling not available"
    try:
        result = converter.convert(str(pdf_path))
        text = result.document.export_to_markdown().strip()
        if len(text.encode("utf-8")) < MIN_TEXT_BYTES:
            return "", "error:docling_empty_output"
        return text, "ocr_ok"
    except Exception as e:
        return "", f"error:docling:{str(e)[:120]}"


def extract_pdf(pdf_path: Path, use_docling: bool = True, docling_all: bool = False) -> tuple[str, str]:
    """
    Return (text, status). Status: 'ok' | 'ocr_ok' | 'scan' | 'error:<msg>'.

    Fast path: PyMuPDF for text-layer PDFs.
    Fallback:  docling OCR for image-only scans (when use_docling=True).
    """
    if docling_all:
        return extract_pdf_docling(pdf_path)

    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        pages_text = [page.get_text() for page in doc]
        doc.close()
        text = "\n".join(pages_text).strip()
        if len(text.encode("utf-8")) >= MIN_TEXT_BYTES:
            return text, "ok"
        # image-only scan — fall through to docling
    except ImportError:
        return "", "error:pymupdf not installed"
    except Exception as e:
        return "", f"error:{str(e)[:80]}"

    if not use_docling:
        return "", "scan"

    return extract_pdf_docling(pdf_path)


def extract_text_file(path: Path) -> tuple[str, str]:
    try:
        if path.suffix.lower() in (".txt", ".md", ".csv"):
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if len(text.encode("utf-8")) < MIN_TEXT_BYTES:
                return "", "empty"
            return text, "ok"
        return "", "skip"
    except Exception as e:
        return "", f"error:{str(e)[:80]}"


def collect_files(base: Path, ocr_only: bool, scan_log: Path) -> list[Path]:
    """Return list of files to process."""
    if ocr_only:
        if not scan_log.exists():
            print(f"No scan log found at {scan_log}. Nothing to re-process.")
            return []
        entries = [l.strip() for l in scan_log.read_text().splitlines() if l.strip()]
        files = [base / e for e in entries if (base / e).exists()]
        missing = len(entries) - len(files)
        if missing:
            print(f"  Warning: {missing} entries from scan log no longer exist, skipping.")
        return files

    target_exts = {".pdf", ".txt", ".md"}
    files = []
    for f in base.rglob("*"):
        if not f.is_file():
            continue
        parts = f.relative_to(base).parts
        if any(p in SKIP_DIRS for p in parts):
            continue
        if f.suffix.lower() in target_exts:
            files.append(f)
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Build full-text index from PDF and text documents"
    )
    parser.add_argument("--base",        default=str(BASE_DEFAULT), help="Root document folder")
    parser.add_argument("--force",       action="store_true", help="Overwrite existing index files")
    parser.add_argument("--no-docling",  action="store_true", help="Skip OCR; log scans as before")
    parser.add_argument("--docling-all", action="store_true", help="Run all PDFs through docling")
    parser.add_argument("--ocr-only",    action="store_true",
                        help="Re-process only files listed in scans_no_ocr.txt (from a prior run)")
    args = parser.parse_args()

    use_docling  = not args.no_docling
    docling_all  = args.docling_all

    base       = Path(args.base)
    index_root = base / INDEX_DIR_NAME / FULLTEXT_SUBDIR
    scan_log   = base / INDEX_DIR_NAME / "scans_no_ocr.txt"

    print(f"Source     : {base}")
    print(f"Index      : {index_root}")
    ocr_label = "ALL via docling" if docling_all else ("docling fallback" if use_docling else "disabled")
    print(f"OCR mode   : {ocr_label}\n")

    files = collect_files(base, args.ocr_only, scan_log)
    total = len(files)
    print(f"Files to process: {total}\n")

    if total == 0:
        return

    # Initialise docling early so the model downloads before the main loop
    if use_docling or docling_all:
        print("Initialising docling (may download models on first run ~258 MB)...")
        if _get_docling_converter() is not None:
            print("Docling ready.\n")
        else:
            print("Docling unavailable — falling back to scan-skip mode.\n")

    stats = {"ok": 0, "ocr_ok": 0, "scan": 0, "skip": 0, "error": 0, "exists": 0}
    remaining_scans = []
    error_files     = []
    t0 = time.time()

    for i, src in enumerate(files, 1):
        rel      = src.relative_to(base)
        out_path = index_root / rel.with_suffix(".txt")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not args.force and not args.ocr_only:
            stats["exists"] += 1
            if i % 100 == 0:
                print(f"  [{i}/{total}] skipping (already indexed)...")
            continue

        ext = src.suffix.lower()
        if ext == ".pdf":
            text, status = extract_pdf(src, use_docling=use_docling, docling_all=docling_all)
        else:
            text, status = extract_text_file(src)

        if status in ("ok", "ocr_ok") and text:
            out_path.write_text(text, encoding="utf-8")
            stats[status] += 1
        elif status == "scan":
            stats["scan"] += 1
            remaining_scans.append(str(rel))
        elif status in ("skip", "empty"):
            stats["skip"] += 1
        else:
            stats["error"] += 1
            error_files.append(f"{rel}: {status}")

        if i % 10 == 0 or i == total:
            elapsed = time.time() - t0
            rate    = i / elapsed if elapsed > 0 else 0
            print(
                f"  [{i}/{total}] ok:{stats['ok']} ocr:{stats['ocr_ok']} "
                f"scan:{stats['scan']} err:{stats['error']}  ({rate:.1f} files/sec)"
            )

    elapsed = time.time() - t0
    print(f"\n{'='*55}")
    print(f"Done in {elapsed:.1f}s")
    print(f"  Text-extracted : {stats['ok']}")
    print(f"  OCR-extracted  : {stats['ocr_ok']}")
    print(f"  Already indexed: {stats['exists']}")
    print(f"  Scans/no-OCR   : {stats['scan']}")
    print(f"  Skipped/empty  : {stats['skip']}")
    print(f"  Errors         : {stats['error']}")

    if remaining_scans:
        scan_log.parent.mkdir(parents=True, exist_ok=True)
        scan_log.write_text("\n".join(remaining_scans), encoding="utf-8")
        print(f"\nImage-only scans logged → {scan_log}")
        print(f"  Re-run with --ocr-only to process them with docling.")
    elif scan_log.exists() and args.ocr_only:
        scan_log.unlink()
        print(f"\nAll scans processed — {scan_log.name} removed.")

    if error_files:
        err_log = index_root.parent / "errors.txt"
        err_log.write_text("\n".join(error_files), encoding="utf-8")
        print(f"Errors logged → {err_log}")

    idx_files = list(index_root.rglob("*.txt"))
    idx_size  = sum(f.stat().st_size for f in idx_files)
    print(f"\nIndex: {len(idx_files)} files, {idx_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
