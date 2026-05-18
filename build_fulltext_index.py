#!/usr/bin/env python3
"""
build_fulltext_index.py — Extract full text from PDF and text files into a flat index.

Mirrors the source document folder structure, producing one .txt file per source document.
The resulting index is consumed by semantic_index.py to build the vector store.

Usage:
    python build_fulltext_index.py --base /path/to/documents
    python build_fulltext_index.py --base ./data/documents --force   # overwrite existing

Dependencies:
    pip install pymupdf   # for PDF extraction (fitz)
"""

import sys
import time
import argparse
from pathlib import Path

BASE_DEFAULT    = Path("./data/documents")
INDEX_DIR_NAME  = "_INDEX"
FULLTEXT_SUBDIR = "fulltext"
SKIP_DIRS       = {INDEX_DIR_NAME, "_UNSORTED", "_TRASH"}
MIN_TEXT_BYTES  = 50  # files below this threshold are treated as image-only scans


def extract_pdf(pdf_path: Path) -> tuple[str, str]:
    """Return (text, status) where status is 'ok' | 'scan' | 'error:<msg>'."""
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        pages_text = [page.get_text() for page in doc]
        doc.close()
        text = "\n".join(pages_text).strip()
        if len(text.encode("utf-8")) < MIN_TEXT_BYTES:
            return "", "scan"
        return text, "ok"
    except ImportError:
        return "", "error:pymupdf not installed"
    except Exception as e:
        return "", f"error:{str(e)[:80]}"


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


def main():
    parser = argparse.ArgumentParser(
        description="Build full-text index from PDF and text documents"
    )
    parser.add_argument("--base",  default=str(BASE_DEFAULT), help="Root document folder")
    parser.add_argument("--force", action="store_true", help="Overwrite existing index files")
    args = parser.parse_args()

    base       = Path(args.base)
    index_root = base / INDEX_DIR_NAME / FULLTEXT_SUBDIR

    print(f"Source : {base}")
    print(f"Index  : {index_root}\n")

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

    total = len(files)
    print(f"Files to index: {total}\n")

    stats = {"ok": 0, "scan": 0, "skip": 0, "error": 0, "exists": 0}
    scan_files  = []
    error_files = []
    t0 = time.time()

    for i, src in enumerate(files, 1):
        rel      = src.relative_to(base)
        out_path = index_root / rel.with_suffix(".txt")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not args.force:
            stats["exists"] += 1
            if i % 100 == 0:
                print(f"  [{i}/{total}] skipping (already indexed)...")
            continue

        ext = src.suffix.lower()
        if ext == ".pdf":
            text, status = extract_pdf(src)
        else:
            text, status = extract_text_file(src)

        if status == "ok" and text:
            out_path.write_text(text, encoding="utf-8")
            stats["ok"] += 1
        elif status == "scan":
            stats["scan"] += 1
            scan_files.append(str(rel))
        elif status in ("skip", "empty"):
            stats["skip"] += 1
        else:
            stats["error"] += 1
            error_files.append(f"{rel}: {status}")

        if i % 50 == 0 or i == total:
            elapsed = time.time() - t0
            rate    = i / elapsed if elapsed > 0 else 0
            print(
                f"  [{i}/{total}] ok:{stats['ok']} scan:{stats['scan']} "
                f"err:{stats['error']}  ({rate:.1f} files/sec)"
            )

    elapsed = time.time() - t0
    print(f"\n{'='*55}")
    print(f"Done in {elapsed:.1f}s")
    print(f"  Indexed      : {stats['ok']}")
    print(f"  Already had  : {stats['exists']}")
    print(f"  Scans/no-OCR : {stats['scan']}")
    print(f"  Skipped      : {stats['skip']}")
    print(f"  Errors       : {stats['error']}")

    if scan_files:
        scan_log = index_root.parent / "scans_no_ocr.txt"
        scan_log.write_text("\n".join(scan_files), encoding="utf-8")
        print(f"\nImage-only scans logged → {scan_log}")

    if error_files:
        err_log = index_root.parent / "errors.txt"
        err_log.write_text("\n".join(error_files), encoding="utf-8")
        print(f"Errors logged → {err_log}")

    idx_files = list(index_root.rglob("*.txt"))
    idx_size  = sum(f.stat().st_size for f in idx_files)
    print(f"\nIndex: {len(idx_files)} files, {idx_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
