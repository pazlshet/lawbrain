# LawBrain — Multilingual Document Intelligence

[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastEmbed](https://img.shields.io/badge/FastEmbed-e5--large-orange?logo=huggingface&logoColor=white)](https://huggingface.co/intfloat/multilingual-e5-large)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-vector%20store-green)](https://www.trychroma.com/)
[![Offline](https://img.shields.io/badge/runs-fully%20offline-brightgreen)](#)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

A production-grade document intelligence system built for large, multilingual corpora.
Originally developed to manage 1,800+ legal documents across Russian, Kazakh, and English.

## What it does

**Three-stage pipeline:**

```
Documents (PDF, MD, TXT)
    │
    ▼
build_fulltext_index.py   ← two-stage extraction:
    │                          1. PyMuPDF (fast) — text-layer PDFs
    │                          2. Docling OCR   — scanned/image-only PDFs
    ▼
semantic_index.py          ← chunks text, embeds with e5-large, stores in ChromaDB
    │
    ▼
ChromaDB vector store      ← queryable by cosine similarity, filtered by category/date
```

## Modules

| File | Purpose |
|------|---------|
| `build_fulltext_index.py` | Two-stage PDF extraction: PyMuPDF (fast) + Docling OCR (scans) |
| `semantic_index.py` | Embed text chunks and upsert into ChromaDB (local, no API) |
| `mail_client.py` | CLI mail interface for Maildir/IMAP + SMTP, designed for AI agent use |

## Key design decisions

**Two-stage PDF extraction.** PyMuPDF handles text-layer PDFs at ~350 files/min. When PyMuPDF finds an image-only scan, [Docling](https://github.com/docling-project/docling) (IBM, Apache 2.0) takes over: it runs AI-based layout analysis + OCR, preserves table structure, and outputs clean Markdown. This means scanned court orders, medical records, and handwritten statements are indexed instead of silently skipped. Flags: `--no-docling` (old behaviour), `--docling-all` (force docling for every PDF), `--ocr-only` (re-process the `scans_no_ocr.txt` backlog from a prior run).

**Fully offline embeddings.** Uses [intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large) via FastEmbed — 1024-dim, supports 100+ languages, runs locally with no API calls and no rate limits. First download is ~450 MB, then cached.

**Checkpoint/resume.** The indexer writes a JSON checkpoint every 10 files. If the process is interrupted (OOM, SIGINT, system restart), it resumes from where it stopped — essential when indexing thousands of files over hours.

**Paragraph-aware chunking.** The chunker respects paragraph boundaries before falling back to character splitting, with configurable overlap between chunks. This preserves semantic coherence better than naive character-based splitting.

**Category-aware metadata.** Each chunk is stored with metadata: `source_file`, `category` (from folder name), `filename`, `chunk_index`. This allows post-retrieval filtering without re-embedding.

**SIGTERM handling.** The indexer catches SIGTERM/SIGINT and saves its checkpoint before exiting — safe for use with systemd or job schedulers.

## Quick start

```bash
pip install -r requirements.txt

# 1. Extract text (PyMuPDF fast path + Docling OCR for scans)
python build_fulltext_index.py --base /path/to/your/documents

# First run downloads Docling models (~258 MB, cached afterwards).
# After the run, scans_no_ocr.txt lists any files docling also failed on.

# Re-process only the scan backlog from a previous run (no docling = no list):
python build_fulltext_index.py --base /path/to/your/documents --ocr-only

# Skip OCR entirely (fast, but scans are skipped):
python build_fulltext_index.py --base /path/to/your/documents --no-docling

# 2. Build the semantic index (downloads ~450 MB e5-large on first run)
export LAWBRAIN_INDEX_DIR=/path/to/your/documents/_INDEX/fulltext
export LAWBRAIN_CHROMA_DIR=./data/chroma
python semantic_index.py --reset

# Resume if interrupted
python semantic_index.py --resume

# Check progress
python semantic_index.py --stats
```

## Mail client

The mail client is designed to be called from scripts and AI agents as a structured interface to an isync/mbsync Maildir.

```bash
export MAIL_FROM=you@example.com
export MAIL_SMTP_HOST=smtp.example.com
export MAIL_SMTP_PASS=yourpassword
export MAIL_MAILDIR=/home/user/Mail/INBOX

python mail_client.py list 20
python mail_client.py read 142
python mail_client.py search "tribunal"
python mail_client.py send recipient@example.com "Subject" body.txt
python mail_client.py reply 142 reply_body.txt
python mail_client.py sync
```

## Performance

Tested on a corpus of 1,866 documents (mix of Russian/Kazakh/English legal PDFs, markdown notes, and plain text):

- Full-text extraction: ~357 files/min (PyMuPDF)
- Embedding throughput: ~8 files/min with e5-large on CPU (Intel Core i5, 8 GB RAM)
- Total indexing time: ~4 hours for 1,800+ files on consumer hardware
- Resulting index: ~180k chunks, ~2 GB ChromaDB store

## Folder structure (recommended)

```
data/
├── documents/          ← your source documents (gitignored)
│   ├── 01_CATEGORY_A/
│   ├── 02_CATEGORY_B/
│   └── _INDEX/
│       └── fulltext/   ← generated by build_fulltext_index.py
└── chroma/             ← ChromaDB store (gitignored)
```

## Querying the index

```python
import chromadb
from fastembed import TextEmbedding

chroma = chromadb.PersistentClient(path="./data/chroma")
col    = chroma.get_collection("lawbrain_docs")
model  = TextEmbedding("intfloat/multilingual-e5-large")

query     = "electroshock torture forensic evidence"
embedding = list(model.embed([query]))[0].tolist()

results = col.query(
    query_embeddings=[embedding],
    n_results=5,
    where={"category": "EVIDENCE"},   # optional filter
)

for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"[{meta['category']}] {meta['filename']}")
    print(doc[:300])
    print()
```

## Why I built this

I needed to search across 1,800+ legal documents in three languages, many of which were scanned PDFs with OCR output, with no reliable internet access and a strict requirement to keep data local. Commercial RAG solutions either require cloud APIs or don't handle Cyrillic/CJK text well. This pipeline does both.

## License

MIT
