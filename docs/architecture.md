# Architecture

## Overview

```
┌─────────────────────────────────────────────────────────┐
│                      Source documents                    │
│         PDF  ·  Markdown  ·  Plain text  ·  DOCX        │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│              build_fulltext_index.py                     │
│                                                          │
│  • PyMuPDF for PDF text extraction                       │
│  • Detects image-only scans (< 50 bytes extracted)       │
│  • Mirrors source folder structure as .txt files         │
│  • Skippable with --force flag                           │
└────────────────────────────┬────────────────────────────┘
                             │  flat .txt files
                             ▼
┌─────────────────────────────────────────────────────────┐
│                  semantic_index.py                       │
│                                                          │
│  ┌─────────────────┐    ┌──────────────────────────┐    │
│  │  chunk_text()   │    │  FastEmbed               │    │
│  │                 │───▶│  intfloat/e5-large       │    │
│  │  paragraph-aware│    │  1024-dim, 100+ languages│    │
│  │  overlap: 200   │    │  fully local, no API     │    │
│  └─────────────────┘    └────────────┬─────────────┘    │
│                                      │ embeddings        │
│  ┌─────────────────────────────────────────────────┐    │
│  │  checkpoint/resume (JSON per-file state)         │    │
│  │  SIGTERM-safe · batch upsert · ETA display       │    │
│  └────────────────────────┬────────────────────────┘    │
└───────────────────────────┼─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                     ChromaDB                            │
│                                                          │
│  Collection: lawbrain_docs                               │
│  Distance:   cosine                                      │
│  Metadata:   source_file · category · filename          │
│              chunk_index · total_chunks                  │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│                  Query interface                          │
│                                                          │
│  collection.query(                                       │
│      query_embeddings=[...],                             │
│      n_results=5,                                        │
│      where={"category": "EVIDENCE"},  # optional        │
│  )                                                       │
└─────────────────────────────────────────────────────────┘
```

## Chunking strategy

The chunker operates in two passes:

1. **Paragraph split** — split on `\n\n`. Paragraphs longer than `CHUNK_SIZE` (1800 chars) are further split by line.
2. **Accumulate with overlap** — paragraphs are accumulated into chunks up to `CHUNK_SIZE`. When a chunk is full, the last `CHUNK_OVERLAP` (200 chars) worth of paragraphs are carried forward into the next chunk.

Chunks shorter than `MIN_CHUNK` (120 chars) are discarded — these are typically stray headers or OCR noise.

**Why paragraph-first?** Legal and medical documents have strong paragraph structure. Splitting at paragraph boundaries preserves semantic units better than character-based splitting, which frequently cuts mid-sentence.

## Checkpoint design

The checkpoint is a JSON file containing the set of relative file paths already indexed. On `--resume`, the indexer skips any file whose relative path is already in the set.

Using *relative* paths (not filenames) avoids collisions when the same filename appears in multiple subdirectories. The checkpoint is written every `CHECKPOINT_INTERVAL` (10) files and on clean exit via SIGTERM/SIGINT.

## Model choice

`intfloat/multilingual-e5-large` was chosen over alternatives for these reasons:

| Model | Dims | Languages | Size | Notes |
|-------|------|-----------|------|-------|
| `multilingual-e5-large` | 1024 | 100+ | 560 MB | **Used** — best quality, handles Cyrillic/CJK |
| `multilingual-e5-small` | 384 | 100+ | 120 MB | 4× faster, lower recall on long queries |
| `all-MiniLM-L6-v2` | 384 | EN only | 80 MB | Fast, English-only |
| `bge-m3` | 1024 | 100+ | 1.1 GB | Comparable quality, larger download |

For corpora with Cyrillic or mixed-language content, `multilingual-e5-large` outperforms smaller alternatives on recall, justifying the larger download and slower throughput.

## Mail client design

`mail_client.py` treats Maildir as the single source of truth and wraps it in a minimal CLI designed for use by AI agents and automation scripts.

Key decisions:
- **Maildir over IMAP live queries** — all reads are local filesystem operations, making them fast and offline-capable
- **UID-based addressing** — messages are addressed by the `U=` field in Maildir filenames, which is stable across syncs
- **No state mutation** — the client never marks messages as read or moves files; that is left to mbsync
- **Stdin fallback** — `send` and `reply` accept `-` as the body file, reading from stdin for pipeline use
