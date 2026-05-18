#!/usr/bin/env python3
"""
semantic_index.py — Multilingual semantic indexing pipeline using FastEmbed.

Indexes a large document corpus into ChromaDB using local embeddings
(no API calls, no rate limits, runs fully offline).

Model: intfloat/multilingual-e5-large (1024-dim, supports RU/EN/KZ and 100+ languages)

Usage:
    python semantic_index.py --reset    # wipe index and reindex from scratch
    python semantic_index.py --resume   # continue interrupted indexing run
    python semantic_index.py --stats    # show index statistics

Configuration (environment variables):
    LAWBRAIN_INDEX_DIR   path to fulltext .txt files  (default: ./data/index/fulltext)
    LAWBRAIN_CHROMA_DIR  path to ChromaDB store       (default: ./data/chroma)
"""

import os
import time
import json
import signal
import argparse
from pathlib import Path

# ── Paths (override via environment variables) ─────────────────────────────────
FULLTEXT_INDEX = Path(os.environ.get("LAWBRAIN_INDEX_DIR", "./data/index/fulltext"))
CHROMA_DIR     = os.environ.get("LAWBRAIN_CHROMA_DIR", "./data/chroma")
COLLECTION     = "lawbrain_docs"
CHECKPOINT     = Path(CHROMA_DIR) / "indexed_files.json"

# ── Model ──────────────────────────────────────────────────────────────────────
EMBED_MODEL = "intfloat/multilingual-e5-large"
EMBED_DIM   = 1024
BATCH_SIZE  = 32

# ── Chunking ───────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 1800   # chars (~400 tokens)
CHUNK_OVERLAP = 200
MIN_CHUNK     = 120    # discard chunks shorter than this (headers, noise)

# ── Category mapping — customise for your folder structure ────────────────────
CATEGORY_MAP = {
    "01_TRIBUNAL":       "TRIBUNAL",
    "02_HO":             "HOME_OFFICE",
    "03_SOLICITORS":     "SOLICITORS",
    "04_EVIDENCE":       "EVIDENCE",
    "05_MEDICAL":        "MEDICAL",
    "06_TRANSLATIONS":   "TRANSLATIONS",
    "07_WITNESS":        "WITNESS",
    "08_CORRESPONDENCE": "CORRESPONDENCE",
    "09_ANALYTICS":      "ANALYTICS",
    "10_MEDIA":          "MEDIA",
}

CHECKPOINT_INTERVAL = 10  # save checkpoint every N files


def build_embedder():
    from fastembed import TextEmbedding
    print(f"Loading model {EMBED_MODEL} (~450 MB on first run, cached afterwards)...")
    model = TextEmbedding(EMBED_MODEL)
    print("Model ready.\n")
    return model


def embed_batch(model, texts: list[str]) -> list[list[float]]:
    results = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        embeddings = list(model.embed(batch))
        results.extend(e.tolist() for e in embeddings)
    return results


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks, respecting paragraph boundaries."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= CHUNK_SIZE:
        return [text]

    paras: list[str] = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if len(block) > CHUNK_SIZE:
            lines = [l.strip() for l in block.split("\n") if l.strip()]
            if lines:
                paras.extend(lines)
            else:
                for i in range(0, len(block), CHUNK_SIZE - CHUNK_OVERLAP):
                    paras.append(block[i : i + CHUNK_SIZE])
        else:
            paras.append(block)

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paras:
        if current_len + len(para) + 2 > CHUNK_SIZE and current:
            chunks.append("\n\n".join(current))
            carry: list[str] = []
            carry_len = 0
            for p in reversed(current):
                if carry_len + len(p) + 2 <= CHUNK_OVERLAP:
                    carry.insert(0, p)
                    carry_len += len(p) + 2
                else:
                    break
            current = carry
            current_len = carry_len
        current.append(para)
        current_len += len(para) + 2

    if current:
        chunks.append("\n\n".join(current))

    return [c for c in chunks if len(c) >= MIN_CHUNK]


def category_from(txt_path: Path) -> str:
    try:
        top = txt_path.relative_to(FULLTEXT_INDEX).parts[0]
        return CATEGORY_MAP.get(top, top)
    except Exception:
        return "UNKNOWN"


def load_checkpoint() -> set:
    if CHECKPOINT.exists():
        try:
            return set(json.loads(CHECKPOINT.read_text()))
        except Exception:
            pass
    return set()


def save_checkpoint(done: set) -> None:
    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
    CHECKPOINT.write_text(json.dumps(sorted(done), ensure_ascii=False))


def chunk_id(rel: str, ci: int) -> str:
    safe = rel.replace("/", "__").replace("\\", "__").replace(".txt", "")
    return f"{safe}__c{ci}"


def main():
    parser = argparse.ArgumentParser(
        description="LawBrain semantic indexer — multilingual FastEmbed + ChromaDB"
    )
    parser.add_argument("--resume", action="store_true", help="Continue interrupted run")
    parser.add_argument("--reset",  action="store_true", help="Wipe index and restart")
    parser.add_argument("--stats",  action="store_true", help="Show index statistics")
    args = parser.parse_args()

    import chromadb

    chroma = chromadb.PersistentClient(path=CHROMA_DIR)

    if args.stats:
        try:
            col  = chroma.get_collection(COLLECTION)
            done = load_checkpoint()
            print(f"\nLawBrain index (FastEmbed local):")
            print(f"  Collection : {COLLECTION}")
            print(f"  Chunks     : {col.count()}")
            print(f"  Files done : {len(done)}")
            print(f"  Model      : {EMBED_MODEL} ({EMBED_DIM}-dim)")
            print(f"  ChromaDB   : {CHROMA_DIR}\n")
        except Exception:
            print("Index not found. Run with --reset to build.")
        return

    if args.reset:
        try:
            chroma.delete_collection(COLLECTION)
            print("Collection deleted.")
        except Exception:
            pass
        if CHECKPOINT.exists():
            CHECKPOINT.unlink()
            print("Checkpoint cleared.")

    collection = chroma.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    done = load_checkpoint() if (args.resume and not args.reset) else set()

    all_files = sorted(FULLTEXT_INDEX.rglob("*.txt"))
    todo = [f for f in all_files if str(f.relative_to(FULLTEXT_INDEX)) not in done]

    print(f"\nTotal files : {len(all_files)}")
    print(f"Already done: {len(done)}")
    print(f"Remaining   : {len(todo)}")
    print(f"Model       : {EMBED_MODEL} ({EMBED_DIM}-dim, batch={BATCH_SIZE})\n")

    if not todo:
        print("Indexing complete — nothing to do.\n")
        return

    model = build_embedder()
    print("Starting indexing...\n")

    total_chunks = 0
    errors: list[tuple] = []
    t_start = time.time()

    _stop = False

    def _handle_signal(sig, frame):
        nonlocal _stop
        print(f"\nSignal {sig} received — saving checkpoint and exiting...")
        _stop = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT,  _handle_signal)

    for i, txt_file in enumerate(todo, 1):
        try:
            text = txt_file.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                done.add(str(txt_file.relative_to(FULLTEXT_INDEX)))
                continue

            category = category_from(txt_file)
            rel      = str(txt_file.relative_to(FULLTEXT_INDEX))
            filename = txt_file.stem
            chunks   = chunk_text(text)

            if not chunks:
                done.add(rel)
                continue

            embeddings = embed_batch(model, chunks)

            ids, documents, metadatas = [], [], []
            for ci, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                ids.append(chunk_id(rel, ci))
                documents.append(chunk)
                metadatas.append({
                    "source_file":  rel,
                    "category":     category,
                    "filename":     filename,
                    "chunk_index":  ci,
                    "total_chunks": len(chunks),
                })

            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

            total_chunks += len(chunks)
            done.add(rel)

            elapsed       = time.time() - t_start
            files_per_sec = i / elapsed
            eta_min       = (len(todo) - i) / files_per_sec / 60 if files_per_sec > 0 else 0

            marker = "█" if i % CHECKPOINT_INTERVAL == 0 else "·"
            print(
                f"  {marker} [{i:4d}/{len(todo)}] {filename[:48]:<48} "
                f"{len(chunks):3d} chunks | ETA ~{eta_min:.0f}m",
                flush=True,
            )

            if i % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(done)
                print(f"    checkpoint: {len(done)} files, {total_chunks} chunks\n", flush=True)

            if _stop:
                break

        except Exception as exc:
            errors.append((txt_file.name, str(exc)))
            print(f"  WARNING [{i:4d}] {txt_file.name}: {exc}", flush=True)
            continue

    save_checkpoint(done)

    elapsed_total = time.time() - t_start
    print(f"\n{'─'*60}")
    print(f"Done in {elapsed_total/60:.1f} min")
    print(f"  Files    : {len(done)}/{len(all_files)}")
    print(f"  Chunks   : {total_chunks}")
    print(f"  In Chroma: {collection.count()}")
    print(f"  Errors   : {len(errors)}")
    if errors:
        for name, err in errors[:5]:
            print(f"    - {name}: {err}")
    print()


if __name__ == "__main__":
    main()
