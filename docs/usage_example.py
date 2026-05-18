#!/usr/bin/env python3
"""
docs/usage_example.py — End-to-end example: index documents and query the vector store.

Run this after installing requirements and setting environment variables:

    export LAWBRAIN_INDEX_DIR=./data/index/fulltext
    export LAWBRAIN_CHROMA_DIR=./data/chroma
    python docs/usage_example.py
"""

import os
from pathlib import Path

# ── 0. Setup paths ─────────────────────────────────────────────────────────────
INDEX_DIR  = Path(os.environ.get("LAWBRAIN_INDEX_DIR", "./data/index/fulltext"))
CHROMA_DIR = os.environ.get("LAWBRAIN_CHROMA_DIR", "./data/chroma")
COLLECTION = "lawbrain_docs"

# ── 1. Create a small sample corpus ────────────────────────────────────────────
# In real use, point INDEX_DIR at the output of build_fulltext_index.py.
# Here we create three tiny synthetic documents to demonstrate the pipeline.

SAMPLE_DOCS = {
    "01_EVIDENCE/report_2009.txt": """\
Forensic Medical Examination Report No. 420
Date: 16 January 2009
Subject: Physical examination of detainee

Objective findings:
On the right side of the abdomen, upper third, a triangular scar approximately
1.5 × 0.7 cm, whitish-grey in colour. Surrounding the scar, a rectangular area
of dark-brown skin pigmentation approximately 7.0 × 5.0 cm —
(indicative of electroshock device contact).

Two bruises on the chest wall: 4.0 × 3.0 cm and 6.0 × 4.0 cm.

Conclusion:
Bruising to the chest wall caused by blunt-force trauma with a limited contact surface.
""",
    "02_CORRESPONDENCE/letter_2009.txt": """\
Letter from Advocate Semin to the General Prosecutor of Kazakhstan
Date: 12 March 2009

I hereby submit that the injuries sustained by my client are the result of
the application of electroshock. Three separate official documents confirm
this finding, including the forensic report dated 16 January 2009.

The investigation has been conducted by the same agency responsible for
the alleged torture, rendering it neither independent nor effective.
""",
    "03_MEDICAL/mri_report_2015.txt": """\
MRI Report — Pituitary Gland
Date: 2015
Institution: City Clinical Hospital

Findings: The pituitary gland is flattened along the floor of the sella turcica.
Dimensions: 4 × 12 × 6 mm. The pituitary stalk is not displaced.
No space-occupying lesion identified.

Impression: Hypoplasia of the pituitary gland. Clinical correlation recommended.
""",
}

def create_sample_corpus():
    for rel_path, content in SAMPLE_DOCS.items():
        out = INDEX_DIR / rel_path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(content, encoding="utf-8")
    print(f"Sample corpus written to {INDEX_DIR} ({len(SAMPLE_DOCS)} files)\n")


# ── 2. Index the corpus ────────────────────────────────────────────────────────
def build_index():
    import chromadb
    from fastembed import TextEmbedding

    print("Loading embedding model (downloads ~450 MB on first run)...")
    model = TextEmbedding("intfloat/multilingual-e5-large")
    print("Model ready.\n")

    chroma     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    all_files = sorted(INDEX_DIR.rglob("*.txt"))
    print(f"Indexing {len(all_files)} files...\n")

    for txt_file in all_files:
        text     = txt_file.read_text(encoding="utf-8").strip()
        rel      = str(txt_file.relative_to(INDEX_DIR))
        category = txt_file.relative_to(INDEX_DIR).parts[0]

        embedding = list(model.embed([text]))[0].tolist()

        collection.upsert(
            ids=[rel],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{"source_file": rel, "category": category, "filename": txt_file.stem}],
        )
        print(f"  indexed: {rel}")

    print(f"\nIndex ready — {collection.count()} chunks in ChromaDB.\n")
    return collection, model


# ── 3. Query the index ─────────────────────────────────────────────────────────
def query(collection, model, query_text: str, n: int = 3, category: str = None):
    print(f"Query: '{query_text}'")
    if category:
        print(f"Filter: category = {category}")
    print("-" * 60)

    embedding = list(model.embed([query_text]))[0].tolist()
    where     = {"category": category} if category else None

    results = collection.query(
        query_embeddings=[embedding],
        n_results=n,
        where=where,
    )

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        score = 1 - dist  # cosine similarity
        print(f"\n[{meta['category']}] {meta['filename']}  (similarity: {score:.3f})")
        print(doc[:300].strip())
        print("...")

    print()


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Step 1 — write sample documents
    create_sample_corpus()

    # Step 2 — build vector index
    collection, model = build_index()

    # Step 3 — run queries
    query(collection, model, "electroshock torture forensic evidence")
    query(collection, model, "pituitary gland MRI findings", category="03_MEDICAL")
    query(collection, model, "independent investigation prosecutor letter")
