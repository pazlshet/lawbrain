# Contributing to LawBrain

Contributions are welcome — bug fixes, performance improvements, new extractors, better chunking strategies.

## Getting started

```bash
git clone https://github.com/pazlshet/lawbrain.git
cd lawbrain
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Areas where help is useful

| Area | What's needed |
|------|--------------|
| **PDF extraction** | Better handling of scanned PDFs with mixed languages |
| **Chunking** | Smarter sentence-boundary chunking (currently paragraph-based) |
| **Models** | Testing other FastEmbed models for speed/quality trade-offs |
| **Search** | BM25 hybrid search alongside vector search |
| **Formats** | DOCX, EPUB, HTML extraction in `build_fulltext_index.py` |
| **Tests** | Unit tests for `chunk_text()` and `category_from()` |

## Code style

- Python 3.10+, no external formatting tools required
- Type hints on all public functions
- Keep functions short and single-purpose
- No print statements in library code — use `logging` or return values
- All user-facing paths configurable via environment variables (no hardcoded paths)

## Submitting a PR

1. Fork the repo and create a branch: `git checkout -b fix/my-thing`
2. Make your changes
3. Test manually against a small document corpus (`--stats` should report correct counts)
4. Open a PR with a clear description of what changed and why

## Reporting issues

Open a GitHub issue with:
- Python version and OS
- The command you ran
- The error output (full traceback)
- Approximate corpus size (number of files, languages)

## Licence

By contributing you agree your code will be released under the MIT licence.
