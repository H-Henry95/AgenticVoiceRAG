"""Ingest documents into the vector store.

Usage:
    python scripts/ingest.py                 # uses data/documents
    python scripts/ingest.py path/to/folder
"""
from __future__ import annotations

import sys

from src.common.config import get_settings
from src.ingestion.pipeline import ingest


def main() -> None:
    folder = sys.argv[1] if len(sys.argv) > 1 else None
    n = ingest(get_settings(), folder)
    print(f"Ingested {n} chunks.")


if __name__ == "__main__":
    main()
