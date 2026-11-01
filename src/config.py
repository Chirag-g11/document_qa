"""Central configuration."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_DIR = Path(os.getenv("DOCUMENTS_DIR", PROJECT_ROOT / "data" / "documents"))
MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.08"))
CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "180"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "35"))
RETRIEVER = os.getenv("QA_RETRIEVER", "tfidf").lower()
