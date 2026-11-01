"""Local TF-IDF retrieval, with optional BM25."""
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TfidfRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in chunks])

    def search(self, query, top_k=4):
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        order = np.argsort(scores)[::-1][:top_k]
        return [dict(self.chunks[i], score=float(scores[i])) for i in order if scores[i] > 0]

class BM25Retriever:
    def __init__(self, chunks):
        from rank_bm25 import BM25Okapi
        self.chunks = chunks
        self.tokenized = [_tokenize(c["text"]) for c in chunks]
        self.model = BM25Okapi(self.tokenized)

    def search(self, query, top_k=4):
        scores = self.model.get_scores(_tokenize(query))
        order = np.argsort(scores)[::-1][:top_k]
        max_score = float(max(scores)) if len(scores) else 0.0
        # Normalize by the best score for a comparable threshold convention.
        return [
            dict(self.chunks[i], score=float(scores[i] / max_score) if max_score > 0 else 0.0)
            for i in order if scores[i] > 0
        ]

def _tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

def build_retriever(chunks, method="tfidf"):
    if not chunks:
        raise ValueError("Cannot build a retriever without passages.")
    if method == "bm25":
        try:
            return BM25Retriever(chunks)
        except ImportError as exc:
            raise RuntimeError("BM25 selected but rank-bm25 is not installed. Run pip install rank-bm25.") from exc
    if method != "tfidf":
        raise ValueError("method must be 'tfidf' or 'bm25'")
    return TfidfRetriever(chunks)
