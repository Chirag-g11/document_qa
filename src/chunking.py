"""Paragraph-aware chunking with word overlap."""
import re

def chunk_documents(documents, chunk_words=180, overlap=35):
    if chunk_words < 1:
        raise ValueError("chunk_words must be at least 1")
    if overlap < 0 or overlap >= chunk_words:
        raise ValueError("overlap must be >= 0 and smaller than chunk_words")
    chunks = []
    for doc in documents:
        # Preserve paragraphs as units when possible, then split oversized paragraphs.
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", doc["text"]) if p.strip()]
        units = []
        for paragraph in paragraphs:
            words = paragraph.split()
            if len(words) <= chunk_words:
                units.append(words)
            else:
                step = chunk_words - overlap
                for start in range(0, len(words), step):
                    units.append(words[start:start + chunk_words])
        current = []
        for words in units:
            if current and len(current) + len(words) > chunk_words:
                chunks.append(_make_chunk(doc["source"], len(chunks), current))
                current = current[-overlap:] if overlap else []
            # A unit may itself be longer than the remaining space; split safely.
            if len(words) > chunk_words:
                words = words[:chunk_words]
            if current and len(current) + len(words) > chunk_words:
                chunks.append(_make_chunk(doc["source"], len(chunks), current))
                current = current[-overlap:] if overlap else []
            current.extend(words)
        if current:
            chunks.append(_make_chunk(doc["source"], len(chunks), current))
    # IDs are stable within each source file, independent of other files.
    counts = {}
    for chunk in chunks:
        source = chunk["source"]
        counts[source] = counts.get(source, 0) + 1
        chunk["passage_id"] = f"P{counts[source]:03d}"
    return chunks

def _make_chunk(source, index, words):
    return {"source": source, "text": " ".join(words), "passage_id": ""}
