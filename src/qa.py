"""Grounded extractive answering and unsupported-query fallback."""
import re

FALLBACK = "Information not found in provided document."

def _sentences(text):
    # Simple sentence splitter suitable for the demonstration corpus.
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]

def answer_question(question, retriever, threshold=0.08, top_k=4):
    if not question or not question.strip():
        return {"answer": "Please enter a question.", "citations": [], "supported": False}

    hits = retriever.search(question.strip(), top_k=top_k)
    if not hits or hits[0]["score"] < threshold:
        return {"answer": FALLBACK, "citations": [], "supported": False}

    # Select sentences with lexical overlap to the query; preserve exact source wording.
    query_terms = set(re.findall(r"[a-z0-9]+", question.lower()))
    candidates = []
    for hit in hits:
        for sentence in _sentences(hit["text"]):
            sentence_terms = set(re.findall(r"[a-z0-9]+", sentence.lower()))
            overlap = len(query_terms & sentence_terms) / max(len(query_terms), 1)
            if overlap > 0:
                candidates.append((overlap + hit["score"] * 0.25, sentence, hit))
    candidates.sort(key=lambda x: x[0], reverse=True)

    selected = []
    seen = set()
    for _, sentence, hit in candidates:
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        selected.append((sentence, hit))
        if len(selected) == 2:
            break

    if not selected:
        # If a relevant passage is retrieved, return its exact text rather than inventing.
        selected = [(hits[0]["text"], hits[0])]

    answer = " ".join(sentence for sentence, _ in selected)
    citations = []
    used = set()
    for _, hit in selected:
        key = (hit["source"], hit["passage_id"])
        if key in used:
            continue
        used.add(key)
        citations.append({
            "source": hit["source"],
            "passage_id": hit["passage_id"],
            "score": hit["score"],
            "text": hit["text"],
        })
    return {"answer": answer, "citations": citations, "supported": True}
