"""Command-line interface for the local document QA assistant."""
from src.ingestion import load_documents
from src.chunking import chunk_documents
from src.retrieval import build_retriever
from src.qa import answer_question
from src.config import DOCUMENTS_DIR, MIN_RELEVANCE_SCORE, CHUNK_WORDS, CHUNK_OVERLAP, RETRIEVER

def build_engine():
    docs = load_documents(DOCUMENTS_DIR)
    chunks = chunk_documents(docs, chunk_words=CHUNK_WORDS, overlap=CHUNK_OVERLAP)
    if not chunks:
        raise RuntimeError(f"No .txt or .md documents found in {DOCUMENTS_DIR}")
    retriever = build_retriever(chunks, method=RETRIEVER)
    return retriever

def main():
    print("=" * 64)
    print("DOCUMENT QUESTION-ANSWERING ASSISTANT")
    print("Local retrieval • grounded answers • source passages")
    print("=" * 64)
    try:
        retriever = build_engine()
    except Exception as exc:
        print(f"Could not initialize assistant: {exc}")
        return
    print(f"Indexed {len(retriever.chunks)} passages from {len(set(c['source'] for c in retriever.chunks))} files.")
    print("Ask a question, or type 'exit' to quit.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break
        if not question:
            continue
        result = answer_question(question, retriever, threshold=MIN_RELEVANCE_SCORE)
        print(f"\nAssistant: {result['answer']}")
        if result["citations"]:
            print("\nSupporting sources:")
            for c in result["citations"]:
                print(f"- [{c['source']} | {c['passage_id']} | score={c['score']:.3f}]")
                print(f"  {c['text']}")
        print()

if __name__ == "__main__":
    main()
