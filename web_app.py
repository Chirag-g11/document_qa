"""Optional local Streamlit UI."""
import streamlit as st
from app import build_engine
from src.qa import answer_question
from src.config import MIN_RELEVANCE_SCORE, DOCUMENTS_DIR
from src.ingestion import load_documents
from src.chunking import chunk_documents
from src.retrieval import build_retriever
from src.config import CHUNK_WORDS, CHUNK_OVERLAP, RETRIEVER

st.set_page_config(page_title="Document QA Assistant", page_icon="📚", layout="wide")
st.title("📚 Document Question-Answering Assistant")
st.caption("Answers are grounded in local documents. Unsupported questions receive a transparent fallback.")

@st.cache_resource
def make_engine():
    docs = load_documents(DOCUMENTS_DIR)
    chunks = chunk_documents(docs, chunk_words=CHUNK_WORDS, overlap=CHUNK_OVERLAP)
    if not chunks:
        return None
    return build_retriever(chunks, method=RETRIEVER)

if st.button("Reload documents"):
    make_engine.clear()
    st.rerun()

retriever = make_engine()
if retriever is None:
    st.error(f"No .txt or .md documents found in `{DOCUMENTS_DIR}`.")
    st.stop()

st.success(f"Ready: {len(retriever.chunks)} passages indexed.")
question = st.text_input("Ask a question about the documents", placeholder="e.g. What is the refund period?")
if st.button("Get answer", type="primary") and question.strip():
    with st.spinner("Searching documents..."):
        result = answer_question(question, retriever, threshold=MIN_RELEVANCE_SCORE)
    st.subheader("Answer")
    st.write(result["answer"])
    st.subheader("Supporting source passages")
    if result["citations"]:
        for c in result["citations"]:
            with st.expander(f"{c['source']} — {c['passage_id']} (score {c['score']:.3f})", expanded=True):
                st.write(c["text"])
    else:
        st.info("No supporting passages met the relevance threshold.")
st.divider()
st.caption("Add your own .txt or .md files to data/documents/ and reload the documents.")
