import unittest
from pathlib import Path
from src.ingestion import load_documents
from src.chunking import chunk_documents
from src.retrieval import build_retriever
from src.qa import answer_question, FALLBACK
from src.config import DOCUMENTS_DIR

class DocumentQATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        docs = load_documents(DOCUMENTS_DIR)
        chunks = chunk_documents(docs, chunk_words=180, overlap=35)
        cls.retriever = build_retriever(chunks, method="tfidf")

    def test_refund_period_is_grounded(self):
        result = answer_question("What is the refund period?", self.retriever, threshold=0.04)
        self.assertTrue(result["supported"])
        self.assertTrue(result["citations"])
        self.assertIn("30 calendar days", result["answer"])

    def test_paid_leave_is_grounded(self):
        result = answer_question("How many paid annual leave days do full-time employees receive?", self.retriever, threshold=0.04)
        self.assertTrue(result["supported"])
        self.assertIn("18 days", result["answer"])

    def test_unknown_query_falls_back(self):
        result = answer_question("Who won the 2032 World Cup?", self.retriever, threshold=0.20)
        self.assertEqual(result["answer"], FALLBACK)
        self.assertFalse(result["supported"])

    def test_blank_question(self):
        result = answer_question("   ", self.retriever)
        self.assertFalse(result["supported"])
        self.assertEqual(result["answer"], "Please enter a question.")

if __name__ == "__main__":
    unittest.main()
