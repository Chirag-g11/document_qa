# Document Question-Answering Assistant

A fully local, retrieval-grounded question-answering project. It searches provided text documents, returns the most relevant passages with source citations, and explicitly refuses to guess when the documents do not contain enough evidence.

## Features

- Load `.txt` and `.md` documents from `data/documents/`
- Split documents into overlapping chunks while retaining filename and passage IDs
- Local retrieval using scikit-learn TF-IDF (no paid APIs, no model download required)
- Optional BM25 retrieval when `rank-bm25` is installed
- Evidence-based extractive answers assembled from the strongest matching sentences
- Citations include the source filename, passage ID, and the supporting passage
- Configurable relevance threshold and transparent unsupported-query fallback
- Command-line interactive mode and a small Streamlit web UI
- Unit tests and sample evaluation transcripts

## Requirements

- Python 3.10+
- Internet is not needed at runtime after dependencies are installed.
- No paid API keys are used.

## Quick start (Windows PowerShell)

```powershell
cd document_qa_assistant
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Ask questions in the terminal. Type `exit` or `quit` to stop.

### Run the web interface

```powershell
pip install -r requirements.txt
streamlit run web_app.py
```

Streamlit opens a local page in your browser. The app reads `.txt` and `.md` files from `data/documents/`.

## Add your own documents

1. Put `.txt` or `.md` files in `data/documents/`.
2. Restart the CLI, or click **Reload documents** in the web interface.
3. Ask questions that can be answered from those documents.

PDF/DOCX ingestion is intentionally not included in this starter version to keep the pipeline transparent and dependency-light. You can add extractors later.

## Example

Question: `What is the refund period?`

The sample corpus contains a policy passage stating that eligible items can be returned within 30 calendar days. The assistant should cite that passage.

Question: `Who won the 2032 World Cup?`

The sample corpus does not answer this. The assistant should say: **Information not found in provided document.**

## Retrieval design

1. **Ingestion:** read supported text files from the corpus folder.
2. **Chunking:** create paragraph-aware chunks with overlap.
3. **Indexing:** fit a local TF-IDF vectorizer over chunks.
4. **Retrieval:** rank chunks by cosine similarity to the question.
5. **Grounding:** choose relevant sentences only from retrieved evidence.
6. **Fallback:** if the score is below the threshold, return the required unsupported answer.
7. **Citations:** display each cited filename, passage ID, score, and exact supporting text.

This starter uses extractive sentence selection rather than a generative LLM. That reduces unsupported claims and makes the answer traceable. It is still important to review the cited passage and tune `MIN_RELEVANCE_SCORE` for your corpus.

## Configuration

Set these environment variables if needed:

- `QA_RETRIEVER=tfidf` (default) or `QA_RETRIEVER=bm25` (requires `rank-bm25`)
- `MIN_RELEVANCE_SCORE=0.08` (default TF-IDF threshold; tune on your own corpus)
- `CHUNK_WORDS=180`
- `CHUNK_OVERLAP=35`

PowerShell example:

```powershell
$env:MIN_RELEVANCE_SCORE="0.10"
python app.py
```

The TF-IDF threshold is corpus-dependent. A low threshold may retrieve irrelevant passages; a high threshold may reject valid questions.

## Project structure

```text
document_qa_assistant/
├── app.py
├── web_app.py
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── ingestion.py
│   ├── chunking.py
│   ├── retrieval.py
│   └── qa.py
├── data/
│   └── documents/
│       ├── employee_handbook.txt
│       └── product_policy.md
├── evaluation/
│   └── sample_transcripts.md
└── tests/
    └── test_qa.py
```

## Test

```powershell
python -m unittest discover -s tests -v
```

## Limitations

- TF-IDF is lexical retrieval, not semantic understanding. BM25 is also lexical.
- The default answer generator selects matching sentences and is not a free-form language model.
- Similarity scores are ranking signals, not calibrated probabilities.
- The included corpus is fictional demonstration content.
- The fallback helps avoid guessing but cannot guarantee zero hallucinations if a source itself is inaccurate or ambiguous.

## Suggested improvements

- Add PDF/DOCX parsing with `pypdf` and `python-docx`
- Add sentence-transformer embeddings for semantic retrieval, if downloading a free model is acceptable
- Add a reranker and better evaluation metrics
- Add upload support and persistent indexing
- Expand the evaluation set and tune retrieval thresholds on representative queries
