"""Read local text and Markdown documents."""
from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".md"}

def load_documents(directory):
    """Return documents as dictionaries with source and text fields."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    documents = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                text = path.read_text(encoding="utf-8").strip()
            except UnicodeDecodeError:
                text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                documents.append({"source": str(path.relative_to(root)), "text": text})
    return documents
