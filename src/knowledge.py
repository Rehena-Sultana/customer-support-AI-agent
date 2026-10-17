"""Small local RAG retriever using TF-IDF. Good for a demo; replace with a vector DB at scale."""
from pathlib import Path
from functools import lru_cache
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .config import KNOWLEDGE_DIR


def _read_documents(folder: str) -> list[dict]:
    root = Path(folder).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    docs = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".md", ".pdf"}:
            continue
        try:
            if path.suffix.lower() == ".pdf":
                from pypdf import PdfReader
                text = "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
            else:
                text = path.read_text(encoding="utf-8", errors="ignore")
            # Chunk by paragraphs for basic retrieval.
            paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 40]
            if not paragraphs and text.strip():
                paragraphs = [text.strip()]
            for i, paragraph in enumerate(paragraphs):
                docs.append({"source": path.name, "chunk": i + 1, "text": paragraph[:2500]})
        except Exception as exc:
            print(f"Skipping knowledge file {path}: {exc}")
    return docs


@lru_cache(maxsize=2)
def _index(folder: str):
    docs = _read_documents(folder)
    if not docs:
        return [], None, None
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=50000)
    matrix = vectorizer.fit_transform([d["text"] for d in docs])
    return docs, vectorizer, matrix


def retrieve(query: str, k: int = 4) -> tuple[str, list[str]]:
    docs, vectorizer, matrix = _index(str(Path(KNOWLEDGE_DIR).expanduser().resolve()))
    if not docs or vectorizer is None:
        return "", []
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).ravel()
    top = scores.argsort()[::-1][:k]
    selected = [docs[i] for i in top if scores[i] > 0.03]
    context = "\n\n".join(f"[Source: {d['source']} | chunk {d['chunk']}]\n{d['text']}" for d in selected)
    sources = list(dict.fromkeys(d["source"] for d in selected))
    return context, sources
