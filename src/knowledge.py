"""Knowledge retrieval using ChromaDB for semantic search with automatic fallback to TF-IDF."""
from pathlib import Path
import hashlib
from typing import Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import KNOWLEDGE_DIR, CHROMA_DIR, EMBEDDING_MODEL


def _read_documents(folder: str) -> list[dict]:
    root = Path(folder).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    docs = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".md", ".pdf"}:
            continue
        try:
            if path.suffix.lower() == ".pdf":
                from pypdf import PdfReader
                text = "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
            else:
                text = path.read_text(encoding="utf-8", errors="ignore")

            chunks = splitter.split_text(text)
            valid_chunks = [c.strip() for c in chunks if len(c.strip()) > 20]
            if not valid_chunks and text.strip():
                valid_chunks = [text.strip()]

            for i, chunk in enumerate(valid_chunks):
                doc_id = f"{path.stem}_{i + 1}"
                docs.append({
                    "id": doc_id,
                    "source": path.name,
                    "chunk": i + 1,
                    "text": chunk
                })
        except Exception as exc:
            print(f"Skipping knowledge file {path}: {exc}")
    return docs


_chroma_client = None
_last_doc_hash = None


def _get_collection():
    global _chroma_client, _last_doc_hash
    import chromadb
    from chromadb.utils import embedding_functions

    chroma_path = Path(CHROMA_DIR).expanduser().resolve()
    chroma_path.mkdir(parents=True, exist_ok=True)

    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=str(chroma_path))

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
        normalize_embeddings=True
    )
    col_suffix = hashlib.md5(EMBEDDING_MODEL.encode("utf-8")).hexdigest()[:8]
    collection = _chroma_client.get_or_create_collection(
        name=f"support_knowledge_{col_suffix}",
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}
    )

    docs = _read_documents(KNOWLEDGE_DIR)
    content_hash = hashlib.md5("".join(d["text"] for d in docs).encode("utf-8")).hexdigest()

    # Sync documents into collection if documents changed or collection is empty
    if (content_hash != _last_doc_hash or collection.count() == 0) and docs:
        existing = collection.get()
        if existing and existing.get("ids"):
            collection.delete(ids=existing["ids"])
        ids = [d["id"] for d in docs]
        documents = [d["text"] for d in docs]
        metadatas = [{"source": d["source"], "chunk": d["chunk"]} for d in docs]
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        _last_doc_hash = content_hash

    return collection


def _retrieve_tfidf(query: str, k: int = 4) -> tuple[str, list[str]]:
    docs = _read_documents(KNOWLEDGE_DIR)
    if not docs:
        return "", []
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=50000)
    matrix = vectorizer.fit_transform([d["text"] for d in docs])
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).ravel()
    top = scores.argsort()[::-1][:k]
    selected = [docs[i] for i in top if scores[i] > 0.03]
    context = "\n\n".join(f"[Source: {d['source']} | chunk {d['chunk']}]\n{d['text']}" for d in selected)
    sources = list(dict.fromkeys(d["source"] for d in selected))
    return context, sources


def retrieve(query: str, k: int = 4) -> tuple[str, list[str]]:
    query_clean = query.strip()
    if not query_clean:
        return "", []

    try:
        collection = _get_collection()
        count = collection.count()
        if count == 0:
            return _retrieve_tfidf(query_clean, k=k)

        n_results = min(k, count)
        results = collection.query(query_texts=[query_clean], n_results=n_results)

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0] if "distances" in results else [0.0] * len(documents)

        selected = []
        if documents:
            try:
                vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
                doc_mat = vec.fit_transform(documents)
                lex_sims = cosine_similarity(vec.transform([query_clean]), doc_mat).ravel()
            except Exception:
                lex_sims = [0.0] * len(documents)

            for doc_text, meta, dist, lex_sim in zip(documents, metadatas, distances, lex_sims):
                # Require strong semantic similarity or moderate semantic similarity backed by lexical overlap
                if dist <= 0.25 or (dist <= 0.50 and lex_sim >= 0.05):
                    selected.append({
                        "source": meta.get("source", "knowledge"),
                        "chunk": meta.get("chunk", 1),
                        "text": doc_text
                    })

        if not selected:
            return "", []

        context = "\n\n".join(f"[Source: {d['source']} | chunk {d['chunk']}]\n{d['text']}" for d in selected)
        sources = list(dict.fromkeys(d["source"] for d in selected))
        return context, sources
    except Exception as exc:
        print(f"ChromaDB retrieval error, falling back to TF-IDF: {exc}")
        return _retrieve_tfidf(query_clean, k=k)


def find_direct_answer(query: str) -> str:
    """Extract only the direct answer from the knowledge base for a specific query without raw excerpts or metadata."""
    root = Path(KNOWLEDGE_DIR).expanduser()
    if not root.exists() or not query.strip():
        return ""

    sections = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".md"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            import re
            parts = re.split(r"\n##\s+", text)
            for p in parts[1:]:
                lines = p.strip().split("\n", 1)
                header = lines[0].strip()
                body = lines[1].strip() if len(lines) > 1 else ""
                if body:
                    sections.append((header, body))
        except Exception:
            pass

    if not sections:
        return ""

    docs = [f"{h} {b}" for h, b in sections]
    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(docs)
        q_vec = vectorizer.transform([query])
        scores = cosine_similarity(q_vec, matrix).ravel()
        best_idx = scores.argmax()
        if scores[best_idx] > 0.08:
            return sections[best_idx][1].strip()
    except Exception:
        pass
    return ""


def extract_best_paragraph(query: str, text: str) -> str:
    """Extract only the single most relevant paragraph or answer from multi-chunk context."""
    import re
    cleaned = re.sub(r"\[Source:[^\]]+\]", "", text).strip()
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n|(?=##?\s+)", cleaned) if len(p.strip()) > 25]
    if not paragraphs:
        return cleaned

    try:
        vec = TfidfVectorizer(stop_words="english")
        matrix = vec.fit_transform(paragraphs)
        q_vec = vec.transform([query])
        sims = cosine_similarity(q_vec, matrix).ravel()
        if sims.max() < 0.05:
            return ""
        best_p = paragraphs[sims.argmax()]
        # Strip markdown or numbered FAQ header if leading
        best_p = re.sub(r"^(?:##?\s*|\d+\.\s*)[^\n]+\n", "", best_p).strip()
        return best_p
    except Exception:
        return ""
