"""
rag_pipeline.py
----------------
Core logic for a Retrieval-Augmented Generation (RAG) Q&A system.

Pipeline:
1. Load documents (PDF / TXT)
2. Split into overlapping chunks
3. Embed chunks using a sentence-transformer model (local, free)
4. Store embeddings in a FAISS vector index (local, free)
5. On a query: embed the query, retrieve top-k similar chunks
6. Pass retrieved chunks + query to an LLM to generate a grounded answer

This file has NO UI code — app.py (Streamlit) imports from here.
Keeping logic separate from UI is good practice and makes the
pipeline reusable / testable.
"""

import os
import re
from dataclasses import dataclass
from typing import List

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from google import genai



# 1. Data structures


@dataclass
class Chunk:
    text: str
    source: str        # filename the chunk came from
    chunk_id: int       # position within the document



# 2. Document loading


def load_pdf(file_path: str) -> str:
    """Extract raw text from a PDF file."""
    reader = PdfReader(file_path)
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text_parts.append(page_text)
    return "\n".join(text_parts)


def load_txt(file_path: str) -> str:
    """Read raw text from a .txt file."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_document(file_path: str) -> str:
    """Dispatch to the right loader based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".txt":
        return load_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .pdf or .txt")



# 3. Chunking


def clean_text(text: str) -> str:
    """Collapse excessive whitespace/newlines."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, source: str, chunk_size: int = 500, overlap: int = 50) -> List[Chunk]:
    """
    Split text into overlapping word-based chunks.

    chunk_size: number of words per chunk
    overlap: number of words shared between consecutive chunks
             (overlap helps avoid cutting a relevant sentence in half
             right at a chunk boundary)
    """
    text = clean_text(text)
    words = text.split(" ")
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_str = " ".join(chunk_words).strip()
        if chunk_str:
            chunks.append(Chunk(text=chunk_str, source=source, chunk_id=chunk_id))
            chunk_id += 1
        start += chunk_size - overlap  # move forward, keeping overlap

    return chunks



# 4. Embedding + Vector index


class VectorStore:
    """
    Wraps a sentence-transformer embedding model + a FAISS index.
    This is the 'retrieval' half of RAG.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # all-MiniLM-L6-v2: small, fast, free, runs on CPU, good enough for demos.
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks: List[Chunk] = []

    def build(self, chunks: List[Chunk]):
        """Embed all chunks and build a FAISS index from scratch."""
        self.chunks = chunks
        texts = [c.text for c in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        embeddings = np.array(embeddings).astype("float32")

        dim = embeddings.shape[1]
        # Inner product on normalized vectors == cosine similarity
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def search(self, query: str, k: int = 4) -> List[Chunk]:
        """Return the top-k most relevant chunks for a query."""
        if self.index is None:
            raise RuntimeError("Index not built yet. Call .build() first.")

        query_vec = self.model.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(query_vec, k)
        results = [self.chunks[i] for i in indices[0] if i != -1]
        return results


# 5. Generation (the "G" in RAG)


def build_prompt(query: str, context_chunks: List[Chunk]) -> str:
    """
    Construct the prompt sent to the LLM: retrieved context + question.
    Explicitly instruct the model to answer ONLY from context and to
    say so if the answer isn't present — this is what makes it a
    grounded / trustworthy RAG answer instead of a hallucination risk.
    """
    context_str = "\n\n".join(
        f"[Source: {c.source}, chunk {c.chunk_id}]\n{c.text}" for c in context_chunks
    )

    prompt = f"""You are a helpful assistant that answers questions using ONLY the context provided below.
If the answer cannot be found in the context, say "I couldn't find this in the provided documents."
Always be concise and cite which source(s) you used.

CONTEXT:
{context_str}

QUESTION:
{query}

ANSWER:"""
    return prompt


# GenAI client is created lazily and cached so we don't reconnect on every
# single call to generate_answer() (Streamlit reruns this file top-to-bottom
# on every interaction, so caching matters here).
_client = None
_client_api_key = None


def _get_client(api_key: str) -> genai.Client:
    global _client, _client_api_key
    if _client is None or _client_api_key != api_key:
        _client = genai.Client(api_key=api_key)
        _client_api_key = api_key
    return _client


def generate_answer(query: str, context_chunks: List[Chunk], api_key: str) -> str:
    client = _get_client(api_key)

    prompt = build_prompt(query, context_chunks)

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
    )

    return response.text



# 6. End-to-end convenience function


def answer_question(query: str, store: VectorStore, api_key: str, k: int = 4):
    """
    Full pipeline for one question:
    retrieve relevant chunks -> generate answer -> return answer + sources
    """
    retrieved = store.search(query, k=k)
    answer = generate_answer(query, retrieved, api_key)
    return answer, retrieved