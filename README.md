# 📄 Document Q&A — A RAG (Retrieval-Augmented Generation) System

Upload PDFs or text files and ask questions about them in plain English. The system retrieves the most relevant passages and uses an LLM to generate an answer **grounded in your documents**, with sources shown for every answer.

This is a from-scratch implementation of the RAG pattern used in production systems like customer-support bots, internal knowledge search, and document assistants.

---

## Why this project

Most fresher ML portfolios are all classification/regression on tabular data (fraud detection, churn, etc.). RAG demonstrates a different, currently in-demand skill set:

- Working with **embeddings** and **vector search** (not just scikit-learn models)
- Understanding how LLMs are grounded in external knowledge to reduce hallucination
- Building a real, deployable pipeline: ingestion → chunking → retrieval → generation
- Comfort with the modern GenAI stack (the thing most companies are actively hiring for)

---

## Architecture

```
 ┌─────────────┐     ┌───────────┐     ┌──────────────┐     ┌─────────────┐
 │  Upload PDF │ --> │  Chunk    │ --> │  Embed with  │ --> │  Store in   │
 │  / TXT file │     │  text     │     │  MiniLM      │     │  FAISS index│
 └─────────────┘     └───────────┘     └──────────────┘     └─────────────┘
                                                                     │
                                                                     ▼
 ┌─────────────┐     ┌───────────┐     ┌──────────────┐     ┌─────────────┐
 │   Answer    │ <-- │  Claude   │ <-- │  Prompt with │ <-- │  User asks  │
 │  + sources  │     │ generates │     │  retrieved   │     │  a question │
 │             │     │  answer   │     │  chunks      │     │             │
 └─────────────┘     └───────────┘     └──────────────┘     └─────────────┘
```

**Retrieval** (finding relevant text) is separate from **generation** (writing the answer) — that separation is the core idea of RAG, and it's what lets the model answer using information it was never trained on.

---

## Tech stack

| Component | Tool | Why |
|---|---|---|
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) | Free, runs locally on CPU, no API cost |
| Vector store | `FAISS` | Free, fast, industry-standard for similarity search |
| LLM | Anthropic Claude API | Generates the final grounded answer |
| UI | Streamlit | Fast to build, easy to deploy, looks professional |
| PDF parsing | `pypdf` | Lightweight, no external dependencies |

---

## Project structure

```
rag_qa_system/
├── app.py              # Streamlit UI (upload, ask, view answers + sources)
├── rag_pipeline.py      # Core logic: loading, chunking, embedding, retrieval, generation
├── requirements.txt      # Python dependencies
├── sample_docs/
│   └── sample.txt        # A sample document to test with immediately
└── README.md
```

`app.py` contains **no business logic** — it only handles UI. All the actual RAG logic lives in `rag_pipeline.py` and can be tested or reused independently (e.g., in a Jupyter notebook, or swapped into a FastAPI backend later). This separation is intentional and worth mentioning in an interview — it shows you think about code structure, not just "make it work."

---

## Setup (run locally)

**1. Clone / download this folder, then install dependencies:**

```bash
cd rag_qa_system
pip install -r requirements.txt
```

**2. Get a free Anthropic API key:**

- Go to https://console.anthropic.com
- Sign up, create an API key (there's a free trial credit)

**3. Run the app:**

```bash
streamlit run app.py
```

**4. In the browser tab that opens:**

- Paste your API key into the sidebar
- Upload `sample_docs/sample.txt` (or any PDF/TXT you like)
- Click **Build Index**
- Ask a question, e.g. *"What happens during the Calvin cycle?"*

The first run will download the embedding model (~90MB) — this needs an internet connection once, then it's cached locally.

---

## Deploying it for free (so recruiters can actually try it)

**Option A — Hugging Face Spaces (recommended, free, easiest):**

1. Create a free account at https://huggingface.co
2. Create a new **Space** → choose **Streamlit** as the SDK
3. Upload `app.py`, `rag_pipeline.py`, `requirements.txt`
4. In Space settings, add `ANTHROPIC_API_KEY` as a secret so it's not hardcoded
5. Your app gets a public URL you can put directly in your resume/LinkedIn

**Option B — Streamlit Community Cloud:**

1. Push this folder to a public GitHub repo
2. Go to https://streamlit.io/cloud → connect your GitHub → deploy
3. Add your API key under app secrets

Either way, put the **live link** in your resume, not just the GitHub repo — recruiters are far more likely to click a working demo than clone a repo.

---

## Ideas to extend this (good talking points in interviews)

- **Swap FAISS for a hosted vector DB** (Pinecone, Weaviate, Chroma Cloud) — shows you understand scaling beyond a local index
- **Add conversation memory** so follow-up questions ("what about the second stage?") work without repeating context
- **Add a "confidence" or "no answer found" check** — if retrieved chunks have low similarity scores, tell the user instead of forcing an answer
- **Support more file types**: .docx, .csv, web pages via URL
- **Add evaluation**: a small set of Q&A pairs to measure retrieval accuracy (this is a big deal in real RAG systems and shows maturity if you mention it)
- **Re-ranking**: after retrieving top-k chunks with FAISS, re-rank them with a cross-encoder for better precision

---

## What to say about this project on your resume / in interviews

> "Built an end-to-end RAG (Retrieval-Augmented Generation) Q&A system that lets users query custom documents. Implemented the full pipeline — chunking, embedding with sentence-transformers, similarity search with FAISS, and grounded answer generation with Claude — and deployed it as a public Streamlit app on Hugging Face Spaces."

That one sentence signals: you understand embeddings, vector search, LLM prompting, and deployment — which is exactly the GenAI skill set companies are screening for right now.
