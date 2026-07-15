# Document Q&A System using Retrieval-Augmented Generation (RAG)

A Retrieval-Augmented Generation (RAG) application that enables users to upload documents and ask natural-language questions about their content.

The system combines semantic search using FAISS and Sentence Transformers with Google's Gemini API to generate accurate, context-aware answers grounded in the uploaded documents.

---

## Features

* Upload and process PDF and TXT documents
* Automatic document chunking with overlap
* Semantic search using vector embeddings
* FAISS-powered similarity retrieval
* Context-aware answer generation using Google Gemini
* Source chunk attribution for transparency
* Interactive Streamlit web interface
* Modular and scalable architecture

---

## Architecture

```text
User Uploads Document
        │
        ▼
Document Loader (PDF/TXT)
        │
        ▼
Text Chunking
        │
        ▼
Sentence Transformer Embeddings
(all-MiniLM-L6-v2)
        │
        ▼
FAISS Vector Store
        │
        ▼
User Question
        │
        ▼
Question Embedding
        │
        ▼
Similarity Search
        │
        ▼
Top Relevant Chunks
        │
        ▼
Context + User Query
        │
        ▼
Google Gemini
        │
        ▼
Generated Answer
```

---

## Tech Stack

| Component           | Technology            |
| ------------------- | --------------------- |
| Frontend            | Streamlit             |
| LLM                 | Google Gemini         |
| Embeddings          | Sentence Transformers |
| Embedding Model     | all-MiniLM-L6-v2      |
| Vector Search       | FAISS                 |
| Document Processing | PyPDF                 |
| Language            | Python                |

---

## Project Structure

```text
rag_qa_system/
│
├── app.py                 # Streamlit application
├── rag_pipeline.py        # RAG pipeline implementation
├── requirements.txt       # Dependencies
├── sample_docs/           # Sample documents
├── .gitignore
└── README.md
```

---

## How It Works

1. User uploads one or more documents.
2. Documents are loaded and parsed.
3. Text is split into manageable chunks.
4. Embeddings are generated for each chunk.
5. Chunks are indexed using FAISS.
6. User submits a question.
7. The system retrieves the most relevant chunks.
8. Retrieved context is sent to Gemini.
9. Gemini generates an answer based on the retrieved information.

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/tanayaaa123/RAG_QA_SYSTEM.git
cd RAG_QA_SYSTEM
```

### Create a Virtual Environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Setup

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_api_key_here
```

Make sure `.env` is included in your `.gitignore`.

---

## Running the Application

```bash
streamlit run app.py
```

The application will launch in your browser.

---

## Example Workflow

### Upload Documents

Upload one or more PDF or TXT files.

### Ask Questions

Example queries:

```text
What are the key topics discussed in this document?
```

```text
Summarize the main findings.
```

```text
What recommendations are provided?
```

### Receive Answers

The system retrieves relevant document sections and generates a grounded response using Gemini.

---

## Why This Project?

Large Language Models are powerful, but they cannot reliably answer questions about private or custom documents without access to the relevant information.

This project demonstrates how Retrieval-Augmented Generation (RAG) can be used to bridge that gap by combining:

* Information Retrieval
* Semantic Search
* Vector Databases
* Embeddings
* Large Language Models

The result is a system capable of answering questions based on user-provided knowledge rather than relying solely on the model's training data.

---

## Skills Demonstrated

* Retrieval-Augmented Generation (RAG)
* Semantic Search
* Vector Databases (FAISS)
* Embedding Models
* Large Language Models (Gemini)
* Prompt Engineering
* Information Retrieval
* Streamlit Development
* Python Application Design

---

## Future Improvements

* Persistent vector storage
* Multi-document collections
* Conversation memory
* Hybrid search (BM25 + Vector Search)
* Source citations with confidence scores
* Docker deployment
* FastAPI backend
* User authentication

---

## Security

API keys and credentials are never stored in the repository.

Sensitive configuration should be managed through environment variables and a local `.env` file that is excluded from version control.

---

## Author

**Tanaya**

Built as a learning project to explore Retrieval-Augmented Generation (RAG), semantic search, vector databases, and LLM-powered question answering.
