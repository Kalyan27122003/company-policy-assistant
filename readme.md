# Agentic Company Policy Assistant

An enterprise-grade **Agentic RAG (Retrieval-Augmented Generation)** system that answers company policy questions using real organizational documents.

This project demonstrates how to build a **reliable, hallucination-safe GenAI application** using LangGraph, LangChain, and vector databases.

---

## 🚀 Features

- Agentic RAG architecture using **LangGraph**
- Query classification (HR, IT, Legal, Travel, Compensation)
- Metadata-filtered retrieval using **ChromaDB**
- Semantic chunking using Markdown headers
- Guardrails to prevent hallucinations
- Source attribution for every answer
- Streamlit-based chat interface

---

## 🧠 Architecture

### 📂 Document Ingestion Pipeline

The system uses a multi-step document ingestion pipeline to normalize enterprise documents before retrieval:

1. **System File Loading**
   - Enterprise policy documents are loaded directly from the local file system (`.docx` format).

2. **Document Normalization**
   - Word documents are converted to Markdown for consistent text structure.

3. **Markdown Ingestion**
   - Markdown files are loaded as LangChain Documents with metadata.

4. **Semantic Chunking**
   - Documents are chunked based on Markdown headers to preserve policy sections.

5. **Vector Storage**
   - Chunks are embedded and stored in ChromaDB with policy-type metadata.

---

### 🤖 Agentic RAG Flow

```text
User Query
   ↓
Query Classification Agent
   ↓
Metadata-Filtered Vector Retrieval
   ↓
Answer Generation Agent
   ↓
Grounded Response + Sources


## ▶️ How to Run

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Convert system documents to Markdown
python src/load_system_docs_to_md.py

# 4. Build vector database
python src/vector_store.py

# 5. Run the application
streamlit run app.py
