# Learn RAG - Chat With Your Documents

[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-blue)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.8+-green)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

An educational RAG (Retrieval Augmented Generation) web application that teaches how modern AI systems combine document retrieval with language models to answer questions accurately.

🔗 **[Live Demo](https://huggingface.co/spaces/)** | 📖 **[GitHub Repo](https://github.com/sandip-patil/rag-learning-app)**

---

## 🎯 Project Goal

Build a **public, free-to-use, educational RAG app** where users can:
- Upload PDF or TXT documents
- Chat with their documents in real-time
- Understand how RAG works internally through detailed explanations
- See the exact process: retrieval → augmentation → generation

**Target Audience:** Students, developers, and AI enthusiasts learning about RAG systems.

---

## 📚 What is RAG?

**RAG (Retrieval Augmented Generation)** = Retrieval + Augmentation + Generation

### Why RAG?

| Problem | Solution |
|---------|----------|
| LLMs have knowledge cutoff | RAG brings current, document-specific knowledge |
| Hallucinations & false facts | RAG grounds answers in actual document content |
| Generic responses | RAG makes responses specific to your documents |
| Expensive fine-tuning | RAG doesn't require training, just retrieval |

---

## 🏗️ Architecture

### Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| **Document Loading** | PyPDF2 | Extract text from PDFs reliably |
| **Text Chunking** | Custom | Control chunk size & overlap |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) | Fast, lightweight, semantic |
| **Vector Search** | FAISS | Efficient similarity search (millions of vectors) |
| **LLM** | Groq (Llama 3.1 8B) | Fast, free-tier available, open model |
| **UI** | Streamlit | Rapid prototyping, great for demos |
| **Hosting** | HuggingFace Spaces | Free, easy deployment, built-in secrets |

### Folder Structure

```
rag-learning-app/
├── app.py                      # Streamlit UI
├── rag/
│   ├── __init__.py
│   ├── loader.py              # Load PDF/TXT
│   ├── chunker.py             # Split text into chunks
│   ├── embedder.py            # Convert chunks to vectors
│   ├── vector_store.py        # FAISS index management
│   └── rag_chain.py           # Orchestrate pipeline
├── assets/
│   └── rag_diagram.png        # Architecture diagram
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── .gitignore
```

---

## 🚀 How to Run Locally

### Prerequisites

- Python 3.8+
- Groq API key (free from [console.groq.com](https://console.groq.com))

### Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/sandip-patil/rag-learning-app.git
   cd rag-learning-app
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variable**
   ```bash
   export GROQ_API_KEY="your-api-key-here"  # On Windows: set GROQ_API_KEY=...
   ```

5. **Run the app**
   ```bash
   streamlit run app.py
   ```

   App opens at `http://localhost:8501`

### Testing the App

**Test with sample files:**

1. Create a test document (`test.txt`):
   ```
   Machine Learning is a subset of Artificial Intelligence.
   It enables computers to learn from data without explicit programming.
   
   Deep Learning uses neural networks with multiple layers.
   It powers computer vision, natural language processing, and more.
   
   LLMs (Large Language Models) are trained on massive text datasets.
   They can generate human-like text and understand context.
   ```

2. Upload to the app
3. Ask questions:
   - "What is machine learning?"
   - "How do LLMs work?"
   - "What are neural networks?"

---

## 🔧 Core Modules Explained

### `loader.py` - Document Loading
```python
# Load PDF or TXT, extract raw text
text = load_file(uploaded_file)
```

**Why it matters:** Different file formats need different parsing strategies.

### `chunker.py` - Text Chunking
```python
# Split text into 500-char chunks with 50-char overlap
chunks = chunk_text(text, chunk_size=500, overlap=50)
```

**Why overlap?** Avoids cutting important context at chunk boundaries.

### `embedder.py` - Vector Embeddings
```python
# Convert chunks to 384-dimensional vectors
embeddings = embedding_model.embed_chunks(chunks)
# Result: np.array shape (num_chunks, 384)
```

**Why embeddings?** Capture semantic meaning. Similar texts have similar vectors.

### `vector_store.py` - FAISS Index
```python
# Build searchable index
store.build_index(embeddings, chunks)

# Find top-5 most similar chunks to query
retrieved_chunks, distances = store.search(query_embedding, top_k=5)
```

**Why FAISS?** O(log n) search vs O(n) naive search. Handles millions of vectors.

### `rag_chain.py` - RAG Orchestration
```python
# Build pipeline
pipeline = RAGPipeline(groq_api_key)
metadata = pipeline.build_pipeline(file)  # Load → Chunk → Embed → Index

# Ask questions
result = pipeline.ask("What is RAG?", top_k=5)
# Returns: answer, retrieved_chunks, distances, prompt
```

---

## 📖 Learning Outcomes

After exploring this app, you'll understand:

✅ How embeddings capture semantic meaning  
✅ How vector databases work for similarity search  
✅ How to orchestrate a multi-stage RAG pipeline  
✅ Prompt engineering for grounded, factual answers  
✅ LLM API integration (Groq)  
✅ Streamlit for rapid prototyping  
✅ Deploying ML apps with Streamlit + HuggingFace  

---

## 🤝 Contributing

Found a bug? Have an idea? Open an issue or submit a PR!

```bash
git checkout -b feature/your-feature
git commit -m "Add your feature"
git push origin feature/your-feature
```

---

## 📄 License

MIT License - feel free to use this for learning and projects!

---

## 👤 About the Author

**Sandip Patil**
- 🔗 GitHub: [@sandip-patil](https://github.com/)
- 💼 LinkedIn: [Sandip Patil](https://linkedin.com/)

Built to teach, built with ❤️

---

## 🙏 Acknowledgments

- **Groq** for fast LLM inference
- **HuggingFace** for Transformers & Spaces
- **FAISS** for vector search
- **Streamlit** for awesome UI framework
- The open-source AI community

---

## 📚 Resources to Learn More

- [What is RAG?](https://aws.amazon.com/what-is/retrieval-augmented-generation/)
- [Vector Databases Explained](https://www.pinecone.io/learn/vector-database/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
- [Groq API Docs](https://groq.com/)
- [Streamlit Docs](https://docs.streamlit.io/)

---

**Star ⭐ this repo if you learned something!**

*Last updated: January 17, 2026*
