"""
Learn RAG - Chat with Your Documents
Educational RAG (Retrieval Augmented Generation) web application
Built by Sandip Patil | GitHub: github.com/
"""

import streamlit as st
import os
from datetime import datetime
from dotenv import load_dotenv
from rag.rag_chain import RAGPipeline

# Load environment variables from .env file
load_dotenv()


# ==================== Page Config ====================
st.set_page_config(
    page_title="Learn RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== Session State ====================
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False
if "pipeline_metadata" not in st.session_state:
    st.session_state.pipeline_metadata = None


# ==================== Styling ====================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5em;
        font-weight: bold;
        color: #FF6B6B;
        margin-bottom: 0.5em;
    }
    .section-header {
        font-size: 1.5em;
        font-weight: bold;
        color: #4ECDC4;
        margin-top: 1.5em;
        margin-bottom: 0.5em;
    }
    .rag-step {
        background-color: #F0F0F0;
        padding: 1em;
        border-radius: 0.5em;
        margin-bottom: 0.5em;
        border-left: 4px solid #4ECDC4;
    }
    .success-box {
        background-color: #D4EDDA;
        padding: 1em;
        border-radius: 0.5em;
        border-left: 4px solid #28A745;
        color: #155724;
    }
    .info-box {
        background-color: #D1ECF1;
        padding: 1em;
        border-radius: 0.5em;
        border-left: 4px solid #17A2B8;
        color: #0C5460;
    }
    .footer {
        margin-top: 3em;
        padding-top: 2em;
        border-top: 1px solid #E0E0E0;
        font-size: 0.85em;
        color: #666;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ==================== Header ====================
st.markdown('<div class="main-header">🤖 Learn RAG</div>', unsafe_allow_html=True)
st.markdown('### Chat with Your Documents')
st.markdown('*Educational RAG (Retrieval Augmented Generation) web application*')
st.markdown('Built by **Sandip Patil** | [GitHub](https://github.com/)')

st.divider()

# ==================== Main Content ====================
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="section-header">📚 What is RAG?</div>', unsafe_allow_html=True)
    
    st.markdown("""
    **RAG (Retrieval Augmented Generation)** combines information retrieval with AI to answer questions based on your documents.
    """)
    
    # R-A-G explanation
    col_r, col_a, col_g = st.columns(3)
    
    with col_r:
        st.markdown("""
        #### **R** - Retrieval
        Find relevant chunks from your document using semantic search
        """)
        st.info("Uses embeddings to find similar content", icon="🔍")
    
    with col_a:
        st.markdown("""
        #### **A** - Augmented
        Combine retrieved chunks with the user's question
        """)
        st.info("Creates rich context for the LLM", icon="🧩")
    
    with col_g:
        st.markdown("""
        #### **G** - Generation
        Generate answers using an LLM
        """)
        st.info("Produces human-like responses", icon="✨")

with col2:
    st.markdown('<div class="section-header">🔄 Pipeline Overview</div>', unsafe_allow_html=True)
    st.markdown("""
    1. **Load** PDF/TXT
    2. **Chunk** text (overlap)
    3. **Embed** chunks
    4. **Index** vectors (FAISS)
    5. **Search** by query
    6. **Generate** answer
    """)

st.divider()

# ==================== File Upload ====================
st.markdown('<div class="section-header">📤 Upload Your Document</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a PDF or TXT file",
    type=["pdf", "txt"],
    help="Maximum 20 MB"
)

if uploaded_file is not None:
    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
    
    if file_size_mb > 20:
        st.error(f"❌ File size ({file_size_mb:.2f} MB) exceeds 20 MB limit")
    else:
        st.markdown(f"""
        <div class="info-box">
        📄 File: <b>{uploaded_file.name}</b><br>
        Size: {file_size_mb:.2f} MB
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Build RAG Pipeline", use_container_width=True):
            with st.spinner("Building RAG pipeline... This may take a minute."):
                try:
                    # Initialize pipeline with API key
                    groq_api_key = os.getenv("GROQ_API_KEY")
                    if not groq_api_key:
                        st.error("❌ GROQ_API_KEY environment variable not set")
                    else:
                        # Build pipeline
                        st.session_state.pipeline = RAGPipeline(groq_api_key=groq_api_key)
                        st.session_state.pipeline_metadata = st.session_state.pipeline.build_pipeline(uploaded_file)
                        st.session_state.file_uploaded = True
                        
                        # Show metadata
                        st.success("✓ Pipeline built successfully!")
                        
                        metadata = st.session_state.pipeline_metadata
                        st.markdown(f"""
                        <div class="success-box">
                        <b>Pipeline Statistics:</b><br>
                        📝 File Size: {metadata['file_size_bytes']:,} characters<br>
                        🧩 Chunks: {metadata['num_chunks']}<br>
                        🧠 Embedding Model: {metadata['embedding_model']}<br>
                        📐 Embedding Dimension: {metadata['embedding_dim']}<br>
                        📑 Vector Index Size: {metadata['index_size']}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Error building pipeline: {str(e)}")

# ==================== Chat Interface ====================
if st.session_state.file_uploaded and st.session_state.pipeline:
    st.divider()
    st.markdown('<div class="section-header">💬 Ask Your Document</div>', unsafe_allow_html=True)
    
    question = st.text_input(
        "Enter your question:",
        placeholder="What is the document about?",
        key="question_input"
    )
    
    col_ask, col_reset = st.columns([4, 1])
    
    with col_ask:
        ask_button = st.button("🔍 Search & Answer", use_container_width=True)
    
    with col_reset:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.pipeline = None
            st.session_state.file_uploaded = False
            st.session_state.pipeline_metadata = None
            st.rerun()
    
    if ask_button and question:
        with st.spinner("Searching and generating answer..."):
            try:
                result = st.session_state.pipeline.ask(question, top_k=5)
                
                # Display answer
                st.markdown('<div class="section-header">✅ Answer</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="success-box">
                {result['answer']}
                </div>
                """, unsafe_allow_html=True)
                
                # Display retrieved chunks
                with st.expander("📑 Retrieved Chunks (Top 5)", expanded=True):
                    st.markdown("These are the chunks from your document that were used to generate the answer:")
                    for i, chunk in enumerate(result['retrieved_chunks'], 1):
                        st.markdown(f"""
                        <div class="rag-step">
                        <b>Chunk {i}</b> (Relevance: {1/(1+result['distances'][i-1]):.1%})<br>
                        {chunk}
                        </div>
                        """, unsafe_allow_html=True)
                
                # Display final prompt
                with st.expander("🤖 Final Prompt Sent to LLM"):
                    st.code(result['prompt'], language="text")
                
                # How it works
                with st.expander("📖 How RAG Answered Your Question"):
                    st.markdown("""
                    1. **Your Question Embedded**: We converted your question into a 384-dimensional vector using sentence-transformers
                    2. **Similarity Search**: Used FAISS to find the 5 most semantically similar chunks from your document
                    3. **Context Assembly**: Combined your question with the retrieved chunks
                    4. **LLM Generation**: Sent the augmented prompt to Llama 3.1 8B via Groq
                    5. **Answer Retrieved**: The model generated a contextual answer using ONLY the provided context
                    """)
                
            except Exception as e:
                st.error(f"❌ Error processing question: {str(e)}")
    
    elif ask_button and not question:
        st.warning("⚠️ Please enter a question first")


# ==================== Footer ====================
st.markdown("""
<div class="footer">
<p>
📚 <b>Learn RAG</b> - Educational RAG Demo<br>
Tech Stack: FAISS + Sentence Transformers + Llama 3.1 (Groq) + Streamlit<br>
Built by <b>Sandip Patil</b> | 
<a href="https://github.com/">GitHub</a>
</p>
</div>
""", unsafe_allow_html=True)
