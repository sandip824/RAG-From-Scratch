"""
Learn RAG - Chat with Your Documents
Educational RAG (Retrieval Augmented Generation) web application
Built by Sandip Patil | GitHub: github.com/
"""

import streamlit as st
import os
import pandas as pd
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

# ==================== Check for GROQ API Key ====================
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("""
    ❌ **GROQ_API_KEY not found!**
    
    **For HuggingFace Space Admins:**
    1. Go to Space Settings → Variables and secrets
    2. Click "New secret"
    3. Add `GROQ_API_KEY` with your Groq API key value
    4. Restart the Space
    
    **For local development:**
    Create a `.env` file in the project root with:
    ```
    GROQ_API_KEY=your_groq_api_key_here
    ```
    """)
    st.stop()

# ==================== Session State ====================
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False
if "pipeline_metadata" not in st.session_state:
    st.session_state.pipeline_metadata = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "top_k" not in st.session_state:
    st.session_state.top_k = 5
if "show_chunk_preview" not in st.session_state:
    st.session_state.show_chunk_preview = True
# PHASE 4: Inside RAG Mode
if "show_rag_internals" not in st.session_state:
    st.session_state.show_rag_internals = False
# PHASE 5: RAG Lab Experiment Settings
if "lab_chunk_size" not in st.session_state:
    st.session_state.lab_chunk_size = 500
if "lab_chunk_overlap" not in st.session_state:
    st.session_state.lab_chunk_overlap = 50
if "lab_top_k" not in st.session_state:
    st.session_state.lab_top_k = 5
if "experiment_mode" not in st.session_state:
    st.session_state.experiment_mode = False
if "experiment_result_a" not in st.session_state:
    st.session_state.experiment_result_a = None
if "experiment_result_b" not in st.session_state:
    st.session_state.experiment_result_b = None
if "experiment_question" not in st.session_state:
    st.session_state.experiment_question = ""


# ==================== Sidebar Controls ====================
with st.sidebar:
    st.markdown("## ⚙️ RAG Controls")
    st.divider()
    
    st.markdown("### Retrieval Settings")
    st.session_state.top_k = st.slider(
        "Number of chunks to retrieve (Top-K)",
        min_value=1,
        max_value=10,
        value=st.session_state.top_k,
        help="Higher values = more context but may introduce noise"
    )
    
    st.session_state.show_chunk_preview = st.checkbox(
        "Show retrieved chunks",
        value=st.session_state.show_chunk_preview,
        help="Display the document chunks used to generate the answer"
    )
    
    st.divider()
    st.markdown("### 🔍 Learning Mode")
    st.session_state.show_rag_internals = st.checkbox(
        "Show RAG Internals",
        value=st.session_state.show_rag_internals,
        help="See embeddings, similarity scores, and exact LLM prompt"
    )
    
    st.divider()
    st.markdown("### About RAG")
    st.markdown("""
    **RAG Pipeline:**
    1. Retrieval → Find relevant chunks
    2. Augmentation → Add context
    3. Generation → Create answer
    
    **Model:** Llama 3.1 8B
    **Embeddings:** all-MiniLM-L6-v2
    """)


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
    .chat-user-message {
        background-color: #FF6B6B;
        color: white;
        padding: 1em;
        border-radius: 0.75em;
        margin-bottom: 0.5em;
        max-width: 80%;
        margin-left: auto;
        word-wrap: break-word;
    }
    .chat-assistant-message {
        background-color: #E8F5E9;
        color: #1B5E20;
        padding: 1em;
        border-radius: 0.75em;
        margin-bottom: 0.5em;
        max-width: 80%;
        word-wrap: break-word;
    }
    .chat-role {
        font-weight: bold;
        margin-bottom: 0.25em;
        font-size: 0.9em;
    }
    .pipeline-stage {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5em;
        border-radius: 0.75em;
        text-align: center;
        font-weight: bold;
        margin: 0.5em;
        flex: 1;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .pipeline-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 2em 0;
        flex-wrap: wrap;
    }
    .pipeline-arrow {
        color: #FF6B6B;
        font-size: 1.5em;
        margin: 0 0.5em;
    }
    .edu-panel {
        background-color: #FFF8E1;
        border-left: 4px solid #FFA726;
        padding: 1em;
        border-radius: 0.5em;
        margin: 1em 0;
    }
    .edu-title {
        font-weight: bold;
        color: #E65100;
        margin-bottom: 0.5em;
    }
    .chunk-preview {
        background-color: #F5F5F5;
        border-left: 4px solid #FF6B6B;
        padding: 0.75em;
        border-radius: 0.5em;
        margin-bottom: 0.5em;
        font-size: 0.9em;
    }
    .metric-card {
        background-color: #E3F2FD;
        border-radius: 0.5em;
        padding: 1em;
        text-align: center;
        border-left: 4px solid #2196F3;
    }
    .metric-value {
        font-size: 1.8em;
        font-weight: bold;
        color: #1976D2;
    }
    .metric-label {
        font-size: 0.85em;
        color: #555;
        margin-top: 0.25em;
    }
</style>
""", unsafe_allow_html=True)


# ==================== Header ====================
st.markdown('<div class="main-header">🤖 Learn RAG</div>', unsafe_allow_html=True)
st.markdown('### Chat with Your Documents')
st.markdown('*Educational RAG (Retrieval Augmented Generation) web application*')
st.markdown('Built by **Sandip Patil** | [GitHub](https://github.com/)')

st.divider()

# ==================== Main Navigation Tabs ====================
main_tab1, main_tab2 = st.tabs(["📚 Learn RAG", "🧪 RAG Lab (Experiment)"])

with main_tab1:
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
                            
                            # ==================== RAG Playground - Chunking Preview ====================
                            st.divider()
                            st.markdown('<div class="section-header">🎓 RAG Playground</div>', unsafe_allow_html=True)
                            
                            # Show RAG pipeline stages
                            st.markdown("""
                            <div class="pipeline-container">
                            <div class="pipeline-stage">📥 Load</div>
                            <div class="pipeline-arrow">→</div>
                            <div class="pipeline-stage">✂️ Chunk</div>
                            <div class="pipeline-arrow">→</div>
                            <div class="pipeline-stage">🧠 Embed</div>
                            <div class="pipeline-arrow">→</div>
                            <div class="pipeline-stage">🗂️ Index</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Tab 1: Chunking preview
                            tab1, tab2, tab3 = st.tabs(["📋 Chunking Preview", "📊 Pipeline Info", "📖 How It Works"])
                            
                            with tab1:
                                st.markdown("**First 3 Chunks from Your Document:**")
                                chunks = st.session_state.pipeline.chunks[:3]
                                for i, chunk in enumerate(chunks, 1):
                                    st.markdown(f"""
                                    <div class="chunk-preview">
                                    <b>Chunk {i}</b> ({len(chunk)} chars)<br>
                                    <small>{chunk[:150]}...</small>
                                    </div>
                                    """, unsafe_allow_html=True)
                                st.info(f"Total {len(st.session_state.pipeline.chunks)} chunks created with 50-char overlap")
                            
                            with tab2:
                                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                            with col_m1:
                                st.markdown(f"""
                                <div class="metric-card">
                                <div class="metric-value">{metadata['num_chunks']}</div>
                                <div class="metric-label">Chunks</div>
                                </div>
                                """, unsafe_allow_html=True)
                            with col_m2:
                                st.markdown(f"""
                                <div class="metric-card">
                                <div class="metric-value">{metadata['embedding_dim']}</div>
                                <div class="metric-label">Vector Dim</div>
                                </div>
                                """, unsafe_allow_html=True)
                            with col_m3:
                                st.markdown(f"""
                                <div class="metric-card">
                                <div class="metric-value">{metadata['file_size_bytes']:,}</div>
                                <div class="metric-label">Characters</div>
                                </div>
                                """, unsafe_allow_html=True)
                            with col_m4:
                                st.markdown(f"""
                                <div class="metric-card">
                                <div class="metric-value">FAISS</div>
                                <div class="metric-label">Vector DB</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            st.markdown("**Technical Details:**")
                            st.markdown(f"""
                            - **Embedding Model**: {metadata['embedding_model']}
                            - **Chunk Size**: 500 characters
                            - **Chunk Overlap**: 50 characters
                            - **Vector Store**: FAISS IndexFlatL2
                            - **Distance Metric**: L2 (Euclidean)
                            """)
                        
                        with tab3:
                            st.markdown("""
                            **Step 1: Load** 📥
                            We read your PDF/TXT file and extract all text.
                            
                            **Step 2: Chunk** ✂️
                            Text is split into overlapping chunks to maintain context between boundaries.
                            
                            **Step 3: Embed** 🧠
                            Each chunk is converted to a 384-dimensional vector using sentence-transformers.
                            This captures semantic meaning in a way computers can understand.
                            
                            **Step 4: Index** 🗂️
                            Vectors are stored in FAISS (Facebook AI Similarity Search) for fast retrieval.
                            """)
                        
                        st.rerun()
                    
                    except Exception as e:
                        st.error(f"❌ Error building pipeline: {str(e)}")

    # ==================== Chat Interface ====================
    if st.session_state.file_uploaded and st.session_state.pipeline:
        st.divider()
        st.markdown('<div class="section-header">💬 Ask Your Document</div>', unsafe_allow_html=True)
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown("**Conversation History:**")
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-user-message">
                    <div class="chat-role">🧑 You</div>
                    {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-assistant-message">
                    <div class="chat-role">🤖 Assistant</div>
                    {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
            
            st.divider()
        
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
                st.session_state.chat_history = []
                st.session_state.last_result = None
                st.rerun()
        
        if ask_button and question:
            with st.spinner("Searching and generating answer..."):
                try:
                    # Use top_k from sidebar
                    result = st.session_state.pipeline.ask(question, top_k=st.session_state.top_k)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": question
                    })
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result['answer']
                    })
                    st.session_state.last_result = result
                    
                    # Rerun to display updated chat history
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error processing question: {str(e)}")
        
        elif ask_button and not question:
            st.warning("⚠️ Please enter a question first")
        
        # Display details of last result
        if st.session_state.last_result:
            st.divider()
            st.markdown('<div class="section-header">📊 Answer Details & Learning Lab</div>', unsafe_allow_html=True)
            
            # Show the 4-step RAG process
            st.markdown("**RAG Process Visualization:**")
            st.markdown("""
            <div class="pipeline-container">
            <div class="pipeline-stage">1️⃣ Embed Question</div>
            <div class="pipeline-arrow">→</div>
            <div class="pipeline-stage">2️⃣ Retrieve Chunks</div>
            <div class="pipeline-arrow">→</div>
            <div class="pipeline-stage">3️⃣ Augment Prompt</div>
            <div class="pipeline-arrow">→</div>
            <div class="pipeline-stage">4️⃣ Generate Answer</div>
            </div>
            """, unsafe_allow_html=True)
            
            # ==================== PHASE 4: INSIDE RAG INTERNALS ====================
            if st.session_state.show_rag_internals:
                st.divider()
                st.markdown('<div class="section-header">🔬 RAG Internals (Learning Mode)</div>', unsafe_allow_html=True)
                
                # Panel 1: Query Embedding
                with st.expander("1️⃣ Query Embedding Details", expanded=True):
                    st.markdown("""
                    <div class="edu-panel">
                    <div class="edu-title">What's happening?</div>
                    Your question is converted into a numeric vector. This allows us to compare semantic similarity in high-dimensional space.
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col_emb1, col_emb2, col_emb3 = st.columns(3)
                    with col_emb1:
                        st.metric("Model", st.session_state.last_result.get('embedding_model', 'N/A'))
                    with col_emb2:
                        st.metric("Vector Dim", st.session_state.last_result.get('embedding_dim', 384))
                    with col_emb3:
                        st.metric("Vector Type", "Float32")
                    
                    st.markdown("**First 12 embedding values:**")
                    embedding = st.session_state.last_result.get('query_embedding', [])
                    if embedding:
                        emb_sample = embedding[:12]
                        emb_df = pd.DataFrame({
                            'Index': list(range(len(emb_sample))),
                            'Value': [f"{v:.6f}" for v in emb_sample]
                        })
                        st.dataframe(emb_df, use_container_width=True)
                        st.caption(f"(Showing first 12 of {len(embedding)} dimensions)")
                
                # Panel 2: Similarity Search Table
                with st.expander("2️⃣ Similarity Search Results", expanded=True):
                    st.markdown("""
                    <div class="edu-panel">
                    <div class="edu-title">What's happening?</div>
                    These chunks were selected because their vectors are closest to your question vector in embedding space.
                    Lower distances = higher similarity = more relevant to your question.
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.session_state.last_result['retrieved_chunks']:
                        search_data = []
                        for i, chunk in enumerate(st.session_state.last_result['retrieved_chunks'], 1):
                            search_data.append({
                                'Rank': i,
                                'Chunk Preview': chunk[:100] + '...' if len(chunk) > 100 else chunk,
                                'Distance': f"{st.session_state.last_result['distances'][i-1]:.4f}",
                                'Relevance': f"{st.session_state.last_result['relevance_scores'][i-1]:.1%}"
                            })
                        
                        search_df = pd.DataFrame(search_data)
                        st.dataframe(search_df, use_container_width=True)
                
                # Panel 3: Context Assembly
                with st.expander("3️⃣ Context Assembly View", expanded=True):
                    st.markdown("""
                    <div class="edu-panel">
                    <div class="edu-title">What's happening?</div>
                    This is exactly what the LLM sees. Your question is augmented with retrieved knowledge from your document.
                    This is the core idea behind RAG - augmenting the prompt with context.
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("**Full Prompt Sent to LLM:**")
                    st.code(st.session_state.last_result['prompt'], language="text")
            
            # Check if no chunks were found
            if st.session_state.last_result.get('no_chunks_found', False):
                st.warning("⚠️ No relevant chunks found in the document for this query.")
            
            # Create tabs for different views
            result_tab1, result_tab2, result_tab3, result_tab4 = st.tabs(
                ["🔍 Step 1: Question Embedding", 
                 "📑 Step 2: Retrieved Chunks", 
                 "🤖 Step 3: Augmented Prompt",
                 "✨ Step 4: Generated Answer"]
            )
            
            with result_tab1:
                st.markdown("""
                <div class="edu-panel">
                <div class="edu-title">What's happening?</div>
                Your question is converted into a 384-dimensional vector using the same embedding model as the document chunks.
                This allows us to compare semantic similarity in vector space.
                </div>
                """, unsafe_allow_html=True)
                st.info(f"✅ Question: *\"{st.session_state.chat_history[-2]['content']}\"* has been embedded into vector space")
                st.markdown("**Vector Representation:** 384-dimensional dense vector")
                st.markdown("**Embedding Model:** all-MiniLM-L6-v2")
            
            with result_tab2:
                st.markdown("""
                <div class="edu-panel">
                <div class="edu-title">What's happening?</div>
                Using L2 distance metric, we find the most semantically similar chunks from your document.
                Lower distances = more relevant chunks.
                </div>
                """, unsafe_allow_html=True)
                
                if st.session_state.show_chunk_preview and st.session_state.last_result['retrieved_chunks']:
                    st.markdown(f"**Found {len(st.session_state.last_result['retrieved_chunks'])} relevant chunks:**")
                    for i, chunk in enumerate(st.session_state.last_result['retrieved_chunks'], 1):
                        relevance = st.session_state.last_result['relevance_scores'][i-1]
                        distance = st.session_state.last_result['distances'][i-1]
                        
                        col_chunk1, col_chunk2 = st.columns([3, 1])
                        with col_chunk1:
                            st.markdown(f"""
                            <div class="chunk-preview">
                            <b>Chunk {i}</b><br>
                            {chunk}
                            </div>
                            """, unsafe_allow_html=True)
                        with col_chunk2:
                            st.metric("Relevance", f"{relevance:.1%}", delta=f"dist: {distance:.3f}")
                else:
                    st.info("Chunk preview disabled. Enable it in sidebar to see retrieved chunks.")
            
            with result_tab3:
                st.markdown("""
                <div class="edu-panel">
                <div class="edu-title">What's happening?</div>
                The retrieved chunks are combined with your question to create a rich context.
                This augmented prompt is then sent to the LLM for generation.
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("**System Instructions:**")
                st.code("You are a helpful tutor who answers questions using ONLY the provided context.\nIf information is not in the context, say so clearly.", language="text")
                
                st.markdown("**Full Prompt:**")
                st.code(st.session_state.last_result['prompt'], language="text")
            
            with result_tab4:
                st.markdown("""
                <div class="edu-panel">
                <div class="edu-title">What's happening?</div>
                The Llama 3.1 8B model processes the augmented prompt and generates a contextual answer.
                All responses are grounded in the retrieved document chunks.
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("**Generated Answer:**")
                st.markdown(f"""
                <div class="success-box">
                {st.session_state.last_result['answer']}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("**Model Details:**")
                st.markdown("""
                - **Model:** Llama 3.1 8B Instant
                - **Provider:** Groq (Fast LLM Inference)
                - **Temperature:** 0.7 (balanced creativity)
                - **Max Tokens:** 500
                """)
            
            # Educational summary section
            st.divider()
            st.markdown('<div class="section-header">📚 Learning Summary</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="edu-panel">
            <div class="edu-title">✨ How RAG Works</div>
            
            **RAG** (Retrieval Augmented Generation) combines three key components:
            
            1. **Retrieval (R)**: Find relevant information from your documents using semantic search
            2. **Augmentation (A)**: Combine retrieved context with the user's question  
            3. **Generation (G)**: Use an LLM to generate answers grounded in the context
            
            **Why RAG matters:**
            - ✅ Answers are grounded in your actual documents
            - ✅ Reduces hallucination (LLM won't make up facts)
            - ✅ Works with domain-specific knowledge
            - ✅ Transparent: you can see which chunks were used
            </div>
            """, unsafe_allow_html=True)

with main_tab2:
    # ==================== PHASE 5: RAG LAB ====================
    st.markdown('<div class="section-header">🧪 RAG Lab - Experiment with Parameters</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="edu-panel">
    <div class="edu-title">🔬 Interactive RAG Experimentation</div>
    
    Adjust RAG pipeline parameters and see how they affect document understanding.
    
    **Parameter Effects:**
    - **Chunk Size** ↓ = Higher precision, more chunks | ↑ = Richer context, fewer chunks
    - **Chunk Overlap** ↑ = Better semantic continuity | ↓ = Faster processing
    - **Top-K** ↑ = Broader coverage, more context | ↓ = Sharper focus, less noise
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.file_uploaded:
        st.info("📤 Upload a document first to start experimenting with RAG parameters!")
    else:
        st.markdown("### ⚙️ Experiment Configuration")
        
        lab_col1, lab_col2, lab_col3 = st.columns(3)
        
        with lab_col1:
            st.session_state.lab_chunk_size = st.slider(
                "Chunk Size (characters)",
                min_value=200,
                max_value=1500,
                value=st.session_state.lab_chunk_size,
                step=50,
                help="How many characters per chunk"
            )
        
        with lab_col2:
            st.session_state.lab_chunk_overlap = st.slider(
                "Chunk Overlap (characters)",
                min_value=0,
                max_value=300,
                value=st.session_state.lab_chunk_overlap,
                step=10,
                help="Overlap between consecutive chunks"
            )
        
        with lab_col3:
            st.session_state.lab_top_k = st.slider(
                "Top-K (results)",
                min_value=1,
                max_value=10,
                value=st.session_state.lab_top_k,
                help="Number of chunks to retrieve"
            )
        
        st.divider()
        
        # Experiment question
        st.markdown("### 📝 Enter Test Question")
        st.session_state.experiment_question = st.text_area(
            "Ask a question to test with different parameters:",
            placeholder="What is the main topic of this document?",
            key="lab_question"
        )
        
        col_run, col_reset = st.columns(2)
        
        with col_run:
            if st.button("🚀 Run Experiment", use_container_width=True):
                if not st.session_state.experiment_question:
                    st.warning("Please enter a question first")
                else:
                    with st.spinner("Running experiment with current settings..."):
                        try:
                            # Run with current parameters
                            result = st.session_state.pipeline.ask(
                                st.session_state.experiment_question,
                                top_k=st.session_state.lab_top_k
                            )
                            st.session_state.experiment_result_a = result
                            st.success("✅ Experiment complete!")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
        
        with col_reset:
            if st.button("🔄 Reset Experiment", use_container_width=True):
                st.session_state.experiment_result_a = None
                st.session_state.experiment_result_b = None
                st.rerun()
        
        # Display experiment results
        if st.session_state.experiment_result_a:
            st.divider()
            st.markdown("### 📊 Experiment Results")
            
            result = st.session_state.experiment_result_a
            
            # Show answer
            st.markdown("**Generated Answer:**")
            st.markdown(f"""
            <div class="success-box">
            {result['answer']}
            </div>
            """, unsafe_allow_html=True)
            
            # Show chunks used
            st.markdown("**Retrieved Chunks:**")
            for i, chunk in enumerate(result['retrieved_chunks'], 1):
                relevance = result['relevance_scores'][i-1]
                col_r1, col_r2 = st.columns([4, 1])
                with col_r1:
                    st.markdown(f"""
                    <div class="chunk-preview">
                    <b>Chunk {i}</b><br>
                    {chunk}
                    </div>
                    """, unsafe_allow_html=True)
                with col_r2:
                    st.metric("Relevance", f"{relevance:.0%}")
            
            # Hallucination check
            st.divider()
            st.markdown("### 🛡️ Hallucination Detection")
            
            max_relevance = max(result['relevance_scores']) if result['relevance_scores'] else 0
            
            if max_relevance < 0.4:
                st.warning(f"""
                ⚠️ **Low Relevance Alert** (max: {max_relevance:.0%})
                
                This question may not be well-answerable from your document.
                RAG prevented hallucination by using only available context.
                Consider: Is the answer based on the document or is the model inferring?
                """)
            else:
                st.success(f"""
                ✅ **High Confidence** (max relevance: {max_relevance:.0%})
                
                The retrieved chunks are relevant to your question.
                The answer is grounded in the document content.
                """)


# ==================== Footer ====================
st.markdown("""
<div class="footer">
<hr style="margin: 2em 0;">
<h4>📚 Learn RAG – Educational RAG Web Application</h4>
<p>
<b>Stack:</b> Streamlit + FAISS + Sentence Transformers + Llama 3.1 8B (Groq)<br>
<b>Built by:</b> Sandip Patil | 
<a href="https://github.com/sandip824/RAG-From-Scratch" target="_blank">GitHub Repository</a> | 
<a href="https://huggingface.co/spaces/sandip824/rag-learning-app" target="_blank">HuggingFace Space</a><br>
<b>Purpose:</b> Educational demonstration of Retrieval Augmented Generation<br>
<br>
✨ <i>Learn how RAG works by interacting with this app. Upload a PDF or TXT file, ask questions, and explore the entire pipeline!</i>
</p>
</div>
""", unsafe_allow_html=True)
