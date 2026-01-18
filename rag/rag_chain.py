"""RAG chain module - orchestrates the complete RAG pipeline."""

import os
from typing import Dict, Any, List
from groq import Groq

from rag.loader import load_file
from rag.chunker import chunk_text
from rag.embedder import EmbeddingModel
from rag.vector_store import VectorStore


class RAGPipeline:
    """Complete RAG pipeline orchestration."""
    
    def __init__(self, groq_api_key: str = None):
        """
        Initialize the RAG pipeline.
        
        Args:
            groq_api_key (str): Groq API key. If None, uses GROQ_API_KEY env var.
        """
        if groq_api_key is None:
            groq_api_key = os.getenv("GROQ_API_KEY")
        
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not provided and not in environment variables")
        
        self.client = Groq(api_key=groq_api_key)
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore()
        
        # Pipeline state
        self.chunks = []
        self.raw_text = ""
        self.is_ready = False
        
        print("✓ RAG Pipeline initialized")
    
    def build_pipeline(self, file) -> Dict[str, Any]:
        """
        Build the RAG pipeline from a file (PDF or TXT).
        
        Args:
            file: File object from Streamlit uploader
        
        Returns:
            Dict with pipeline metadata
        
        Example:
            >>> pipeline = RAGPipeline()
            >>> metadata = pipeline.build_pipeline(uploaded_file)
            >>> metadata['num_chunks']
            25
        """
        print(f"\n{'='*50}")
        print(f"Building RAG Pipeline")
        print(f"{'='*50}\n")
        
        # Step 1: Load file
        print("[1/4] Loading document...")
        self.raw_text = load_file(file)
        print(f"✓ Loaded {len(self.raw_text)} characters")
        
        # Step 2: Chunk text
        print("\n[2/4] Chunking text...")
        self.chunks = chunk_text(self.raw_text, chunk_size=500, overlap=50)
        print(f"✓ Created {len(self.chunks)} chunks")
        
        # Step 3: Embed chunks
        print("\n[3/4] Generating embeddings...")
        embeddings = self.embedding_model.embed_chunks(self.chunks)
        
        # Step 4: Build index
        print("\n[4/4] Building vector index...")
        self.vector_store.build_index(embeddings, self.chunks)
        
        self.is_ready = True
        
        metadata = {
            "filename": file.name,
            "file_size_bytes": len(self.raw_text),
            "num_chunks": len(self.chunks),
            "embedding_model": self.embedding_model.model_name,
            "embedding_dim": self.embedding_model.get_embedding_dim(),
            "index_size": self.vector_store.get_index_size(),
        }
        
        print(f"\n{'='*50}")
        print(f"✓ Pipeline Ready!")
        print(f"{'='*50}\n")
        
        return metadata
    
    def ask(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Ask a question and get an answer using RAG.
        
        Args:
            question (str): User's question
            top_k (int): Number of chunks to retrieve (default: 5)
        
        Returns:
            Dict with:
                - answer: LLM response
                - retrieved_chunks: List of relevant chunks
                - distances: Similarity distances
                - prompt: Final prompt sent to LLM
                - relevance_scores: List of relevance scores (1 / (1 + distance))
                - no_chunks_found: Boolean indicating if retrieval found chunks
                - query_embedding: The embedding vector of the question
                - embedding_model: Name of embedding model
        
        Raises:
            ValueError: If pipeline not built
        """
        if not self.is_ready:
            raise ValueError("Pipeline not ready. Call build_pipeline() first.")
        
        # Step 1: Embed query
        query_embedding = self.embedding_model.embed_query(question)
        
        # Step 2: Retrieve relevant chunks
        retrieved_chunks, distances = self.vector_store.search(query_embedding, top_k=top_k)
        
        # Calculate relevance scores (inverse of distance)
        relevance_scores = [1.0 / (1.0 + dist) for dist in distances]
        
        # Check if we found chunks
        no_chunks_found = len(retrieved_chunks) == 0 or (len(retrieved_chunks) == 1 and retrieved_chunks[0].strip() == "")
        
        # Step 3: Build context
        context = "\n\n".join([f"[{i+1}] {chunk}" for i, chunk in enumerate(retrieved_chunks)])
        
        # Step 4: Create strict prompt with anti-hallucination instructions
        if no_chunks_found:
            prompt = f"""You are a helpful AI assistant. The user has asked a question, but no relevant context from their document could be found.

Question: {question}

Respond by saying the answer is NOT found in the provided document."""
        else:
            prompt = f"""You are a helpful AI assistant answering questions based on provided context.

IMPORTANT RULES:
1. Answer ONLY using the context provided below
2. If the answer is not found in the context, explicitly say: "This information is not present in the document."
3. Do not use any external knowledge or assumptions
4. If the question is unclear or partially answered by the context, be honest about the limitations

Context from document:
{context}

Question: {question}

Answer:"""
        
        # Step 5: Call LLM
        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful tutor who answers questions using ONLY the provided context. If information is not in the context, say so clearly."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500,
        )
        
        answer = response.choices[0].message.content
        
        return {
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
            "distances": distances,
            "relevance_scores": relevance_scores,
            "prompt": prompt,
            "no_chunks_found": no_chunks_found,
            "query_embedding": query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding,
            "embedding_model": self.embedding_model.model_name,
            "embedding_dim": self.embedding_model.get_embedding_dim(),
        }


def build_rag_pipeline(file, groq_api_key: str = None) -> RAGPipeline:
    """
    Convenience function to build a complete RAG pipeline.
    
    Args:
        file: File object (PDF or TXT)
        groq_api_key (str): Groq API key
    
    Returns:
        RAGPipeline: Initialized and ready pipeline
    """
    pipeline = RAGPipeline(groq_api_key=groq_api_key)
    pipeline.build_pipeline(file)
    return pipeline
