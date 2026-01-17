"""Vector store module - manages FAISS index for similarity search."""

import numpy as np
import faiss
from typing import List, Tuple


class VectorStore:
    """FAISS-based vector store for efficient similarity search."""
    
    def __init__(self):
        """Initialize an empty vector store."""
        self.index = None
        self.chunks = []
        self.embeddings = None
        print("✓ Vector store initialized")
    
    def build_index(self, embeddings: np.ndarray, chunks: List[str]) -> None:
        """
        Build a FAISS index from embeddings.
        
        Args:
            embeddings (np.ndarray): Shape (num_chunks, embedding_dim)
            chunks (List[str]): Original text chunks (for retrieval)
        
        Example:
            >>> embeddings = np.random.rand(10, 384)
            >>> chunks = ["text1", "text2", ..., "text10"]
            >>> store = VectorStore()
            >>> store.build_index(embeddings, chunks)
            >>> store.index is not None
            True
        """
        # Ensure embeddings are float32 (FAISS requirement)
        embeddings = embeddings.astype(np.float32)
        
        embedding_dim = embeddings.shape[1]
        
        # Create a simple flat index
        # Using IndexFlatL2 - computes L2 distance (Euclidean)
        # Alternative: IndexFlatIP for inner product similarity
        self.index = faiss.IndexFlatL2(embedding_dim)
        
        # Add embeddings to the index
        self.index.add(embeddings)
        
        # Store chunks and embeddings for later retrieval
        self.chunks = chunks
        self.embeddings = embeddings
        
        print(f"✓ Built FAISS index with {len(chunks)} vectors (dim={embedding_dim})")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> Tuple[List[str], List[float]]:
        """
        Search for top-k most similar chunks to a query.
        
        Args:
            query_embedding (np.ndarray): Query embedding (shape: (embedding_dim,))
            top_k (int): Number of results to retrieve (default: 5)
        
        Returns:
            Tuple[List[str], List[float]]: 
                - List of top-k chunks
                - List of corresponding distances (lower = more similar)
        
        Example:
            >>> query_emb = np.random.rand(384)
            >>> chunks, distances = store.search(query_emb, top_k=3)
            >>> len(chunks)
            3
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Reshape query to 2D (1, embedding_dim) as FAISS expects
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Convert to lists
        distances = distances[0].tolist()
        retrieved_chunks = [self.chunks[int(idx)] for idx in indices[0]]
        
        return retrieved_chunks, distances
    
    def get_index_size(self) -> int:
        """Get number of vectors in the index."""
        if self.index is None:
            return 0
        return self.index.ntotal
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        if self.index is None:
            return 0
        return self.index.d
