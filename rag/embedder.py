"""Embedding module - converts text chunks into dense vector representations."""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
from tqdm import tqdm


class EmbeddingModel:
    """Wrapper for sentence-transformers embedding model."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.
        
        Args:
            model_name (str): HuggingFace model name (default: all-MiniLM-L6-v2)
                - all-MiniLM-L6-v2: Fast, lightweight, good for semantic similarity
                - all-mpnet-base-v2: Larger, more accurate but slower
        """
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        print(f"✓ Loaded embedding model: {model_name}")
    
    def embed_chunks(self, chunks: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed a list of text chunks into vectors.
        
        Args:
            chunks (List[str]): List of text chunks to embed
            batch_size (int): Batch size for encoding (default: 32)
        
        Returns:
            np.ndarray: Shape (len(chunks), embedding_dim) - 384 for all-MiniLM-L6-v2
        
        Example:
            >>> model = EmbeddingModel()
            >>> chunks = ["Hello world", "This is a test"]
            >>> embeddings = model.embed_chunks(chunks)
            >>> embeddings.shape
            (2, 384)
        """
        print(f"Embedding {len(chunks)} chunks...")
        
        embeddings = self.model.encode(
            chunks,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.
        
        Args:
            query (str): Query text
        
        Returns:
            np.ndarray: Shape (embedding_dim,) - 384 for all-MiniLM-L6-v2
        """
        embedding = self.model.encode(query, convert_to_numpy=True)
        return embedding
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        return self.model.get_sentence_embedding_dimension()
