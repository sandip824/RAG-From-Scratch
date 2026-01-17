"""Text chunking module - splits documents into overlapping chunks."""

from typing import List


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks for embedding.
    
    Args:
        text (str): The text to chunk
        chunk_size (int): Size of each chunk in characters (default: 500)
        overlap (int): Overlap between consecutive chunks in characters (default: 50)
    
    Returns:
        List[str]: List of text chunks
    
    Example:
        >>> text = "This is a long document. " * 100
        >>> chunks = chunk_text(text, chunk_size=100, overlap=20)
        >>> len(chunks) > 1
        True
    """
    chunks = []
    start = 0
    
    while start < len(text):
        # Calculate end position
        end = start + chunk_size
        
        # Extract chunk
        chunk = text[start:end].strip()
        
        # Only add non-empty chunks
        if chunk:
            chunks.append(chunk)
        
        # Move start position by (chunk_size - overlap)
        start += chunk_size - overlap
    
    return chunks


def chunk_by_sentences(text: str, sentences_per_chunk: int = 5) -> List[str]:
    """
    Alternative chunking strategy: split by sentences.
    
    Args:
        text (str): The text to chunk
        sentences_per_chunk (int): Number of sentences per chunk (default: 5)
    
    Returns:
        List[str]: List of text chunks
    """
    # Simple sentence splitting (could be enhanced with NLTK or spaCy)
    sentences = text.replace('!', '.').replace('?', '.').split('.')
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = '. '.join(sentences[i:i + sentences_per_chunk])
        if chunk.strip():
            chunks.append(chunk.strip() + '.')
    
    return chunks
