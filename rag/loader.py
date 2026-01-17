"""Document loader module - handles PDF and TXT file loading."""

from pypdf import PdfReader
from typing import Union


def load_pdf(file) -> str:
    """
    Load and extract text from a PDF file.
    
    Args:
        file: A file-like object (typically from Streamlit file uploader)
    
    Returns:
        str: Extracted text from the PDF
    """
    pdf_reader = PdfReader(file)
    text = ""
    
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text() + "\n"
    
    return text.strip()


def load_txt(file) -> str:
    """
    Load text from a TXT file.
    
    Args:
        file: A file-like object (typically from Streamlit file uploader)
    
    Returns:
        str: Text content from the file
    """
    # If bytes, decode to string
    if isinstance(file, bytes):
        return file.decode('utf-8')
    
    # If file object with read method
    content = file.read()
    if isinstance(content, bytes):
        return content.decode('utf-8')
    
    return content


def load_file(file) -> str:
    """
    Auto-detect and load either PDF or TXT file.
    
    Args:
        file: A file-like object
    
    Returns:
        str: Extracted text content
    """
    file_name = file.name.lower()
    
    if file_name.endswith('.pdf'):
        return load_pdf(file)
    elif file_name.endswith('.txt'):
        return load_txt(file)
    else:
        raise ValueError(f"Unsupported file type: {file_name}. Supported: PDF, TXT")
