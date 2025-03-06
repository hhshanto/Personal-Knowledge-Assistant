from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader
)
from typing import List
from langchain_core.documents import Document
import os

def load_document(file_path: str) -> List[Document]:
    """Load a document based on its file extension."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    if file_path.endswith('.pdf'):
        return PyPDFLoader(file_path).load()
    elif file_path.endswith('.txt'):
        return TextLoader(file_path).load()
    elif file_path.endswith('.docx'):
        return Docx2txtLoader(file_path).load()
    elif file_path.endswith('.md'):
        return UnstructuredMarkdownLoader(file_path).load()
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
        
def load_documents_from_directory(directory_path: str) -> List[Document]:
    """Load all supported documents from a directory."""
    if not os.path.isdir(directory_path):
        raise NotADirectoryError(f"Directory not found: {directory_path}")
        
    documents = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            try:
                file_docs = load_document(file_path)
                documents.extend(file_docs)
                print(f"Loaded {len(file_docs)} documents from {filename}")
            except (ValueError, FileNotFoundError) as e:
                print(f"Error loading {filename}: {str(e)}")
    
    return documents