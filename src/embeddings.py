from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
# Updated import for Chroma
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List, Optional, Union
import os
from src.api_loader import LLMLoader

def chunk_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)

def create_vector_store(chunks: List[Document], use_azure: bool = True, persist_directory: str = None) -> Chroma:
    """Create a vector store from document chunks using ChromaDB."""
    # Get LLM and embeddings through our API loader
    llm_loader = LLMLoader(use_azure=use_azure)
    llm = llm_loader.get_langchain_llm()
    
    # Create embeddings model
    if use_azure:
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment="text-embedding-ada-002",
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY")
        )
    else:
        embeddings = OpenAIEmbeddings()
    
    # Create ChromaDB vector store
    if persist_directory:
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # In newer versions of langchain_chroma, persistence is automatic
        # when persist_directory is provided
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        # No need to call persist() as it's handled automatically
    else:
        # Create in-memory ChromaDB instance
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings
        )
        
    return vector_store

def save_vector_store(vector_store: Chroma, path: str = None) -> None:
    """
    Save the ChromaDB vector store to disk.
    Note: In newer versions of langchain_chroma, persistence is automatic
    when the vector store is created with a persist_directory.
    This function now just ensures the directory exists and prints info.
    """
    if path:
        os.makedirs(path, exist_ok=True)
        print(f"Vector store directory ensured at: {path}")
        # Note: In newer versions, we don't need to manually persist
    else:
        print("No persistence path specified for vector store")

def load_vector_store(path: str, use_azure: bool = True) -> Union[Chroma, None]:
    """Load the ChromaDB vector store from disk."""
    if not os.path.exists(path):
        print(f"Vector store path does not exist: {path}")
        return None
    
    # Get embeddings model
    if use_azure:
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment="text-embedding-ada-002",
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY")
        )
    else:
        embeddings = OpenAIEmbeddings()
    
    # Load ChromaDB from the persist directory
    return Chroma(
        persist_directory=path,
        embedding_function=embeddings
    )