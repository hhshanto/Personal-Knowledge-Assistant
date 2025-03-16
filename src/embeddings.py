from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
# Updated import for Chroma
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List, Optional, Union, Dict, Any
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
def get_embedding_function(use_azure: bool = True):
    """Get the appropriate embedding function based on configuration."""
    if use_azure:
        return AzureOpenAIEmbeddings(
            azure_deployment="text-embedding-ada-002",
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY")
        )
    else:
        return OpenAIEmbeddings()

# Then update the add_documents_to_vector_store function:
def add_documents_to_vector_store(documents, vector_store_path, use_azure=True):
    """Add new documents to an existing vector store."""
    
    # Add source filename to metadata if not present
    for doc in documents:
        if not doc.metadata.get('source'):
            doc.metadata['source'] = 'unknown'
        
        # Extract filename from source path if it exists
        source = doc.metadata.get('source', '')
        if isinstance(source, str) and os.path.exists(source):
            doc.metadata['filename'] = os.path.basename(source)
            doc.metadata['filetype'] = os.path.splitext(source)[1].lower()[1:]  # Remove the dot
            
            # Add creation and modified times
            try:
                doc.metadata['created_at'] = os.path.getctime(source)
                doc.metadata['modified_at'] = os.path.getmtime(source)
            except:
                pass
    
    # Load the existing vector store
    vector_store = load_vector_store(vector_store_path, use_azure)
    
    if vector_store is None:
        # Create a new one if it doesn't exist
        vector_store = create_vector_store(documents, use_azure, vector_store_path)
    else:
        # Add documents to the existing store
        embedding_function = get_embedding_function(use_azure)
        vector_store.add_documents(documents)
    
    return vector_store

def get_all_documents_metadata(path: str, use_azure: bool = True) -> List[dict]:
    """Get metadata for all documents in the vector store."""
    vector_store = load_vector_store(path, use_azure)
    if not vector_store:
        return []
    
    # Get all documents from ChromaDB
    results = vector_store._collection.get()
    
    # Extract metadata and IDs
    documents = []
    for i, (doc_id, metadata, content) in enumerate(zip(
        results['ids'], 
        results['metadatas'], 
        results['documents']
    )):
        # Add ID to metadata for reference
        metadata['id'] = doc_id
        metadata['content_preview'] = content[:100] + "..." if len(content) > 100 else content
        documents.append(metadata)
    
    return documents

def delete_documents_by_ids(path: str, doc_ids: List[str], use_azure: bool = True) -> bool:
    """Delete specific documents from the vector store by their IDs."""
    vector_store = load_vector_store(path, use_azure)
    if not vector_store:
        return False
    
    try:
        # Delete documents by IDs
        vector_store._collection.delete(doc_ids)
        # No need to explicitly call persist in newer versions of langchain_chroma
        # The deletion is automatically persisted
        return True
    except Exception as e:
        print(f"Error deleting documents: {str(e)}")
        return False

def delete_documents_by_source(path: str, source_path: str, use_azure: bool = True) -> tuple:
    """Delete all documents from a specific source file."""
    vector_store = load_vector_store(path, use_azure)
    if not vector_store:
        return False, 0
    
    try:
        # Query for documents with the matching source
        results = vector_store._collection.get(
            where={"source": source_path}
        )
        
        if results and results['ids']:
            # Delete documents by IDs
            vector_store._collection.delete(results['ids'])
            # No need to explicitly call persist in newer versions of langchain_chroma
            # The deletion is automatically persisted
            return True, len(results['ids'])
        return False, 0
    except Exception as e:
        print(f"Error deleting documents by source: {str(e)}")
        return False, 0

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