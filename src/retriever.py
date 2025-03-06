from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from typing import List
import os
from src.embeddings import load_vector_store

class KnowledgeRetriever:
    """Retriever for the Personal Knowledge Assistant."""
    
    def __init__(self, vector_store_path: str, use_azure: bool = True):
        """
        Initialize the knowledge retriever.
        """
        print(f"Initializing retriever with vector store path: {vector_store_path}")
        
        # Ensure the path is absolute and exists
        self.vector_store_path = os.path.abspath(vector_store_path)
        if not os.path.exists(self.vector_store_path):
            print(f"Warning: Vector store path does not exist: {self.vector_store_path}")
            # Try creating the directory if it doesn't exist
            os.makedirs(self.vector_store_path, exist_ok=True)
        
        self.use_azure = use_azure
        
        # Load vector store with detailed error handling
        try:
            self.vector_store = load_vector_store(self.vector_store_path, use_azure)
            
            if self.vector_store is None:
                raise ValueError(f"Failed to load vector store from {self.vector_store_path}")
                
            # Verify the vector store has content
            try:
                collection_stats = self.vector_store._collection.count()
                print(f"Vector store loaded with {collection_stats} documents")
            except Exception as e:
                print(f"Warning: Unable to get vector store stats: {str(e)}")
                
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            import traceback
            traceback.print_exc()
            raise ValueError(f"Failed to initialize vector store: {str(e)}")
        
        print(f"Vector store type: {type(self.vector_store).__name__}")
            
    def retrieve(self, query: str, top_k: int = 3) -> List[Document]:
        """
        Retrieve relevant documents for the given query.
        
        Args:
            query: The user's question
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        try:
            results = self.vector_store.similarity_search(query, k=top_k)
            return results
        except Exception as e:
            print(f"Error during retrieval: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
        
    def get_relevant_context(self, query: str, top_k: int = 5) -> str:
        """Get relevant context as a string."""
        # Adding debug print statements
        print(f"Retrieving context for query: '{query}' (top_k={top_k})")
        
        # Get documents with a higher top_k value
        docs = self.retrieve(query, top_k)
        print(f"Retrieved {len(docs)} documents")
        
        if docs:
            # Print the first result source
            print(f"First result from: {docs[0].metadata.get('source', 'unknown')}")
            
        context = "\n\n".join([doc.page_content for doc in docs])
        return context