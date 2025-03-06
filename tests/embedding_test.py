import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.document_loader import load_documents_from_directory
from src.embeddings import chunk_documents, create_vector_store, save_vector_store, load_vector_store
from dotenv import load_dotenv

def test_embeddings_pipeline():
    """Test the document embedding pipeline"""
    print("=== Embeddings Pipeline Test ===")
    load_dotenv()
    
    # Define paths - update to use vector_store instead of chroma
    project_root = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(project_root, "data")
    vector_store_dir = os.path.join(project_root, "vector_store") 
    
    # Step 1: Load documents
    print(f"Loading documents from {data_dir}...")
    try:
        documents = load_documents_from_directory(data_dir)
        if not documents:
            print("No documents found or loaded. Please add documents to the data directory.")
            return
        print(f"Loaded {len(documents)} document chunks.")
        
        # Step 2: Chunk documents
        print("Splitting documents into chunks...")
        chunks = chunk_documents(documents)
        print(f"Created {len(chunks)} chunks.")
        
        # Step 3: Create vector store with the vector_store directory
        print("Creating vector store...")
        try:
            # Pass the vector_store_dir parameter to create_vector_store
            vector_store = create_vector_store(chunks, use_azure=True, persist_directory=vector_store_dir)
            print(f"Vector store created successfully at {vector_store_dir}.")
            
            # Step 5: Test loading vector store
            print("Testing vector store loading...")
            loaded_vs = load_vector_store(vector_store_dir, use_azure=True)  # Use the same path for loading
            if loaded_vs:
                print("Vector store loaded successfully.")
                
                # Perform a simple similarity search to verify it works
                print("\nTesting similarity search with a sample query...")
                query = "sample question about the documents"
                results = loaded_vs.similarity_search(query, k=2)
                print(f"Found {len(results)} results for query: '{query}'")
                if results:
                    print("\nFirst result excerpt:")
                    print(f"{results[0].page_content[:150]}...")
            else:
                print("Failed to load vector store.")
                
        except Exception as e:
            print(f"Error in vector store creation or loading: {str(e)}")
            
    except Exception as e:
        print(f"Error loading documents: {str(e)}")

if __name__ == "__main__":
    test_embeddings_pipeline()