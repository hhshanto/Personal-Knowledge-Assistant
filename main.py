import os
import sys
from dotenv import load_dotenv
from src.document_loader import load_documents_from_directory
from src.embeddings import chunk_documents, create_vector_store, save_vector_store
from src.rag_processor import RAGProcessor
from src.conversation_graph import ConversationAgent 

def setup_knowledge_base():
    """Set up the knowledge base."""
    try:
        load_dotenv()
        print("Loading documents...")
        data_dir = os.getenv("DOCUMENTS_DIRECTORY") or os.path.join(os.path.dirname(__file__), "data")
        documents = load_documents_from_directory(data_dir)
        
        print(f"Loaded {len(documents)} document chunks")
        print("Document sources:")
        sources = set(doc.metadata.get('source', 'unknown') for doc in documents)
        for source in sources:
            print(f"- {source}")
        
        print("\nChunking documents...")
        chunks = chunk_documents(documents)
        print(f"Created {len(chunks)} chunks")
        
        # Determine if we should use Azure - prioritize environment variable
        use_azure = os.getenv("USE_AZURE", "true").lower() == "true" or bool(os.getenv("AZURE_OPENAI_API_KEY"))
        print(f"Using Azure: {use_azure}")
        
        print("\nCreating vector store...")
        vs_path = os.path.join(os.path.dirname(__file__), "vector_store", "index")
        
        # Create the directory if it doesn't exist
        os.makedirs(vs_path, exist_ok=True)
        
        vector_store = create_vector_store(chunks, use_azure=use_azure, persist_directory=vs_path)
        
        print(f"\nSaving vector store to {vs_path}...")
        save_vector_store(vector_store, vs_path)
        
        return True
    except Exception as e:
        print(f"Error setting up knowledge base: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def interactive_query_loop(conversation_agent):
    """Run an interactive query loop."""
    print("\nPersonal Knowledge Assistant is ready for your questions.")
    print("Type 'exit', 'quit', or 'q' to end the session.")
    
    while True:
        query = input("\nYour question: ")
        
        if query.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break
            
        try:
            response = conversation_agent.process_message(query)
            print("\nAnswer:")
            print(response)
            
        except Exception as e:
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    """Main entry point."""
    print("=== Personal Knowledge Assistant ===")
    
    # Check if knowledge base needs to be set up
    vs_path = os.path.join(os.path.dirname(__file__), "vector_store", "index")
    if not os.path.exists(vs_path) or "--rebuild" in sys.argv:
        print("Setting up knowledge base...")
        if not setup_knowledge_base():
            print("Failed to set up knowledge base. Exiting.")
            return
    else:
        print(f"Knowledge base exists at {vs_path}")
        # Verify if it has content
        try:
            from langchain_chroma import Chroma
            from src.api_loader import LLMLoader
            from src.embeddings import load_vector_store
            
            # Use the same Azure determination logic as in setup
            use_azure = os.getenv("USE_AZURE", "true").lower() == "true" or bool(os.getenv("AZURE_OPENAI_API_KEY"))
            
            print(f"Checking vector store content (using Azure: {use_azure})...")
            vs = load_vector_store(vs_path, use_azure=use_azure)
            if vs:
                try:
                    collection_stats = vs._collection.count()
                    print(f"Vector store has {collection_stats} documents")
                    if collection_stats == 0:
                        print("Vector store is empty. Rebuilding...")
                        if not setup_knowledge_base():
                            print("Failed to rebuild knowledge base. Exiting.")
                            return
                except Exception as e:
                    print(f"Couldn't verify vector store content: {str(e)}")
        except Exception as e:
            print(f"Error checking vector store: {str(e)}")
    
    # Initialize RAG processor
    try:
        # Use the same Azure determination logic as in setup
        use_azure = os.getenv("USE_AZURE", "true").lower() == "true" or bool(os.getenv("AZURE_OPENAI_API_KEY"))
        
        print(f"Initializing RAG processor (using Azure: {use_azure})...")
        rag_processor = RAGProcessor(vs_path, use_azure=use_azure)
        
        # Create conversation agent with LangGraph
        print("Initializing conversation agent...")
        conversation_agent = ConversationAgent(rag_processor)
        
        # Start interactive query loop
        interactive_query_loop(conversation_agent)
        
    except Exception as e:
        print(f"Error initializing: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
if __name__ == "__main__":
    main()