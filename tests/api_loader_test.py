from dotenv import load_dotenv
from src.api_loader import LLMLoader
import os

def test_direct_completion(loader, name, prompt="What are the three laws of robotics?"):
    """Test direct completion with the given loader"""
    print(f"\n--- Testing {name} Direct Completion ---")
    try:
        response = loader.get_completion(prompt, max_tokens=150)
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        return True
    except Exception as e:
        print(f"Error with {name}: {str(e)}")
        return False

def test_langchain_llm(loader, name):
    """Test LangChain LLM creation with the given loader"""
    print(f"\n--- Testing {name} LangChain Integration ---")
    try:
        llm = loader.get_langchain_llm()
        response = llm.invoke("Explain what a vector database is in one paragraph.")
        print(f"LangChain Response: {response.content}")
        return True
    except Exception as e:
        print(f"Error with {name} LangChain: {str(e)}")
        return False

def main():
    """Main function to test the LLM loader"""
    print("=== LLM Loader Test ===")
    load_dotenv()
    
    # Test Azure OpenAI
    azure_success = False
    if os.getenv("AZURE_OPENAI_API_KEY"):
        try:
            azure_loader = LLMLoader(use_azure=True)
            azure_direct_success = test_direct_completion(azure_loader, "Azure OpenAI")
            azure_langchain_success = test_langchain_llm(azure_loader, "Azure OpenAI")
            azure_success = azure_direct_success and azure_langchain_success
        except Exception as e:
            print(f"Error initializing Azure OpenAI loader: {str(e)}")
    else:
        print("\nSkipping Azure OpenAI tests (API key not found in .env)")
    

    
    # Print summary
    print("\n=== Test Summary ===")
    if azure_success:
        print("✅ Azure OpenAI: All tests passed")
    elif os.getenv("AZURE_OPENAI_API_KEY"):
        print("❌ Azure OpenAI: Tests failed")
    else:
        print("⚠️ Azure OpenAI: Not tested")
        
        
    # Next steps guidance
    print("\n=== Next Steps ===")
    if azure_success:
        print("LLM loader is working! You can now:")
        print("1. Begin implementing document loading and embedding")
        print("2. Set up your RAG chain using the LangChain LLM")
        print("3. Start building the conversation flow with LangGraph")
    else:
        print("Please check your API keys and try again.")
        print("Make sure your .env file contains the necessary credentials.")

if __name__ == "__main__":
    main()