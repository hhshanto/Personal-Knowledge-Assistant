from src.retriever import KnowledgeRetriever
from src.api_loader import LLMLoader
from typing import Dict, Any

class RAGProcessor:
    """RAG processor for the Personal Knowledge Assistant."""
    
    def __init__(self, vector_store_path: str, use_azure: bool = True):
        """
        Initialize the RAG processor.
        
        Args:
            vector_store_path: Path to the vector store
            use_azure: Whether to use Azure OpenAI (True) or regular OpenAI (False)
        """
        self.retriever = KnowledgeRetriever(vector_store_path, use_azure)
        self.llm_loader = LLMLoader(use_azure=use_azure)
        self.llm = self.llm_loader.get_langchain_llm()
        
    def answer_question(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Answer a question using RAG.
        
        Args:
            question: The user's question
            top_k: Number of documents to retrieve
            
        Returns:
            Dictionary containing the answer and metadata
        """
        context = self.retriever.get_relevant_context(question, top_k)
        
        prompt = f"""You are a helpful assistant that answers questions based on the provided context.
        
Context:
{context}

Question: {question}

Please answer the question based only on the provided context. If the context doesn't contain the information needed to answer the question, say "I don't have enough information to answer that question." and suggest what information might be needed.

Answer:"""
        
        response = self.llm.invoke(prompt)
        
        return {
            "question": question,
            "answer": response.content,
            "context": context
        }