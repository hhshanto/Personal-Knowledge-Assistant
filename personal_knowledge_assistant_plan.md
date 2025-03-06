# Personal Knowledge Assistant Project Plan

## Overview
This document outlines a step-by-step approach to building a Personal Knowledge Assistant using LangChain and LangGraph. The assistant will help you organize and retrieve information from your personal documents or notes.

## Prerequisites
- Python 3.9+
- Basic understanding of Python
- Familiarity with LLMs and AI concepts
- OpenAI API key or access to another LLM

## Project Phases

### Phase 1: Environment Setup & Document Processing

#### 1.1 Set up your development environment
- [ ] Create a new Python project
- [ ] Set up a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
- [ ] Install required packages
```bash
pip install langchain langchain-core langchain-openai langchain-community langchain-experimental langchain_text_splitters chromadb langsmith langserve langraph
pip install python-dotenv faiss-cpu tiktoken
```
- [ ] Create a `.env` file for your API keys
```
OPENAI_API_KEY=your_openai_key_here
```

#### 1.2 Implement document loading
- [ ] Create utility functions to load various document types
```python
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader
)

def load_document(file_path):
    """Load a document based on its file extension."""
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
```

#### 1.3 Implement text chunking
- [ ] Create a text splitter to chunk documents
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)
```

#### 1.4 Create embeddings and vector store
- [ ] Set up embeddings model
- [ ] Create a vector database for storage
```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def create_vector_store(chunks):
    """Create a vector store from document chunks."""
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store
```

#### 1.5 Integrate the components
- [ ] Create a document processing pipeline
- [ ] Test with sample documents

### Phase 2: Basic Query-Response System

#### 2.1 Set up a simple retriever
```python
def get_retriever(vector_store):
    """Create a retriever from the vector store."""
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
```

#### 2.2 Create a basic RAG chain
```python
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

def create_rag_chain(retriever):
    """Create a RAG chain for question answering."""
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    # Create a prompt template
    prompt = ChatPromptTemplate.from_template("""
    Answer the question based only on the following context:
    {context}
    
    Question: {input}
    """)
    
    # Create a chain to combine documents and answer questions
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create the RAG chain
    rag_chain = create_retrieval_chain(retriever, document_chain)
    
    return rag_chain
```

#### 2.3 Build a simple interface
- [ ] Create a basic CLI interface for querying
```python
def query_knowledge_base(rag_chain, query):
    """Query the knowledge base with a question."""
    response = rag_chain.invoke({"input": query})
    return response["answer"]
```

### Phase 3: Implementing LangGraph for Conversation Flow

#### 3.1 Define conversation states
- [ ] Plan the states of your conversation agent
  - Initial Query State
  - Document Retrieval State
  - Response Generation State
  - Clarification State
  - Follow-up Question State

#### 3.2 Create the basic graph structure
```python
import operator
from typing import TypedDict, Annotated, Sequence
from langchain_core.documents import Document

from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, AIMessage

class AgentState(TypedDict):
    messages: Sequence[HumanMessage | AIMessage]
    context: list[Document]
    current_query: str
    clarification_needed: bool

# Define the nodes
def query_analyzer(state: AgentState) -> AgentState:
    """Analyze the user query to determine if clarification is needed."""
    # Your implementation here
    return state

def retrieve_documents(state: AgentState) -> AgentState:
    """Retrieve relevant documents based on the query."""
    # Your implementation here
    return state

def generate_response(state: AgentState) -> AgentState:
    """Generate a response based on retrieved documents."""
    # Your implementation here
    return state

def request_clarification(state: AgentState) -> AgentState:
    """Request clarification from the user."""
    # Your implementation here
    return state

# Create the graph
def create_knowledge_assistant_graph(retriever, llm):
    """Create a LangGraph for the knowledge assistant."""
    
    # Initialize the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("query_analyzer", query_analyzer)
    workflow.add_node("retrieve_documents", retrieve_documents)
    workflow.add_node("generate_response", generate_response)
    workflow.add_node("request_clarification", request_clarification)
    
    # Add edges
    workflow.add_edge("query_analyzer", "request_clarification", 
                     condition=lambda state: state["clarification_needed"])
    workflow.add_edge("query_analyzer", "retrieve_documents", 
                     condition=lambda state: not state["clarification_needed"])
    workflow.add_edge("retrieve_documents", "generate_response")
    workflow.add_edge("request_clarification", "query_analyzer")
    
    # Set the entry point
    workflow.set_entry_point("query_analyzer")
    
    return workflow.compile()
```

#### 3.3 Implement node functions
- [ ] Develop each node function with proper logic
- [ ] Add error handling and edge cases
- [ ] Test the basic flow

#### 3.4 Add memory and conversation history
- [ ] Implement conversation history tracking
- [ ] Enable context-aware responses

### Phase 4: Enhancement and Refinement

#### 4.1 Add source tracking and citation
```python
def format_response_with_sources(response, documents):
    """Format the response with source citations."""
    # Implementation here to reference source documents
```

#### 4.2 Implement feedback loop
- [ ] Add user feedback collection
- [ ] Use feedback to improve future responses

#### 4.3 Add persistence
- [ ] Save vector store to disk
- [ ] Implement conversation history persistence
```python
def save_vector_store(vector_store, path):
    """Save the vector store to disk."""
    vector_store.save_local(path)

def load_vector_store(path):
    """Load the vector store from disk."""
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(path, embeddings)
```

#### 4.4 Create document management features
- [ ] Add/remove documents from knowledge base
- [ ] Update existing documents
- [ ] List available documents

## Project Structure
```
personal_knowledge_assistant/
├── .env                      # Environment variables
├── main.py                   # Entry point for the application
├── requirements.txt          # Dependencies
├── data/                     # Directory for storing documents
│   └── .gitkeep
├── vector_store/            # Directory for storing vector database
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── document_loader.py    # Document loading utilities
│   ├── embeddings.py         # Embedding and vector store creation
│   ├── retriever.py          # Retrieval functionality
│   ├── graph.py              # LangGraph implementation
│   └── utils.py              # General utilities
└── tests/                    # Test cases
    └── __init__.py
```

## Learning Resources

### LangChain Resources
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering/)
- [Vector Stores Guide](https://python.langchain.com/docs/modules/data_connection/vectorstores/)

### LangGraph Resources
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph Tutorials](https://github.com/langchain-ai/langgraph/blob/main/examples/tutorials/quickstart.ipynb)
- [State Management in LangGraph](https://langchain-ai.github.io/langgraph/concepts/state/)

### General References
- [Retrieval Augmented Generation (RAG) Patterns](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [Building AI Applications with LangChain (YouTube)](https://www.youtube.com/playlist?list=PLqZXAkvF1bPNQER9mLmDbntNfSpzdDIU5)

## Next Steps
After completing the basic implementation:

1. Enhance the system with metadata filtering
2. Implement advanced retrieval techniques (hybrid search, re-ranking)
3. Add a web interface (Streamlit, Gradio, or Flask)
4. Explore multi-modal capabilities (handling images, audio)
5. Implement advanced features like query routing and result verification


```mermaid
flowchart TD
    %% Main Entry Point
    main[main.py: main] --> check_vs{Vector Store Exists?}
    check_vs -->|No| setup[setup_knowledge_base]
    check_vs -->|Yes| validate[Validate Vector Store]
    validate -->|Empty| setup
    validate -->|Valid| init_rag[Initialize RAG]
    setup --> init_rag
    
    %% Setup Knowledge Base Flow
    setup --> load_docs[document_loader: load_documents_from_directory]
    load_docs --> chunk_docs[embeddings: chunk_documents]
    chunk_docs --> create_vs[embeddings: create_vector_store]
    create_vs --> save_vs[embeddings: save_vector_store]
    
    %% RAG Processor Initialization
    init_rag --> create_rag[rag_processor: RAGProcessor.__init__]
    create_rag --> load_llm[api_loader: LLMLoader]
    create_rag --> init_retriever[retriever: KnowledgeRetriever.__init__]
    init_retriever --> load_vs[embeddings: load_vector_store]
    
    %% Conversation Agent
    init_rag --> create_agent[conversation_graph: ConversationAgent.__init__]
    create_agent --> create_graph[conversation_graph: create_conversation_graph]
    create_graph --> workflow[StateGraph: workflow]
    workflow --> retrieve_node[retrieve_context function]
    workflow --> generate_node[generate_response function]
    workflow --> should_end_node[should_end function]
    
    %% Query Loop
    main --> query_loop[interactive_query_loop]
    query_loop --> process_msg[conversation_graph: process_message]
    
    %% Conversation Processing
    process_msg --> direct_rag{Try Direct RAG}
    direct_rag -->|Success| return_answer[Return Answer]
    direct_rag -->|Insufficient| invoke_graph[graph.invoke]
    
    %% Graph Execution
    invoke_graph --> retrieve_exec[Execute: retrieve_context]
    retrieve_exec --> get_context[rag_processor: answer_question]
    get_context --> retriever_get[retriever: get_relevant_context]
    retriever_get --> vs_search[vector_store: similarity_search]
    
    retrieve_exec --> generate_exec[Execute: generate_response]
    generate_exec --> extract_name[Extract Personal Info] 
    extract_name --> build_prompt[Build Prompt]
    build_prompt --> invoke_llm[llm: invoke]
    invoke_llm --> update_state[Update Conversation State]
    
    %% Final Response
    update_state --> return_response[Return Response to User]
    return_response --> query_loop
    
    %% Style
    classDef main fill:#f96,stroke:#333,stroke-width:2px
    classDef vector fill:#9cf,stroke:#333
    classDef llm fill:#f9f,stroke:#333
    classDef graph fill:#9f9,stroke:#333
    
    class main,setup,query_loop main
    class create_vs,save_vs,load_vs,vs_search vector
    class load_llm,invoke_llm llm
    class workflow,retrieve_node,generate_node,should_end_node graph
```