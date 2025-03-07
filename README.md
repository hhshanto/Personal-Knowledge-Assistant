# Personal Knowledge Assistant

A smart, conversational RAG (Retrieval Augmented Generation) system that answers questions based on your personal knowledge base.

## 📚 Overview

Personal Knowledge Assistant lets you create a personalized AI assistant that answers questions using your own documents and knowledge. It uses advanced embedding techniques to retrieve relevant information and LLM technology to generate natural, conversational responses.

## ✨ Features

- **Document-based knowledge**: Answers questions based on your own documents
- **Conversation memory**: Remembers personal information shared during conversations
- **Fallback strategies**: Multiple retrieval approaches to maximize answer quality
- **LangGraph architecture**: Modular conversation flow with clear state management
- **Warm, conversational tone**: Responds in a friendly manner even with technical content
- **Streamlit Interface**: User-friendly web interface for interaction
- **Document Upload**: Upload new documents directly through the interface
- **Dynamic Vector Store Update**: Add new documents to the vector store without restarting the application

## 🔧 Technologies

- **LangChain**: For vector retrieval and document processing
- **LangGraph**: For conversational state management
- **ChromaDB**: For vector storage
- **Azure OpenAI**: For embeddings and response generation
- **Streamlit**: For building the web interface

## 🚀 Installation

1. **Clone the repository**

```bash
git clone https://github.com/hhshanto/Personal-Knowledge-Assistant.git
cd Personal-Knowledge-Assistant
```

2. **Set up a virtual environment**

```bash
conda create -n rag python=3.10
conda activate rag
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure environment variables**

Create a .env file in the root directory with the following variables:

```
# Azure OpenAI
AZURE_OPENAI_API_KEY=your_azure_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_DEPLOYMENT_NAME=your_deployment_name
AZURE_EMBEDDINGS_DEPLOYMENT_NAME=your_embeddings_deployment

# Documents
DOCUMENTS_DIRECTORY=./data
```

## 📋 Usage

1. **Add documents to your knowledge base**

Place your documents in the data directory or configure an alternative location in the .env file.

2. **Run the application**

```bash
# For first-time setup or to rebuild the vector store
python main.py --rebuild

# For normal operation
python main.py
```

3. **Ask questions**

Once the application is running, you can ask questions about the content in your documents.

```
Personal Knowledge Assistant is ready for your questions.
Type 'exit', 'quit', or 'q' to end the session.

Your question: tell me about regex
```

4. **Use the Streamlit Interface**

Run the Streamlit application to interact with the assistant through a web interface:

```bash
streamlit run app.py
```

## 🗂️ Project Structure

```
Personal-Knowledge-Assistant/
├── data/                     # Your documents go here
├── src/
│   ├── api_loader.py         # LLM API integration
│   ├── conversation_graph.py # LangGraph conversation flow
│   ├── document_loader.py    # Document processing
│   ├── embeddings.py         # Vector embedding functionality
│   ├── rag_processor.py      # Core RAG implementation
│   └── retriever.py          # Knowledge retrieval from vector store
├── vector_store/             # Generated vector embeddings
├── main.py                   # Main application entry point
├── app.py                    # Streamlit application entry point
├── .env                      # Environment variables (create this)
└── requirements.txt          # Project dependencies
```

## ⚙️ Configuration

The application can be configured through the .env file:

- **USE_AZURE**: Set to "true" to use Azure OpenAI or "false" to use regular OpenAI
- **DOCUMENTS_DIRECTORY**: Path to your document repository
- **Vector store**: Located in index by default

## 🧠 LangGraph Implementation

The conversation flow is managed using LangGraph with a simple two-node architecture:

1. **retrieve_context**: Gets relevant knowledge from the vector store
2. **generate_response**: Creates a natural language response using the retrieved context

Flow visualization:
```
Entry point: retrieve → generate → [conditional]
  - If completed: END
  - Otherwise: retrieve
```

## 📝 License

MIT License

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

Built with ❤️ using LangChain and LangGraph