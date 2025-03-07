import sqlite_patch
import streamlit as st
from dotenv import load_dotenv
from src.rag_processor import RAGProcessor
from src.conversation_graph import ConversationAgent
import os
import time
import tempfile
from src.document_loader import load_documents_from_files
from src.embeddings import chunk_documents, add_documents_to_vector_store

st.set_page_config(
    page_title="Personal Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Main app background */
    .main {
        background-color: #f0f2f6;
    }
    
    /* App container */
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Base chat message styling */
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
    }
    
    /* User message - blue theme - slightly adjusted */
    .chat-message.user {
        background-color: #d4e6ff; /* Slightly deeper blue */
        border-left: 5px solid #1565c0;
    }

    /* Assistant message - green theme - slightly adjusted */
    .chat-message.assistant {
        background-color: #e0f2e9; /* Slightly deeper green */
        border-left: 5px solid #2e7d32;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Avatar styling */
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin-right: 1rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        color: white;
    }
    
    /* User avatar color */
    .chat-message.user .avatar {
        background-color: #1976d2;
    }
    
    /* Assistant avatar color */
    .chat-message.assistant .avatar {
        background-color: #4caf50;
    }
    
    /* Message content */
    .chat-message .content {
        flex-grow: 1;
        color: #212121; /* Very dark gray, almost black */
    }
    
    /* Input field styling */
    .stTextInput > div > div > input {
        border-radius: 20px;
        padding-left: 20px;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 20px;
        padding: 0.5rem 1rem;
    }
    
    /* Headings */
    h1, h3 {
        color: #1e3799;
    }
    
    /* Sidebar adjustments */
    [data-testid="stSidebar"] {
        padding-left: 0 !important;
        margin-left: 0 !important;
        width: 250px; /* Adjust the width as needed */
    }
</style>
""", unsafe_allow_html=True)



# Load environment variables from both .env and Streamlit secrets
load_dotenv()

# Set Azure OpenAI credentials from Streamlit secrets
if 'AZURE_OPENAI_API_KEY' in st.secrets:
    os.environ['AZURE_OPENAI_API_KEY'] = st.secrets['AZURE_OPENAI_API_KEY']
if 'AZURE_OPENAI_ENDPOINT' in st.secrets:
    os.environ['AZURE_OPENAI_ENDPOINT'] = st.secrets['AZURE_OPENAI_ENDPOINT']
if 'AZURE_DEPLOYMENT_NAME' in st.secrets:
    os.environ['AZURE_DEPLOYMENT_NAME'] = st.secrets['AZURE_DEPLOYMENT_NAME']
if 'AZURE_EMBEDDINGS_DEPLOYMENT_NAME' in st.secrets:
    os.environ['AZURE_EMBEDDINGS_DEPLOYMENT_NAME'] = st.secrets['AZURE_EMBEDDINGS_DEPLOYMENT_NAME']

vs_path = os.path.join(os.path.dirname(__file__), "vector_store", "index")
use_azure = os.getenv("USE_AZURE", "true").lower() == "true"

# Initialize RAG processor and conversation agent
@st.cache_resource
def get_conversation_agent():
    with st.spinner("Initializing knowledge assistant..."):
        rag_processor = RAGProcessor(vs_path, use_azure=use_azure)
        return ConversationAgent(rag_processor)

# Sidebar with app information and document upload
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/brain--v2.png", width=80)
    st.title("Personal Knowledge Assistant")
    
    # About section
    with st.expander("About", expanded=True):
        st.markdown("""
        This assistant helps you access and interact with your personal knowledge base.
        
        **Features:**
        - Answer questions about your documents
        - Remember personal information
        - Natural conversational interface
        
        Built with LangChain and LangGraph
        """)
    
    # Add a clear conversation button
    if st.button("Clear Conversation", type="primary"):
        st.session_state.messages = []
        st.session_state.thinking = False
        st.rerun()
    
    # Add a divider for visual separation
    st.divider()
    
    # Document upload section
    st.markdown("### 📄 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Add files to your knowledge base", 
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        file_details = ""
        for file in uploaded_files:
            file_size_mb = round(file.size / (1024 * 1024), 2)
            file_details += f"- {file.name} ({file_size_mb} MB)\n"
        
        st.markdown(f"**Selected files:**\n{file_details}")
        
        col1, col2 = st.columns(2)
        with col1:
            process_button = st.button("Process Documents", type="primary", key="process_docs")
        with col2:
            cancel_button = st.button("Cancel", key="cancel_upload")
            
        if cancel_button:
            st.session_state.uploaded_files = None
            st.rerun()
            
        if process_button:
            with st.spinner("Processing documents..."):
                # Create temporary files
                temp_files = []
                for uploaded_file in uploaded_files:
                    # Create a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        # Write the uploaded file content to the temp file
                        tmp_file.write(uploaded_file.getvalue())
                        temp_files.append(tmp_file.name)
                
                try:
                    # Load documents
                    documents = load_documents_from_files(temp_files)
                    
                    if documents:
                        # Get document count
                        doc_count = len(documents)
                        
                        # Chunk documents
                        chunks = chunk_documents(documents)
                        chunk_count = len(chunks)
                        
                        # Add to vector store
                        add_documents_to_vector_store(
                            chunks, 
                            vs_path, 
                            use_azure=use_azure
                        )
                        
                        st.success(f"✅ Successfully processed {doc_count} documents into {chunk_count} chunks!")
                        
                        # Add a refresh button
                        if st.button("Refresh Knowledge Base"):
                            # Clear the cached agent to force reloading with new documents
                            st.cache_resource.clear()
                            st.rerun()
                    else:
                        st.error("No documents were successfully loaded. Please check file formats.")
                
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")
                
                finally:
                    # Clean up temporary files
                    import os
                    for temp_file in temp_files:
                        try:
                            os.unlink(temp_file)
                        except:
                            pass
    
    # Add another divider before settings
    st.divider()
    
    # Optional settings expandable section
    with st.expander("Settings"):
        st.slider("Response Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
        st.checkbox("Show sources", value=True)
    

# Main content area
st.header("🧠 Personal Knowledge Assistant")

# Initialize agent
agent = get_conversation_agent()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "thinking" not in st.session_state:
    st.session_state.thinking = False

def display_message(role, content):
    # Use better emoji icons
    avatar = "👨‍💼" if role == "user" else "🤖"
    
    st.markdown(f"""
    <div class="chat-message {role}">
        <div class="avatar">{avatar}</div>
        <div class="content">{content}</div>
    </div>
    """, unsafe_allow_html=True)

# Display chat messages from history
if not st.session_state.messages:
    st.info("👋 Hello! I'm your Personal Knowledge Assistant. Ask me anything about your documents!")

for message in st.session_state.messages:
    display_message(message["role"], message["content"])

# Replace your input area section with this fixed version:
input_container = st.container()

# Store the form key in session state to ensure consistency
if "form_key" not in st.session_state:
    st.session_state.form_key = "chat_form_1"

# Use a flag to control form showing
if "processing" not in st.session_state:
    st.session_state.processing = False

# Handle form display and submission
if not st.session_state.processing:
    with st.form(key=st.session_state.form_key, clear_on_submit=True):
        user_input = st.text_input("Type your question here...", key="user_input", label_visibility="collapsed")
        submit_button = st.form_submit_button("Send", type="primary")
        
        if submit_button and user_input:
            # Set flag to prevent duplicate forms during processing
            st.session_state.processing = True
            
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Store the question to process after rerun
            st.session_state.current_question = user_input
            
            # Rerun once to update the UI with user message
            st.rerun()
else:
    # Process the question and show thinking animation
    with st.spinner("Thinking..."):
        try:
            response = agent.process_message(st.session_state.current_question)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "I encountered an error processing your request. Please try again."
            })
    
    # Reset processing state and update form key for next input
    st.session_state.processing = False
    st.session_state.form_key = f"chat_form_{len(st.session_state.messages)}"
    
    # Rerun to show the form again
    st.rerun()