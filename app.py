import sqlite_patch
# app.py
import streamlit as st
from dotenv import load_dotenv
from src.rag_processor import RAGProcessor
from src.conversation_graph import ConversationAgent
import os
import time

st.set_page_config(
    page_title="Personal Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
# Custom CSS for better styling with improved contrast
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
    
    /* User message - blue theme */
    .chat-message.user {
        background-color: #e3f2fd; /* Light blue background */
        border-left: 5px solid #1976d2;
    }
    
    /* Assistant message - green theme */
    .chat-message.assistant {
        background-color: #e8f5e9; /* Light green background */
        border-left: 5px solid #4caf50;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
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
    
    /* Sidebar padding */
    .sidebar-content {
        padding: 1rem;
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

# Sidebar with app information
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/brain--v2.png", width=80)
    st.title("About")
    st.markdown("""
    ### Personal Knowledge Assistant
    
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

# Function to display custom chat bubbles
def display_message(role, content):
    avatar = "👤" if role == "user" else "🧠"
    st.markdown(f"""
    <div class="chat-message {role}">
        <div class="avatar">{avatar}</div>
        <div class="content"><strong>{'You' if role == 'user' else 'Assistant'}:</strong><br>{content}</div>
    </div>
    """, unsafe_allow_html=True)

# Display chat messages from history
if not st.session_state.messages:
    st.info("👋 Hello! I'm your Personal Knowledge Assistant. Ask me anything about your documents!")

for message in st.session_state.messages:
    display_message(message["role"], message["content"])

# Replace your input area section with this:
input_container = st.container()

# Accept user input
with input_container:
    col1, col2 = st.columns([6, 1])
    with col1:
        # Create a unique key for the input that changes when needed
        input_key = f"input_{len(st.session_state.messages)}"
        user_input = st.text_input("Type your question here...", key=input_key, label_visibility="collapsed")
    with col2:
        submit_button = st.button("Send", type="primary")

# Check for Enter key press (by checking if input has content)
submitted = submit_button or (user_input and not st.session_state.thinking)

# Process user input
if submitted and user_input:
    if not st.session_state.thinking:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.thinking = True
        st.rerun()
    else:
        # Display thinking animation
        with st.spinner("Thinking..."):
            # Use conversation agent
            response = agent.process_message(user_input)
            
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.thinking = False
        st.rerun()