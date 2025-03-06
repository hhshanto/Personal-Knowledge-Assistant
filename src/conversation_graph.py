from langchain_core.messages import HumanMessage, AIMessage
from typing import Dict, Any, List, TypedDict, Optional, Union
from langgraph.graph import StateGraph, END
from src.rag_processor import RAGProcessor
import tempfile
import webbrowser
import os
import re

# Define the state structure
class ConversationState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    context: Optional[str]
    current_question: Optional[str]
    completed: bool  # Add a flag to track completion
    memory: Dict[str, Any]  # Add explicit memory for important facts

def create_conversation_graph(rag_processor: RAGProcessor):
    """Create a conversation graph using LangGraph."""
    
    # Define the nodes for our graph
    def retrieve_context(state: ConversationState) -> ConversationState:
        """Retrieve context for the current question."""
        if not state.get("current_question"):
            return {**state, "completed": True}
            
        question = state["current_question"]
        # Add debugging
        print(f"Retrieving context for: {question}")
        
        # Initialize memory if not present
        memory = state.get("memory", {})
        
        # Check if this is a personal information question
        personal_info_pattern = re.compile(r"(what( is|'s)? my name|who am i)", re.IGNORECASE)
        if personal_info_pattern.search(question) and "name" in memory:
            # If asking about stored personal info, no need to retrieve from documents
            return {
                **state,
                "context": f"Personal information: The user's name is {memory['name']}.",
                "memory": memory
            }
        
        result = rag_processor.answer_question(question)
        
        # Add debugging
        print(f"Retrieved context length: {len(result['context'])}")
        
        return {
            **state,
            "context": result["context"],
            "memory": memory
            # Don't mark as completed here - let generate_response do it
        }
    
    def generate_response(state: ConversationState) -> ConversationState:
        """Generate a response using the LLM."""
        if not state.get("current_question") or not state.get("context"):
            return {**state, "completed": True}
            
        # Get conversation history
        conversation_history = "\n".join([
            f"{'You' if isinstance(msg, AIMessage) else 'Human'}: {msg.content}" 
            for msg in state["messages"][-5:]  # Last 5 messages
        ])
            
        question = state["current_question"]
        context = state["context"]
        memory = state.get("memory", {})
        
        # Extract and store personal information (like name)
        name_pattern = re.compile(r"my name is (\w+)", re.IGNORECASE)
        name_match = name_pattern.search(question)
        if name_match:
            memory["name"] = name_match.group(1)
        
        # Include memory context if available
        memory_context = ""
        if memory:
            memory_context = "Personal information:\n"
            for key, value in memory.items():
                memory_context += f"- User's {key}: {value}\n"
        
        prompt = f"""You are a friendly, conversational assistant named Personal Knowledge Assistant. 
Respond in a natural, warm tone as if chatting with a friend.

Conversation History:
{conversation_history}

Context from documents:
{context}

{memory_context}

Question: {question}

Please answer the question using both the retrieved context AND conversation history. 
Remember personal information like names when shared.
Keep your tone warm and conversational, not academic or formal.
If you don't have enough information, say so in a friendly way.

Answer:"""
        
        response = rag_processor.llm.invoke(prompt)
        
        # Add the response to messages
        new_messages = state["messages"] + [
            AIMessage(content=response.content)
        ]
        
        return {
            **state,
            "messages": new_messages,
            "current_question": None,  # Reset the question
            "memory": memory,  # Preserve memory
            "completed": True  # Mark as completed
        }
    
    def should_end(state: ConversationState) -> str:
        """Determine if we should continue or end."""
        if state.get("completed", False):
            return "end"
        return "continue"
    
    # Create the graph
    workflow = StateGraph(ConversationState)
    
    # Add nodes
    workflow.add_node("retrieve", retrieve_context)
    workflow.add_node("generate", generate_response)
    
    # Add edges
    workflow.add_edge("retrieve", "generate")
    workflow.add_conditional_edges(
        "generate", 
        should_end,
        {
            "continue": "retrieve", 
            "end": END
        }
    )
    
    # Set the entry point
    workflow.set_entry_point("retrieve")
    
    # Compile the graph
    graph = workflow.compile()
    
    # Try different visualization approaches based on LangGraph version
    try:
        # Approach 1: Try using .draw() method (common in newer versions)
        temp_file = os.path.join(tempfile.gettempdir(), "conversation_graph.html")
        try:
            workflow.draw(temp_file)
            print(f"Graph visualization saved to: {temp_file}")
            webbrowser.open(f"file://{temp_file}")
        except AttributeError:
            # Approach 2: Try using .to_graphviz() if available
            try:
                dot_graph = workflow.to_graphviz()
                dot_graph.render(temp_file, format="png")
                print(f"Graph visualization saved to: {temp_file}.png")
                webbrowser.open(f"file://{temp_file}.png")
            except AttributeError:
                # Approach 3: Try a minimal text-based visualization
                print("\nGraph Structure (text visualization):")
                print("* Entry point: retrieve")
                print("* Node: retrieve → generate")
                print("* Node: generate → [conditional]")
                print("  - If completed: END")
                print("  - Otherwise: retrieve")
                
    except Exception as e:
        print(f"Could not visualize graph: {str(e)}")
        print("You can still use the graph functionality without visualization")
    
    return graph

class ConversationAgent:
    def __init__(self, rag_processor):
        self.rag_processor = rag_processor
        self.graph = create_conversation_graph(rag_processor)
        self.conversation_state = {
            "messages": [],
            "context": None,
            "current_question": None,
            "completed": True,
            "memory": {}  # Initialize memory
        }
    
    def process_message(self, message):
        # Add the new message to existing conversation
        self.conversation_state["messages"].append(HumanMessage(content=message))
        self.conversation_state["current_question"] = message
        self.conversation_state["completed"] = False
        
        # Pass the accumulated state to the graph
        result = self.graph.invoke(self.conversation_state)
        
        # Update the conversation state for next time
        self.conversation_state = result
        
        # Return just the latest response
        last_message = result["messages"][-1]
        return last_message.content