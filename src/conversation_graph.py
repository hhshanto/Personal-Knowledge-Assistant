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

def extract_personal_info(state: ConversationState, rag_processor) -> Dict[str, Any]:
    """Extract personal information using LLM instead of regex."""
    question = state["current_question"]
    memory = state.get("memory", {})
    
    # Use a structured extraction prompt
    extraction_prompt = f"""
    Extract ONLY IMPORTANT personal information from message. 
    If none is present or it's trivial, return "None".
    
    Important information includes: names, professions, relationships, preferences.
    Trivial information to ignore: greetings, small talk.
    
    Message: "{question}"
    
    Please respond in this JSON format only:
    {{
        "name": "extracted name or null if no name mentioned",
        "other_info": "any other personal info or null"
    }}
    """
    
    try:
        # Get structured output from LLM
        response = rag_processor.llm.invoke(extraction_prompt)
        
        # Use regex just to extract the JSON portion (not for name extraction)
        import json
        import re
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        
        if json_match:
            extracted_info = json.loads(json_match.group(0))
            
            # Update memory with extracted information
            if extracted_info.get("name") and extracted_info["name"] != "null":
                memory["name"] = extracted_info["name"]
                
            if extracted_info.get("other_info") and extracted_info["other_info"] != "null":
                memory["other_info"] = extracted_info["other_info"]
        
    except Exception as e:
        print(f"Error in information extraction: {str(e)}")
    
    return memory
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
        personal_info_pattern = re.compile(r"(what'?s? my name|who am i)", re.IGNORECASE)
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
        memory = extract_personal_info(state, rag_processor)
        
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
Remember personal information like names when shared, but don't explicitly mention how you know this information.
Keep your tone warm and conversational, not academic or formal.
If you don't have enough information, say so in a friendly way.
Never use hashtags in your responses.
Never mention "previous chat" or "as you mentioned earlier" in your responses. Also you don't have to always start with Hey "name".

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
        """Determine if we should continue or end with more sophisticated logic."""
        # Existing logic - explicit completion flag
        if state.get("completed", False):
            return "end"
        
        # Add context-based ending conditions
        messages = state.get("messages", [])
        if messages:
            last_message = messages[-1]
            
            # 1. Check for conversation ending phrases
            if isinstance(last_message, AIMessage) and any(phrase in last_message.content.lower() 
                for phrase in ["goodbye", "farewell", "have a nice day", "have a great day"]):
                return "end"
            
            # 2. Check for request completion indicators
            if isinstance(last_message, AIMessage) and any(phrase in last_message.content.lower()
                for phrase in ["is there anything else", "can i help you with anything else", 
                            "do you have any other questions"]):
                # This is a good stopping point, though we allow continuation
                pass
            
            # 3. Check for long conversation - might need restarting for performance
            if len(messages) > 20:  # Consider restarting after very long conversations
                # Optional: could end based on length, but we'll continue
                pass
        
        # Check question complexity indicators
        current_question = state.get("current_question", "")
        if current_question and len(current_question) > 200:
            # Long questions might need special handling
            pass
        
        # Default: continue processing
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

def prune_messages(messages: List[Union[HumanMessage, AIMessage]], max_messages: int = 10) -> List[Union[HumanMessage, AIMessage]]:
    """Prune messages to keep state size manageable."""
    if len(messages) <= max_messages:
        return messages
        
    # Always keep the first system message if present
    first_message = []
    if messages and messages[0].type == "system":
        first_message = [messages[0]]
        messages = messages[1:]
        
    # Keep the most recent messages
    return first_message + messages[-max_messages:]


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
        
        # Prune messages if they get too long
        self.conversation_state["messages"] = prune_messages(
            self.conversation_state["messages"],
            max_messages=10  # Adjust based on your needs
        )
        
        # Pass the accumulated state to the graph
        result = self.graph.invoke(self.conversation_state)
        
        # Update the conversation state for next time
        self.conversation_state = result
        
        # Return just the latest response
        last_message = result["messages"][-1]
        return last_message.content