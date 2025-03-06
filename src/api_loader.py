from openai import OpenAI, AzureOpenAI
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from dotenv import load_dotenv
import os
from typing import Optional, Union, Dict, Any

class LLMLoader:
    """
    A class to load and configure OpenAI or Azure OpenAI language models.
    Compatible with direct API calls and LangChain integration.
    """
    
    def __init__(self, use_azure: bool = False):
        """
        Initialize the LLM loader.
        
        Args:
            use_azure: Whether to use Azure OpenAI (True) or regular OpenAI (False)
        """
        load_dotenv()
        self.use_azure = use_azure
        self.client = self._initialize_client()
        
    def _initialize_client(self) -> Union[OpenAI, AzureOpenAI]:
        """Initialize the appropriate OpenAI client based on configuration."""
        if self.use_azure:
            # Check required environment variables
            required_vars = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_ENDPOINT"]
            for var in required_vars:
                if not os.getenv(var):
                    raise ValueError(f"Environment variable {var} is required for Azure OpenAI")
                    
            return AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            )
        else:
            # Check OpenAI API key
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("Environment variable OPENAI_API_KEY is required for OpenAI")
                
            return OpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )
    
    def get_completion(self, 
                       prompt: str, 
                       model: str = None, 
                       temperature: float = 0.7, 
                       max_tokens: int = 500,
                       **kwargs) -> str:
        """
        Get a completion from the OpenAI model.
        
        Args:
            prompt: The text prompt to complete
            model: The model to use (default depends on provider)
            temperature: The temperature for generation (0.0-1.0)
            max_tokens: Maximum tokens in the response
            **kwargs: Additional arguments to pass to the API
            
        Returns:
            The generated text as a string
        """
        if not model:
            model = "gpt-35-turbo" if self.use_azure else "gpt-3.5-turbo"
            
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error getting completion: {str(e)}")
    
    def get_langchain_llm(self, 
                          model: str = None, 
                          temperature: float = 0.7, 
                          **kwargs) -> Union[ChatOpenAI, AzureChatOpenAI]:
        """
        Get a LangChain-compatible LLM instance.
        
        Args:
            model: The model name to use
            temperature: The temperature for generation
            **kwargs: Additional arguments for the LLM
            
        Returns:
            A LangChain LLM instance
        """
        if not model:
            model = "gpt-35-turbo" if self.use_azure else "gpt-3.5-turbo"
            
        if self.use_azure:
            deployment_name = model  # In Azure, this is your deployment name
            return AzureChatOpenAI(
                openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                deployment_name=deployment_name,
                temperature=temperature,
                **kwargs
            )
        else:
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                **kwargs
            )

# Example usage
if __name__ == "__main__":
    # Test with Azure OpenAI
    try:
        azure_loader = LLMLoader(use_azure=True)
        response = azure_loader.get_completion(
            "Once upon a time, in a land far, far away,",
            max_tokens=100
        )
        print("Response from Azure OpenAI:")
        print(response)
        
        # Get LangChain model
        llm = azure_loader.get_langchain_llm()
        print("\nLangChain model initialized successfully!")
        
    except Exception as e:
        print(f"Azure test error: {str(e)}")
    
    # Test with regular OpenAI
    try:
        openai_loader = LLMLoader(use_azure=False)
        response = openai_loader.get_completion(
            "Tell me a short joke.",
            max_tokens=50
        )
        print("\nResponse from OpenAI:")
        print(response)
        
    except Exception as e:
        print(f"OpenAI test error: {str(e)}")