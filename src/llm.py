import os
import openai
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

load_dotenv()

class OpenAIClient:
    def __init__(self, api_key: str = None):
        """
        Initialize the OpenAI client with API key.
        
        Args:
            api_key: OpenAI API key. If None, will try to load from environment variable.
        """
        try:
            if api_key is None:
                self.api_key = os.environ.get("OPENAI_API_KEY")
            else:
                self.api_key = api_key
            if not self.api_key:
                raise ValueError("OpenAI API key not found in environment variables")
        except Exception as e:
            raise ValueError(f"Error initializing OpenAI client: {e}")
        
        self.client = openai.OpenAI(api_key=self.api_key)

    def generate_response(
        self, 
        prompt: str,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        force_json: bool = False,
    ) -> str:
        """
        Generate a text response using OpenAI's chat completion API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            model: The OpenAI model to use
            temperature: Controls randomness (0-1)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            The generated text response
        """
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            response_format={"type": "json_object"} if force_json else None,
        )
        return response.choices[0].message.content
    
    def generate_response_with_file(
        self,
        prompt: str,
        file_path: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        force_json: bool = False,
    ) -> str:
        file = self.client.files.create(
                file=open(file_path, "rb"),
                purpose="user_data"
        )
        completion = self.client.chat.completions.create(
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"} if force_json else None,
            messages=[
                {"role": "user",
                    "content": [
                        {
                    "type": "file",
                        "file": {
                        "file_id": file.id,
                    }
                },
                {
                    "type": "text",
                            "text": prompt,
                        },
                    ]
                }
            ]
        )
        return completion.choices[0].message.content

# Example usage
if __name__ == "__main__":
    client = OpenAIClient()
    
    # Example with PDF file (uncomment to test)
    file_response = client.generate_response_with_file(
        prompt="explain the details of this paper in order?",
        file_path="data/Cities/pdf/10.1016@j.cities.2019.01.039.pdf",
        model="gpt-4o-mini"
    )
    print("File response:", file_response)
