import requests
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass
import json

@dataclass
class LLMResponse:
    success: bool
    content: Optional[str] = None
    error: Optional[str] = None
    provider: Optional[str] = None

class LLMClient:
    def __init__(self, api_key: str, endpoint: str, model: str, timeout: int = 10):
        self.api_key = api_key
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout
        
    def _prepare_headers(self) -> Dict[str, str]:
        """Prepare HTTP headers for the API request."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
    def _prepare_payload(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare the request payload based on content type."""
        if content['type'] == 'text':
            return {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "Process this copied content."},
                    {"role": "user", "content": content['content']}
                ]
            }
        elif content['type'] == 'image':
            return {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "Process this copied image."},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{content['content']}"
                                }
                            }
                        ]
                    }
                ]
            }
        else:
            raise ValueError(f"Unsupported content type: {content['type']}")

    def send_request(self, content: Dict[str, Any]) -> LLMResponse:
        """Send a request to the LLM API."""
        try:
            payload = self._prepare_payload(content)
            headers = self._prepare_headers()
            
            response = requests.post(
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    return LLMResponse(
                        success=True,
                        content=result['choices'][0]['message']['content'],
                        provider=self.model
                    )
                except (KeyError, json.JSONDecodeError) as e:
                    return LLMResponse(
                        success=False,
                        error=f"Failed to parse API response: {str(e)}"
                    )
            else:
                return LLMResponse(
                    success=False,
                    error=f"API request failed with status {response.status_code}"
                )
                
        except requests.Timeout:
            return LLMResponse(
                success=False,
                error="Request timed out"
            )
        except Exception as e:
            return LLMResponse(
                success=False,
                error=f"Request failed: {str(e)}"
            )

class LLMManager:
    def __init__(self, primary_client: LLMClient, fallback_client: Optional[LLMClient] = None):
        self.primary = primary_client
        self.fallback = fallback_client
        
    def process_content(self, content: Dict[str, Any]) -> LLMResponse:
        """Process content using primary LLM with fallback support."""
        # Try primary LLM
        response = self.primary.send_request(content)
        if response.success:
            return response
            
        # If primary fails and fallback is available, try fallback
        if self.fallback and not response.success:
            print(f"Primary LLM failed: {response.error}. Trying fallback...")
            fallback_response = self.fallback.send_request(content)
            if fallback_response.success:
                return fallback_response
            print(f"Fallback LLM also failed: {fallback_response.error}")
            
        return response  # Return primary error if no fallback or both failed
