import os
import requests
import io
from typing import Optional, Dict, Any, List
from .base import ASRModel


class OmniLingualAPIModel(ASRModel):
    """OmniLingual ASR model using external API."""
    
    def __init__(self, endpoint: str = None, api_key: str = None):
        """
        Initialize the OmniLingual API model.
        
        Args:
            endpoint: API endpoint URL (default from environment)
            api_key: API key (default from environment)
        """
        self.endpoint = endpoint or os.getenv("OMNILINGUAL_ENDPOINT", "http://hanoi2.ucd.ie/asr")
        self.api_key = api_key or os.getenv("OMNILINGUAL_API_KEY", "")
        self.supported_languages = self._load_supported_languages()
        
    def _load_supported_languages(self) -> List[str]:
        """Load supported languages for OmniLingual model."""
        # This is a placeholder - OmniLingual supports 1600+ languages
        # We'll return a subset of common languages for now
        common_languages = [
            "eng_Latn",  # English (Latin)
            "vie_Latn",  # Vietnamese (Latin)
            "fra_Latn",  # French (Latin)
            "spa_Latn",  # Spanish (Latin)
            "deu_Latn",  # German (Latin)
            "ita_Latn",  # Italian (Latin)
            "por_Latn",  # Portuguese (Latin)
            "rus_Cyrl",  # Russian (Cyrillic)
            "jpn_Jpan",  # Japanese (Japanese script)
            "kor_Hang",  # Korean (Hangul)
            "cmn_Hans",  # Chinese (Simplified)
            "cmn_Hant",  # Chinese (Traditional)
            "ara_Arab",  # Arabic (Arabic script)
            "hin_Deva",  # Hindi (Devanagari)
        ]
        return common_languages
    
    def transcribe(
        self,
        audio_bytes: bytes,
        task: str = "transcribe",
        language: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Transcribe audio bytes to text using OmniLingual API.
        
        Args:
            audio_bytes: Audio data in bytes
            task: Only "transcribe" is supported (OmniLingual doesn't do translation)
            language: Language code in format {language_code}_{script} (e.g., "vie_Latn")
            **kwargs: Additional parameters (ignored for API)
            
        Returns:
            Transcribed text
        """
        if task != "transcribe":
            raise ValueError("OmniLingual API only supports transcription, not translation")
        
        # Prepare the request headers and data as per the curl example
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "accept": "application/json",
        }
        
        # Prepare form data exactly as curl does
        # curl: -F "audio=@Test.wav" -F "lang_code=vie_Latn"
        # We'll use the same field names and filename
        # Ensure the BytesIO is at position 0
        import time
        audio_stream = io.BytesIO(audio_bytes)
        audio_stream.seek(0)
        
        # Use unique filename to avoid caching issues
        timestamp = int(time.time() * 1000)
        filename = f"audio_{timestamp}.wav"
        files = {
            "audio": (filename, audio_stream, "audio/wav")
        }
        
        data = {}
        if language:
            data["lang_code"] = language
        else:
            # Default to English if not specified
            data["lang_code"] = "eng_Latn"
        
        # Compute hash of audio bytes to verify uniqueness
        import hashlib
        audio_hash = hashlib.md5(audio_bytes).hexdigest()[:8]
        
        print(f"[DEBUG] Calling OmniLingual API: {self.endpoint}")
        print(f"[DEBUG] Language: {data.get('lang_code')}")
        print(f"[DEBUG] Audio size: {len(audio_bytes)} bytes")
        print(f"[DEBUG] Audio hash: {audio_hash}")
        print(f"[DEBUG] API Key present: {'Yes' if self.api_key else 'No'}")
        print(f"[DEBUG] Using filename: {filename}")
        print(f"[DEBUG] First 100 bytes: {audio_bytes[:100].hex()}")
        
        try:
            # Make the request
            response = requests.post(
                self.endpoint,
                headers=headers,
                files=files,
                data=data,
                timeout=60
            )
            
            print(f"[DEBUG] Response status: {response.status_code}")
            print(f"[DEBUG] Response headers: {response.headers}")
            
            if response.status_code != 200:
                print(f"[DEBUG] Response error: {response.text}")
                raise Exception(f"API request failed with status {response.status_code}: {response.text}")
            
            # Parse response
            result = response.json()
            print(f"[DEBUG] Response JSON: {result}")
            
            # Extract text from response (check for 'text' field)
            if 'text' in result:
                text = result['text'].strip()
                print(f"[DEBUG] Extracted text: {text}")
                return text
            else:
                print(f"[DEBUG] No 'text' field in response. Available keys: {result.keys()}")
                # Try to get the first value that might be text
                for key, value in result.items():
                    if isinstance(value, str) and len(value) > 0:
                        print(f"[DEBUG] Using alternative field '{key}': {value}")
                        return value.strip()
                raise Exception(f"No text field found in response: {result}")
            
        except Exception as e:
            print(f"[DEBUG] Exception in transcribe: {e}")
            raise Exception(f"Transcription failed: {e}")
    
    def get_available_languages(self) -> list:
        """Get list of supported language codes."""
        return self.supported_languages
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "name": "OmniLingual API",
            "endpoint": self.endpoint,
            "supported_languages_count": len(self.supported_languages),
            "supported_languages": self.supported_languages[:10],  # First 10
            "task": "transcribe",
            "provider": "External API"
        }
