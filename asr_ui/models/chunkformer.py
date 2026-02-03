import os
import requests
import io
from typing import Optional, Dict, Any, List
from .base import ASRModel
from ..core.audio_utils import convert_to_wav_bytes


class ChunkformerModel(ASRModel):
    """Chunkformer Vietnamese ASR model using external API."""
    
    def __init__(self, endpoint: str = None, api_key: str = None):
        """
        Initialize the Chunkformer API model.
        
        Args:
            endpoint: API endpoint URL (default from environment)
            api_key: API key (default from environment)
        """
        self.endpoint = endpoint or os.getenv("CHUNKFORMER_ENDPOINT", "http://hanoi2.ucd.ie/asr_chunkformer")
        self.api_key = api_key or os.getenv("CHUNKFORMER_API_KEY", "AIRRVie_api_key")
        
    def transcribe(
        self,
        audio_bytes: bytes,
        task: str = "transcribe",
        language: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Transcribe audio bytes to text using Chunkformer API.
        
        Args:
            audio_bytes: Audio data in bytes
            task: Only "transcribe" is supported
            language: Ignored (model is Vietnamese-specific)
            **kwargs: Additional parameters (return_timestamps)
            
        Returns:
            Transcribed text
        """
        if task != "transcribe":
            raise ValueError("Chunkformer API only supports transcription, not translation")
        
        # Convert audio to WAV format to ensure compatibility
        try:
            wav_bytes = convert_to_wav_bytes(audio_bytes)
        except Exception as e:
            raise Exception(f"Audio format conversion failed: {str(e)}")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        
        audio_stream = io.BytesIO(wav_bytes)
        audio_stream.seek(0)
        
        import time
        timestamp = int(time.time() * 1000)
        filename = f"audio_{timestamp}.wav"
        
        files = {
            "audio": (filename, audio_stream, "audio/wav")
        }
        
        data = {
            "return_timestamps": str(kwargs.get("return_timestamps", False)).lower()
        }
        
        print(f"[DEBUG] Calling Chunkformer API: {self.endpoint}")
        print(f"[DEBUG] Original audio size: {len(audio_bytes)} bytes")
        print(f"[DEBUG] WAV audio size: {len(wav_bytes)} bytes")
        print(f"[DEBUG] API Key: {self.api_key[:10]}..." if self.api_key else "No API key")
        
        try:
            response = requests.post(
                self.endpoint,
                headers=headers,
                files=files,
                data=data,
                timeout=120
            )
            
            print(f"[DEBUG] Response status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"[DEBUG] Response error: {response.text}")
                raise Exception(f"API request failed with status {response.status_code}: {response.text}")
            
            result = response.json()
            print(f"[DEBUG] Response JSON: {result}")
            
            if result.get("status") == "error":
                raise Exception(f"API error: {result.get('message', 'Unknown error')}")
            
            text = result.get('text', '').strip()
            print(f"[DEBUG] Extracted text: {text}")
            return text
            
        except Exception as e:
            print(f"[DEBUG] Exception in transcribe: {e}")
            raise Exception(f"Transcription failed: {e}")
    
    def get_available_languages(self) -> list:
        """Get list of supported language codes."""
        return ["vi"]  # Vietnamese only
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "name": "Chunkformer Vietnamese ASR",
            "model_id": "khanhld/chunkformer-ctc-large-vie",
            "endpoint": self.endpoint,
            "supported_languages": ["vi"],
            "task": "transcribe",
            "provider": "Chunkformer API"
        }