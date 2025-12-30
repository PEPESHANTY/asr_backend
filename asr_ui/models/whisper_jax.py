import requests
import io
from typing import Optional, Dict, Any
from .base import ASRModel


class WhisperJAXModel(ASRModel):
    """Whisper JAX model using HTTP API endpoint."""
    
    def __init__(self, endpoint: str = "http://127.0.0.1:8008/transcribe"):
        self.endpoint = endpoint
        self._available_languages = ["en", "vi", "hi", "auto"]
        
    def transcribe(
        self,
        audio_bytes: bytes,
        task: str = "transcribe",
        language: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Transcribe audio using Whisper JAX endpoint.
        
        Additional kwargs can include:
            num_beams, temperature, chunk_sec, stride_leading, stride_trailing, prompt
        """
        form_data = {
            "task": task,
            "return_timestamps": "false"
        }
        
        if language:
            form_data["language"] = language
            
        # Add optional parameters from kwargs
        optional_params = ["num_beams", "temperature", "chunk_sec", 
                         "stride_leading", "stride_trailing", "prompt"]
        for param in optional_params:
            if param in kwargs and kwargs[param] is not None:
                form_data[param] = str(kwargs[param])
        
        files = {"file": ("audio.wav", io.BytesIO(audio_bytes), "audio/wav")}
        
        try:
            response = requests.post(self.endpoint, files=files, data=form_data, timeout=None)
            response.raise_for_status()
            result = response.json()
            return (result.get("text") or "").strip() if isinstance(result, dict) else str(result)
        except Exception as e:
            raise Exception(f"Transcription failed: {str(e)}")
    
    def get_available_languages(self) -> list:
        return self._available_languages.copy()
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "Whisper JAX",
            "endpoint": self.endpoint,
            "supported_languages": self._available_languages,
            "tasks": ["transcribe", "translate"]
        }
