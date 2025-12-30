from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class ASRModel(ABC):
    """Abstract base class for all ASR models."""
    
    @abstractmethod
    def transcribe(
        self,
        audio_bytes: bytes,
        task: str = "transcribe",
        language: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Transcribe audio bytes to text.
        
        Args:
            audio_bytes: Audio data in bytes
            task: "transcribe" or "translate"
            language: Language code (e.g., "en", "vi")
            **kwargs: Additional model-specific parameters
            
        Returns:
            Transcribed text
        """
        pass
    
    @abstractmethod
    def get_available_languages(self) -> list:
        """Get list of supported language codes."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        pass
