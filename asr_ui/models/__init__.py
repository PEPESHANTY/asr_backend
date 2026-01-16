from .base import ASRModel
from .whisper_jax import WhisperJAXModel
from .omni_lingual import OmniLingualAPIModel

# Model registry
MODEL_REGISTRY = {
    "whisper_jax": WhisperJAXModel,
    "omni_lingual": OmniLingualAPIModel,
    "chunkformer": ChunkformerModel,
}

def get_model(model_name: str, **kwargs) -> ASRModel:
    """
    Factory function to get an ASR model instance.
    
    Args:
        model_name: Name of the model (e.g., "whisper_jax")
        **kwargs: Arguments to pass to the model constructor
        
    Returns:
        An instance of the requested ASR model
        
    Raises:
        ValueError: If the model name is not in the registry
    """
    if model_name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Model '{model_name}' not found. Available models: {available}")
    
    model_class = MODEL_REGISTRY[model_name]
    return model_class(**kwargs)

__all__ = ["ASRModel", "WhisperJAXModel", "OmniLingualAPIModel", "get_model", "MODEL_REGISTRY"]
