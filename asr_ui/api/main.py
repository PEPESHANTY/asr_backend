from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any
import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from ..models import get_model, MODEL_REGISTRY
from ..core.audio_utils import wav_bytes_from_array, record_audio

app = FastAPI(title="ASR API", description="Automatic Speech Recognition API")

# CORS middleware configuration for production and development
CORS_ORIGINS_RAW = os.getenv(
    "CORS_ORIGINS", 
    "http://localhost:3000",
    "https://asr-models.pepeshanty.store",
    "https://asr-models-backend.pepeshanty.store"
)

# Split and strip whitespace to handle environment variable formatting
CORS_ORIGINS = [origin.strip() for origin in CORS_ORIGINS_RAW.split(",") if origin.strip()]

# Log allowed origins for debugging (in production, this should be in logs)
print(f"[CORS] Allowed origins: {CORS_ORIGINS}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Default model (can be configured via environment variable)
DEFAULT_MODEL = os.getenv("ASR_DEFAULT_MODEL", "whisper_jax")

# Model cache
model_cache: Dict[str, Any] = {}


def get_model_for_request(model_name: str):
    """Get or create a model instance for the given model name."""
    if model_name in model_cache:
        return model_cache[model_name]

    try:
        if model_name == "whisper_jax":
            endpoint = os.getenv("WHISPER_ENDPOINT", "http://127.0.0.1:8008/transcribe")
            model = get_model(model_name, endpoint=endpoint)
        elif model_name == "omni_lingual":
            endpoint = os.getenv("OMNILINGUAL_ENDPOINT", "http://hanoi2.ucd.ie/asr")
            api_key = os.getenv("OMNILINGUAL_API_KEY", "")
            model = get_model(model_name, endpoint=endpoint, api_key=api_key)
        else:
            # For any other model, try to get it without additional arguments
            model = get_model(model_name)

        model_cache[model_name] = model
        return model
    except Exception as e:
        print(f"Failed to load model {model_name}: {e}")
        raise


@app.get("/")
async def root():
    return {"message": "ASR API is running", "default_model": DEFAULT_MODEL}


@app.get("/health")
async def health():
    """Health check using the default model."""
    try:
        model = get_model_for_request(DEFAULT_MODEL)
        # Try to get model info to verify it's working
        model.get_model_info()
        return {"status": "healthy", "model": DEFAULT_MODEL}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model {DEFAULT_MODEL} not loaded: {str(e)}")


@app.get("/models")
async def list_models():
    """List all available models and their information."""
    models_info = {}
    
    for model_name in MODEL_REGISTRY.keys():
        try:
            model = get_model_for_request(model_name)
            models_info[model_name] = model.get_model_info()
        except Exception as e:
            models_info[model_name] = {"error": str(e), "status": "failed_to_load"}
    
    return {
        "available_models": list(MODEL_REGISTRY.keys()),
        "default_model": DEFAULT_MODEL,
        "models_info": models_info
    }


@app.post("/transcribe/upload")
async def transcribe_upload(
    file: UploadFile = File(...),
    task: str = Form("transcribe"),
    language: Optional[str] = Form(None),
    model: str = Form(DEFAULT_MODEL),
    num_beams: Optional[int] = Form(None),
    temperature: Optional[float] = Form(None),
    chunk_sec: Optional[float] = Form(None),
    stride_leading: Optional[float] = Form(None),
    stride_trailing: Optional[float] = Form(None),
    prompt: Optional[str] = Form(None),
):
    """
    Transcribe an uploaded audio file.
    """
    try:
        asr_model = get_model_for_request(model)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"ASR model '{model}' not available: {str(e)}")
    
    # Read the uploaded file
    contents = await file.read()
    
    print(f"[DEBUG] Received upload: filename={file.filename}, size={len(contents)} bytes, mime_type={file.content_type}")
    
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    
    # Determine mime type
    mime_type = file.content_type or "audio/wav"
    
    # Prepare additional parameters
    extra_params = {}
    if num_beams is not None:
        extra_params["num_beams"] = num_beams
    if temperature is not None:
        extra_params["temperature"] = temperature
    if chunk_sec is not None:
        extra_params["chunk_sec"] = chunk_sec
    if stride_leading is not None:
        extra_params["stride_leading"] = stride_leading
    if stride_trailing is not None:
        extra_params["stride_trailing"] = stride_trailing
    if prompt is not None:
        extra_params["prompt"] = prompt
    
    try:
        # Transcribe
        text = asr_model.transcribe(
            audio_bytes=contents,
            task=task,
            language=language,
            **extra_params
        )
        return {
            "text": text,
            "task": task,
            "language": language,
            "model": model,
            "file_name": file.filename,
            "file_size": len(contents)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.post("/transcribe/record")
async def transcribe_record(
    task: str = Form("transcribe"),
    language: Optional[str] = Form(None),
    model: str = Form(DEFAULT_MODEL),
    seconds: float = Form(8.0),
    sample_rate: int = Form(16000),
    device: Optional[str] = Form(None),
    num_beams: Optional[int] = Form(None),
    temperature: Optional[float] = Form(None),
    chunk_sec: Optional[float] = Form(None),
    stride_leading: Optional[float] = Form(None),
    stride_trailing: Optional[float] = Form(None),
    prompt: Optional[str] = Form(None),
):
    """
    Record audio from microphone and transcribe.
    """
    try:
        asr_model = get_model_for_request(model)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"ASR model '{model}' not available: {str(e)}")
    
    # Record audio
    try:
        audio_array = record_audio(seconds, sample_rate, device)
        audio_bytes = wav_bytes_from_array(audio_array, sample_rate)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recording failed: {str(e)}")
    
    # Prepare additional parameters
    extra_params = {}
    if num_beams is not None:
        extra_params["num_beams"] = num_beams
    if temperature is not None:
        extra_params["temperature"] = temperature
    if chunk_sec is not None:
        extra_params["chunk_sec"] = chunk_sec
    if stride_leading is not None:
        extra_params["stride_leading"] = stride_leading
    if stride_trailing is not None:
        extra_params["stride_trailing"] = stride_trailing
    if prompt is not None:
        extra_params["prompt"] = prompt
    
    try:
        # Transcribe
        text = asr_model.transcribe(
            audio_bytes=audio_bytes,
            task=task,
            language=language,
            **extra_params
        )
        return {
            "text": text,
            "task": task,
            "language": language,
            "model": model,
            "recorded_seconds": seconds,
            "sample_rate": sample_rate
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")




if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
