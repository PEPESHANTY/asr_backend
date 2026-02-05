from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any, List
import uvicorn
import os
from dotenv import load_dotenv
import logging

# Load environment variables from .env file (if present)
load_dotenv()

from ..models import get_model, MODEL_REGISTRY
from ..core.audio_utils import wav_bytes_from_array, record_audio

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = FastAPI(title="ASR API", description="Automatic Speech Recognition API")

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("asr_api")

# -----------------------------------------------------------------------------
# CORS
# -----------------------------------------------------------------------------
# Put this in Coolify env (recommended):
# CORS_ORIGINS=https://asr-models.pepeshanty.store,https://asr-models-backend.pepeshanty.store,http://localhost:3000
#
# Or allow all (only if you understand the risk):
# CORS_ORIGINS=*
CORS_ORIGINS_RAW = os.getenv("CORS_ORIGINS", "")

DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "https://asr-models.pepeshanty.store",
    "https://asr-models-backend.pepeshanty.store",

]

def parse_cors_origins(raw: str) -> List[str]:
    raw = (raw or "").strip()
    if raw == "" or raw.lower() in {"none", "null"}:
        # fallback to defaults if not set
        return DEFAULT_CORS_ORIGINS
    if raw == "*":
        return ["*"]
    # comma-separated list
    return [o.strip() for o in raw.split(",") if o.strip()]

CORS_ORIGINS = parse_cors_origins(CORS_ORIGINS_RAW)
logger.info(f"[CORS] Allowed origins: {CORS_ORIGINS}")

# NOTE: If you use allow_credentials=True, using ["*"] is not recommended.
# In production, set explicit origins via CORS_ORIGINS env var.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Model selection and cache
# -----------------------------------------------------------------------------
DEFAULT_MODEL = os.getenv("ASR_DEFAULT_MODEL", "whisper_jax")
model_cache: Dict[str, Any] = {}

def get_model_for_request(model_name: str):
    """Get or create a model instance for the given model name."""
    if model_name in model_cache:
        return model_cache[model_name]

    try:
        if model_name == "whisper_jax":
            endpoint = os.getenv("WHISPER_ENDPOINT", "http://35.186.40.29:8008/transcribe")
            model = get_model(model_name, endpoint=endpoint)
        elif model_name == "omni_lingual":
            endpoint = os.getenv("OMNILINGUAL_ENDPOINT", "http://hanoi2.ucd.ie/asr_omnilingual")
            api_key = os.getenv("OMNILINGUAL_API_KEY", "")
            model = get_model(model_name, endpoint=endpoint, api_key=api_key)
        elif model_name == "chunkformer":
            endpoint = os.getenv("CHUNKFORMER_ENDPOINT", "http://hanoi2.ucd.ie/asr_chunkformer")
            api_key = os.getenv("CHUNKFORMER_API_KEY", "")
            model = get_model(model_name, endpoint=endpoint, api_key=api_key)
        elif model_name == "qwen3_1_7B":
            endpoint = os.getenv("QWEN3_1_7B_ENDPOINT", "http://hanoi2.ucd.ie/asr_q3_1_7B")
            api_key = os.getenv("QWEN3_1_7B_API_KEY", "AIRRVie_api_key")
            model = get_model(model_name, endpoint=endpoint, api_key=api_key)
        elif model_name == "qwen3_0_6B":
            endpoint = os.getenv("QWEN3_0_6B_ENDPOINT", "http://hanoi2.ucd.ie/asr_q3_0_6B")
            api_key = os.getenv("QWEN3_0_6B_API_KEY", "AIRRVie_api_key")
            model = get_model(model_name, endpoint=endpoint, api_key=api_key)
        else:
            model = get_model(model_name)

        model_cache[model_name] = model
        return model
    except Exception as e:
        logger.exception(f"Failed to load model {model_name}: {e}")
        raise

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/")
async def root():
    return {"message": "ASR API is running", "default_model": DEFAULT_MODEL}

@app.get("/health")
async def health():
    """
    Light health check.
    IMPORTANT: Don't force loading external models here, otherwise healthcheck can fail
    if the upstream endpoint is temporarily down.
    """
    return {"status": "ok"}

@app.get("/health/model")
async def health_model():
    """
    Optional deeper health check that verifies default model is reachable.
    You can keep Coolify healthcheck on /health (recommended).
    """
    try:
        model = get_model_for_request(DEFAULT_MODEL)
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
        "models_info": models_info,
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
    """Transcribe an uploaded audio file."""
    try:
        asr_model = get_model_for_request(model)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"ASR model '{model}' not available: {str(e)}")

    contents = await file.read()

    logger.info(
        f"[UPLOAD] filename={file.filename}, size={len(contents)} bytes, mime_type={file.content_type}"
    )

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

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
        text = asr_model.transcribe(
            audio_bytes=contents,
            task=task,
            language=language,
            **extra_params,
        )
        return {
            "text": text,
            "task": task,
            "language": language,
            "model": model,
            "file_name": file.filename,
            "file_size": len(contents),
        }
    except Exception as e:
        logger.exception(f"Transcription failed: {e}")
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
    """Record audio from microphone and transcribe."""
    try:
        asr_model = get_model_for_request(model)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"ASR model '{model}' not available: {str(e)}")

    # IMPORTANT: recording from microphone inside a server/container usually doesn't work
    # unless the container has audio devices passed through. You already handled errors
    # in record_audio; we keep try/except anyway.
    try:
        audio_array = record_audio(seconds, sample_rate, device)
        audio_bytes = wav_bytes_from_array(audio_array, sample_rate)
    except Exception as e:
        logger.exception(f"Recording failed: {e}")
        raise HTTPException(status_code=500, detail=f"Recording failed: {str(e)}")

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
        text = asr_model.transcribe(
            audio_bytes=audio_bytes,
            task=task,
            language=language,
            **extra_params,
        )
        return {
            "text": text,
            "task": task,
            "language": language,
            "model": model,
            "recorded_seconds": seconds,
            "sample_rate": sample_rate,
        }
    except Exception as e:
        logger.exception(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

# -----------------------------------------------------------------------------
# Local run
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
