import io
import numpy as np
import sounddevice as sd
import soundfile as sf
import soundfile as sf
from pathlib import Path
import wave


def ensure_dir(p: Path) -> None:
    """Ensure directory exists."""
    p.mkdir(parents=True, exist_ok=True)


def wav_bytes_from_array(x_int16: np.ndarray, sr: int) -> bytes:
    """Convert int16 numpy array to WAV bytes using standard wave module for compatibility."""
    buf = io.BytesIO()
    # Use standard wave module to ensure PCM WAV format
    with wave.open(buf, 'wb') as wav_file:
        wav_file.setnchannels(1)  # mono
        wav_file.setsampwidth(2)  # 2 bytes for int16
        wav_file.setframerate(sr)
        wav_file.writeframes(x_int16.tobytes())
    return buf.getvalue()


def record_audio(seconds: float, sr: int, device=None) -> np.ndarray:
    """Record audio and return as int16 numpy array."""
    if device is not None:
        sd.default.device = (device, None)
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="int16")
    sd.wait()
    return audio.reshape(-1)
