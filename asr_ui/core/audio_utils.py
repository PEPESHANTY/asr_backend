import io
import numpy as np
from pathlib import Path
import wave

try:
    import sounddevice as sd
except Exception:
    sd = None

try:
    import soundfile as sf
    import librosa
except Exception:
    sf = None
    librosa = None


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def wav_bytes_from_array(x_int16: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sr)
        wav_file.writeframes(x_int16.tobytes())
    return buf.getvalue()


def convert_to_wav_bytes(audio_bytes: bytes) -> bytes:
    """
    Convert audio bytes from any format (m4a, mp3, wav, etc.) to WAV format.
    
    Args:
        audio_bytes: Audio data in any format
        
    Returns:
        Audio data in WAV format
    """
    if sf is None or librosa is None:
        raise RuntimeError("soundfile and librosa are required for audio format conversion")
    
    # Load audio using soundfile/librosa (supports many formats)
    audio_stream = io.BytesIO(audio_bytes)
    try:
        # Try soundfile first (faster, supports more formats)
        audio_array, sample_rate = sf.read(audio_stream)
    except Exception:
        # Fall back to librosa
        audio_stream.seek(0)
        audio_array, sample_rate = librosa.load(audio_stream, sr=None, mono=False)
    
    # Convert to mono if stereo
    if len(audio_array.shape) > 1:
        audio_array = np.mean(audio_array, axis=1)
    
    # Convert to int16 format
    audio_int16 = (audio_array * 32767).astype(np.int16)
    
    # Create WAV bytes
    return wav_bytes_from_array(audio_int16, int(sample_rate))


def record_audio(seconds: float, sr: int, device=None) -> np.ndarray:
    if sd is None:
        raise RuntimeError("sounddevice/PortAudio not available in this environment")

    if device is not None:
        sd.default.device = (device, None)

    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="int16")
    sd.wait()
    return audio.reshape(-1)
