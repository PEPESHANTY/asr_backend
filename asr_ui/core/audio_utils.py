import io
import numpy as np
from pathlib import Path
import wave
import subprocess
import tempfile
import os

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
    Uses soundfile/librosa first, falls back to ffmpeg for unsupported formats like M4A.
    
    Args:
        audio_bytes: Audio data in any format
        
    Returns:
        Audio data in WAV format
    """
    if sf is None or librosa is None:
        raise RuntimeError("soundfile and librosa are required for audio format conversion")
    
    # Try soundfile/librosa first
    audio_stream = io.BytesIO(audio_bytes)
    try:
        # Try soundfile first (faster, supports more formats like MP3, OGG, FLAC)
        audio_array, sample_rate = sf.read(audio_stream)
    except Exception as e1:
        # Fall back to librosa
        try:
            audio_stream.seek(0)
            audio_array, sample_rate = librosa.load(audio_stream, sr=None, mono=False)
        except Exception as e2:
            # Final fallback: use ffmpeg for formats like M4A/AAC that soundfile doesn't support
            try:
                return _convert_with_ffmpeg(audio_bytes)
            except Exception as e3:
                raise RuntimeError(
                    f"Failed to convert audio format. "
                    f"soundfile error: {str(e1)}, "
                    f"librosa error: {str(e2)}, "
                    f"ffmpeg error: {str(e3)}"
                )
    
    # Convert to mono if stereo
    if len(audio_array.shape) > 1:
        audio_array = np.mean(audio_array, axis=1)
    
    # Convert to int16 format
    audio_int16 = (audio_array * 32767).astype(np.int16)
    
    # Create WAV bytes
    return wav_bytes_from_array(audio_int16, int(sample_rate))


def _convert_with_ffmpeg(audio_bytes: bytes) -> bytes:
    """
    Convert audio to WAV using ffmpeg (fallback for M4A and other formats).
    
    Args:
        audio_bytes: Audio data in any format
        
    Returns:
        Audio data in WAV format
    """
    # Create temporary files for input and output
    with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as input_file:
        input_path = input_file.name
        input_file.write(audio_bytes)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as output_file:
        output_path = output_file.name
    
    try:
        # Use ffmpeg to convert to WAV
        result = subprocess.run([
            'ffmpeg',
            '-i', input_path,
            '-acodec', 'pcm_s16le',  # 16-bit PCM
            '-ar', '16000',           # 16kHz sample rate
            '-ac', '1',               # Mono
            '-y',                     # Overwrite output
            output_path
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")
        
        # Read the converted WAV file
        with open(output_path, 'rb') as f:
            wav_bytes = f.read()
        
        return wav_bytes
        
    finally:
        # Clean up temporary files
        try:
            os.unlink(input_path)
        except Exception:
            pass
        try:
            os.unlink(output_path)
        except Exception:
            pass


def record_audio(seconds: float, sr: int, device=None) -> np.ndarray:
    if sd is None:
        raise RuntimeError("sounddevice/PortAudio not available in this environment")

    if device is not None:
        sd.default.device = (device, None)

    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="int16")
    sd.wait()
    return audio.reshape(-1)
