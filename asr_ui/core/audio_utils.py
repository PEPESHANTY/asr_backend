import io
import numpy as np
from pathlib import Path
import wave

try:
    import sounddevice as sd
except Exception:
    sd = None


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


def record_audio(seconds: float, sr: int, device=None) -> np.ndarray:
    if sd is None:
        raise RuntimeError("sounddevice/PortAudio not available in this environment")

    if device is not None:
        sd.default.device = (device, None)

    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="int16")
    sd.wait()
    return audio.reshape(-1)
