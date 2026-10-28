"""Optional audio transcription with faster-whisper with in-memory caching."""
import tempfile
from pathlib import Path
from functools import lru_cache


@lru_cache(maxsize=1)
def get_whisper_model():
    """Load Whisper once and keep it cached in memory for sub-second transcriptions."""
    from faster_whisper import WhisperModel
    # 'base' is fast (<1s transcription) and highly accurate for English customer support.
    try:
        return WhisperModel("base", device="cpu", compute_type="int8")
    except Exception:
        # Fallback to small if available
        return WhisperModel("small", device="cpu", compute_type="int8")


def transcribe_audio(uploaded_file) -> str:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("Audio transcription is optional. Install it with: pip install faster-whisper") from exc

    suffix = Path(getattr(uploaded_file, "name", "audio.wav")).suffix or ".wav"
    
    # Ensure uploaded file pointer is at start and read bytes safely
    if hasattr(uploaded_file, "seek"):
        try:
            uploaded_file.seek(0)
        except Exception:
            pass

    data = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    if not data or len(data) == 0:
        raise RuntimeError("The uploaded audio file is empty (0 bytes).")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
        temp.write(data)
        temp.flush()
        audio_path = temp.name

    try:
        model = get_whisper_model()
        segments, info = model.transcribe(audio_path, vad_filter=True)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        if not text:
            raise RuntimeError("No speech could be transcribed from this audio.")
        return text
    finally:
        Path(audio_path).unlink(missing_ok=True)
