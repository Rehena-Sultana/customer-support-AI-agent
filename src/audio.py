"""Optional audio transcription with faster-whisper."""
import tempfile
from pathlib import Path


def transcribe_audio(uploaded_file) -> str:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("Audio transcription is optional. Install it with: pip install faster-whisper") from exc

    suffix = Path(getattr(uploaded_file, "name", "audio.wav")).suffix or ".wav"
    data = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
        temp.write(data)
        audio_path = temp.name
    try:
        # CPU/int8 works without a CUDA Whisper setup. For GPU, change device/compute_type as appropriate.
        model = WhisperModel("small", device="cpu", compute_type="int8")
        segments, info = model.transcribe(audio_path, vad_filter=True)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        if not text:
            raise RuntimeError("No speech could be transcribed from this audio.")
        return text
    finally:
        Path(audio_path).unlink(missing_ok=True)
