"""Voice layer — interfaces first, wiring later (Phase 4).

Free/local defaults, matched to their paid equivalents named in the JD:
    STT: faster-whisper   (local)  <->  Deepgram        (hosted)
    TTS: Kokoro / Piper   (local)  <->  ElevenLabs/Cartesia (hosted)
Real-time transport: LiveKit (open-source, self-hostable) — the exact tool named
in the posting, so use the real thing rather than a substitute.

Define the interface now so the agent doesn't care which implementation runs;
you swap local <-> hosted with config, same as the LLM and vector store.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class SpeechToText(ABC):
    @abstractmethod
    def transcribe(self, audio_bytes: bytes) -> str: ...


class TextToSpeech(ABC):
    @abstractmethod
    def synthesize(self, text: str) -> bytes: ...


class FasterWhisperSTT(SpeechToText):
    """Local STT. `pip install faster-whisper`; runs on CPU for small models."""

    def __init__(self, model_size: str = "base.en") -> None:
        self.model_size = model_size
        self._model = None

    def transcribe(self, audio_bytes: bytes) -> str:  # pragma: no cover - Phase 4
        raise NotImplementedError(
            "Phase 4: load WhisperModel(self.model_size), write audio_bytes to a "
            "temp wav, and return the joined segment text."
        )


class KokoroTTS(TextToSpeech):
    """Local TTS with natural-sounding output. Piper is the lighter alternative."""

    def synthesize(self, text: str) -> bytes:  # pragma: no cover - Phase 4
        raise NotImplementedError("Phase 4: return synthesized audio bytes.")


# Phase 4 integration note:
# LiveKit Agents provides a VoicePipelineAgent that chains STT -> LLM -> TTS with
# turn detection and interruption handling. Plug your agent graph in as the "LLM"
# step so the same reasoning powers both text and voice.
