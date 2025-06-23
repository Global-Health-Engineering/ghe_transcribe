"""Custom exceptions for ghe_transcribe."""


class TranscriptionError(Exception):
    """Base exception for transcription errors."""

    pass


class AudioConversionError(TranscriptionError):
    """Raised when audio file conversion fails."""

    pass


class ModelInitializationError(TranscriptionError):
    """Raised when model initialization fails."""

    pass


class DiarizationError(TranscriptionError):
    """Raised when speaker diarization fails."""

    pass
