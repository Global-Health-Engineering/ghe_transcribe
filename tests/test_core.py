import glob
import os

import pytest

from ghe_transcribe.core import transcribe_core
from ghe_transcribe.exceptions import AudioConversionError, ModelInitializationError

TEST_AUDIO_PATH = "media/testing_audio.mp3"
huggingface_token = os.environ.get("HUGGINGFACE_TOKEN")


def test_transcribe_snippet():
    """Tests the transcribe function with a snippet and speaker count."""
    # Skip if test audio file doesn't exist
    if not os.path.exists(TEST_AUDIO_PATH):
        pytest.skip(f"Test audio file {TEST_AUDIO_PATH} not found")
    text = transcribe_core(
        TEST_AUDIO_PATH,
        trim=5,
        device="cpu",  # Force CPU to avoid MPS memory issues in CI
        cpu_threads=None,
        whisper_model="tiny.en",
        device_index=0,
        compute_type="int8",
        beam_size=5,
        temperature=0.0,
        word_timestamps=None,
        vad_filter=False,
        min_silence_duration_ms=2000,
        num_speakers=1,
        min_speakers=None,
        max_speakers=None,
        save_output=False,
        info=False,
        hf_token=huggingface_token,
    )

    # Add assertions to check the text
    assert isinstance(text, list), "Transcribe should return a list."
    assert len(text) > 0, "The text should not be empty."


def test_transcribe_invalid_file():
    """Test transcription with invalid file."""
    with pytest.raises(AudioConversionError):
        transcribe_core(
            "non_existent_file.mp3",
            whisper_model="tiny.en",
            save_output=False,
            info=False,
            hf_token=huggingface_token,
        )


def test_transcribe_invalid_device():
    """Test transcription with invalid device."""
    if not os.path.exists(TEST_AUDIO_PATH):
        pytest.skip(f"Test audio file {TEST_AUDIO_PATH} not found")

    with pytest.raises(ModelInitializationError):
        transcribe_core(
            TEST_AUDIO_PATH,
            device="invalid_device",
            whisper_model="tiny.en",
            save_output=False,
            info=False,
            hf_token=huggingface_token,
        )


def teardown_module():
    """Cleans up any .wav files created in the current directory."""
    for filename in glob.glob("media/*.wav"):
        try:
            os.remove(filename)
        except OSError:
            pass
