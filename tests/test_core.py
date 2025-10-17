import glob
import os

import pytest

from ghe_transcribe.core import transcribe
from ghe_transcribe.exceptions import AudioConversionError, ModelInitializationError

TEST01 = "media/test01.mp3"
TEST02 = "media/test02.m4a"
huggingface_token = os.environ.get("HUGGINGFACE_TOKEN")

def test_transcribe_invalid_file():
    """Test transcription with invalid file."""
    with pytest.raises(AudioConversionError):
        transcribe(
            "non_existent_file.mp3",
            whisper_model="tiny.en",
            save_output=False,
            info=False,
            hf_token=huggingface_token,
        )


def test_transcribe_invalid_device():
    """Test transcription with invalid device."""
    if not os.path.exists(TEST01):
        pytest.skip(f"Test audio file {TEST01} not found")

    with pytest.raises(ModelInitializationError):
        transcribe(
            TEST01,
            device="invalid_device",
            whisper_model="tiny.en",
            save_output=False,
            info=False,
            hf_token=huggingface_token,
        )


def test_transcribe_two_files():
    """Test the transcribe function with two files."""
    test_files = [TEST01, TEST02]

    # Skip if test audio files don't exist
    for file_path in test_files:
        if not os.path.exists(file_path):
            pytest.skip(f"Test audio file {file_path} not found")

    results = transcribe(
        files=test_files,
        trim=5,
        device="cpu",
        whisper_model="tiny.en",
        compute_type="int8",
        num_speakers=1,
        save_output=False,
        info=False,
        hf_token=huggingface_token,
    )

    assert isinstance(results, dict), "Multiple files should return a dict."
    assert len(results) == 2, "Should have results for both files."
    for file_path in test_files:
        assert file_path in results, f"Should have result for {file_path}"


def teardown_module():
    """Cleans up any .wav files created in the current directory."""
    for filename in glob.glob("media/*.wav"):
        try:
            os.remove(filename)
        except OSError:
            pass
