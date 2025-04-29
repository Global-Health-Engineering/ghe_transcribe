import os

from ghe_transcribe.core import transcribe

# Assume your config.json is in the root of your project for testing
TEST_AUDIO_PATH = "media/testing_audio.mp3"

huggingface_token = os.environ.get("HUGGINGFACE_TOKEN")

def test_transcribe_snippet():
    """Tests the transcribe function with a snippet and speaker count."""
    result = transcribe(TEST_AUDIO_PATH, huggingface_token=huggingface_token, trim=5, num_speakers=1, save_output=False, info=False)

    # Add assertions to check the result
    assert isinstance(result, list), "Transcribe should return a list."
    assert len(result) > 0, "The result should not be empty."

# Clean up dummy files after testing (optional)
def teardown_module():
    if os.path.exists(os.path.splitext(TEST_AUDIO_PATH)[0]+".wav"):
        os.remove(os.path.splitext(TEST_AUDIO_PATH)[0]+".wav")
    if os.path.exists(os.path.splitext(TEST_AUDIO_PATH)[0] + "_snippet.wav"):
        os.remove(os.path.splitext(TEST_AUDIO_PATH)[0] + "_snippet.wav")