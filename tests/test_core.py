import os
import glob

from ghe_transcribe.core import transcribe

TEST_AUDIO_PATH = "media/testing_audio.mp3"
# huggingface_token = os.environ.get("HUGGINGFACE_TOKEN")

def test_transcribe_snippet():
    """Tests the transcribe function with a snippet and speaker count."""
    text = transcribe(TEST_AUDIO_PATH, 
                        trim=5, 
                        device="auto", 
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
                        info=False)

    # Add assertions to check the text
    assert isinstance(text, list), "Transcribe should return a list."
    assert len(text) > 0, "The text should not be empty."

def teardown_module():
    """Cleans up any .wav files created in the current directory."""
    for filename in glob.glob("media/*.wav"):
        try:
            os.remove(filename)
            print(f"Removed: {filename}")
        except OSError as e:
            print(f"Error removing {filename}: {e}")