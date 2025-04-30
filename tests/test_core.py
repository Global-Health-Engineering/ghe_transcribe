import os

from ghe_transcribe.core import transcribe

TEST_AUDIO_PATH = "media/testing_audio.mp3"
huggingface_token = os.environ.get("HUGGINGFACE_TOKEN")

def test_transcribe_snippet():
    """Tests the transcribe function with a snippet and speaker count."""
    result = transcribe(TEST_AUDIO_PATH, 
                        huggingface_token=huggingface_token, 
                        trim=5, 
                        device="cpu", 
                        cpu_threads=None,
                        whisper_model="tiny.en",
                        device_index=0,
                        compute_type="int8",
                        beam_size=5,
                        temperature=0.0,
                        word_timestamps=None,
                        vad_filter=False,
                        min_silence_duration_ms=2000,
                        pyannote_model="pyannote/speaker-diarization-3.1",
                        num_speakers=1,
                        min_speakers=None,
                        max_speakers=None,
                        save_output=False, 
                        info=False)

    # Add assertions to check the result
    assert isinstance(result, list), "Transcribe should return a list."
    assert len(result) > 0, "The result should not be empty."

# Clean up dummy files after testing (optional)
def teardown_module():
    if os.path.exists(os.path.splitext(TEST_AUDIO_PATH)[0]+".wav"):
        os.remove(os.path.splitext(TEST_AUDIO_PATH)[0]+".wav")
    if os.path.exists(os.path.splitext(TEST_AUDIO_PATH)[0] + "_snippet.wav"):
        os.remove(os.path.splitext(TEST_AUDIO_PATH)[0] + "_snippet.wav")