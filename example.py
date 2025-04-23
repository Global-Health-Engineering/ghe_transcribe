import json
from huggingface_hub import login
from ghe_transcribe import transcribe

with open('config.json', 'r') as f:
    config = json.load(f)

login(token=config["HF_TOKEN"])

result = transcribe("media/testing_audio_01.mp3", 
                    device="cpu",
                    whisper_model='large-v3-turbo', 
                    device_index = 0,
                    compute_type = 'float32', 
                    cpu_threads = None,
                    beam_size = 5,
                    temperature = 0.0,
                    word_timestamps = False,
                    vad_filter = False,
                    vad_parameters = dict(min_silence_duration_ms=2000),
                    pyannote_model = 'pyannote/speaker-diarization-3.1',
                    num_speakers = 2,
                    min_speakers = None,
                    max_speakers = None,
                    save_output=True,
                    info=True)