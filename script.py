import json

from huggingface_hub import login

from ghe_transcribe.core import transcribe

with open("config.json", "r") as f:
    config = json.load(f)

login(token=config["HF_TOKEN"])

result = transcribe("media/testing_audio_01.mp3", num_speakers=2)
