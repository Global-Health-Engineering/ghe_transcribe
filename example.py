import json
from huggingface_hub import login
from ghe_transcribe import transcribe

with open('config.json', 'r') as f:
    config = json.load(f)

login(token=config["HF_TOKEN"])

result = transcribe("media/241118_1543.mp3", device="cpu")