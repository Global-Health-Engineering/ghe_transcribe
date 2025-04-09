import json
from huggingface_hub import login

with open('config.json', 'r') as f:
    config = json.load(f)

login(token=config["HF_TOKEN"])

from ghe_transcribe import transcribe

result = transcribe("media/241118_1543.mp3")