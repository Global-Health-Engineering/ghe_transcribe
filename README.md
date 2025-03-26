# ghe_transcribe

## Mac OS system requirements
````bash
brew install ffmpeg cmake python3.12
````

## Euler cluster requirements
````bash
module load stack/2024-06 python/3.12.8
````

## Python libraries
````bash
python3.12 -m venv venv_ghe_transcribe --system-site-packages
source venv_ghe_transcribe/bin/activate
pip3.12 install huggingface_hub pyannote.audio torch faster-whisper ipython ipykernel ffmpeg-python
ipython kernel install --user --name=venv_ghe_transcribe
````
