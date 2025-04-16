# ghe_transcribe: A Tool to Transcribe Audio Files with Speaker Diarization

This repository contains a Python script called `ghe_transcribe` that transcribes audio files into text using **Faster Whisper** (a fast reimplementation of OpenAI's Whisper model) and **Pyannote** (for speaker diarization). This tool is especially useful for transcribing long audio recordings, improving transcription accuracy, and separating the audio into individual speakers.

## Table of Contents
1. [**Installation on Euler**](#installation-on-euler)
2. [**Installation on MacOS**](#installation-on-macos)
3. [**How to use `ghe_transcribe`**](#how-to-use-ghe_transcribe)
4. [**Tools for...**](#tools-for)

## Installation on Euler

### First login in Euler
[Here](https://scicomp.ethz.ch/wiki/Getting_started_with_clusters) for the official wiki. If you have never logged into Euler, follow this [great documentation](https://www.gdc-docs.ethz.ch/EulerManual/site/access/) provided by the Genetic Diversity Center (GDC) at ETH.

### Open a terminal in JupyterHub:
Open [https://jupyter.euler.hpc.ethz.ch/](https://jupyter.euler.hpc.ethz.ch/) and login with your @ethz.ch account. Then, click on terminal.

### Load modules
We can load the modules we need by running
```bash
module load stack/2024-06 python/3.11.6
```
### Create a Python environment and create a kernel:
```bash
python3.11 -m venv venv3.11_ghe_transcribe --system-site-packages
source venv3.11_ghe_transcribe/bin/activate
pip3.11 install faster-whisper pyannote.audio ffmpeg-python huggingface-hub
ipython kernel install --user --name=venv3.11_ghe_transcribe
```
### Setup JupyterHub starting configuration:
To have all new JupyterHub instanced with the `venv3.11_ghe_transcribe` Python environment,
```bash
nano .config/euler/jupyterhub/jupyterlabrc
```
and write:
```bash
module load stack/2024-06 python/3.11.6
source venv3.11_ghe_transcribe/bin/activate
```

## Installation on MacOS
```bash
brew install ffmpeg cmake python3.11
```

```bash
python3.11 -m venv venv3.11_ghe_transcribe --system-site-packages
source venv3.11_ghe_transcribe/bin/activate
pip3.11 install faster-whisper pyannote.audio huggingface-hub
ipython kernel install --user --name=venv3.11_ghe_transcribe
```

## How to use `ghe_transcribe`

### Quick Start

Let's say you have an audio file called `testing_audio_01.mp3`, in the `media` folder, that you want to transcribe into a `.csv` and `.md` file. 
- setup `config.json` file
```json
{
    "HF_TOKEN": "hf_*********************"
}
```
- run the following command:
```bash
python ghe_transcribe.py media/testing_audio_01.mp3
```

### Options

Options for `ghe_transcribe`:

```python
ghe_transcribe(audio_file,
               device='cpu'|'cuda'|'mps',
               whisper_model='small.en'|'base.en'|'medium.en'|'small'|'base'|'medium'|'large'|'turbo',
               pyannote_model='pyannote/speaker-diarization@2.1'|'pyannote/speaker-diarization-3.1',
               save_output=True|False,
               semicolon=True|False,
               info=True|False
)
```

- `audio_file`: The path to the audio file you want to transcribe. Accepted formats are .mp3, .wav.
- `device` (optional): The device on which to run the model (`cpu`|`cuda`|`mps`). By default, the device is automatically detected based on whether CUDA or MPS is available.
- `whisper_model` (optional): The size of the Faster Whisper model to use for transcription. Available options are:
    ```python
    _MODELS = {
        "tiny.en": "Systran/faster-whisper-tiny.en",
        "tiny": "Systran/faster-whisper-tiny",
        "base.en": "Systran/faster-whisper-base.en",
        "base": "Systran/faster-whisper-base",
        "small.en": "Systran/faster-whisper-small.en",
        "small": "Systran/faster-whisper-small",
        "medium.en": "Systran/faster-whisper-medium.en",
        "medium": "Systran/faster-whisper-medium",
        "large-v1": "Systran/faster-whisper-large-v1",
        "large-v2": "Systran/faster-whisper-large-v2",
        "large-v3": "Systran/faster-whisper-large-v3",
        "large": "Systran/faster-whisper-large-v3",
        "distil-large-v2": "Systran/faster-distil-whisper-large-v2",
        "distil-medium.en": "Systran/faster-distil-whisper-medium.en",
        "distil-small.en": "Systran/faster-distil-whisper-small.en",
        "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
        "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    }
    ```
    By default, `large-v3-turbo` is used.
- `pyannote_model` (optional): The Pyannote model, defaults to `pyannote/speaker-diarization-3.1`.
- `save_output` (optional): Default is `True`. It will create both `output.csv` and `output.md`. If `output = None`, the transcription will only be returned as a list of strings.
- `semicolon` (optional): Specify whether to use semicolons or commas as the column separator in the CSV output. The default is commas.
- `info` (optional): If you want the transcription tool to print additional information about the detected language and its probability.

### Timings

Timing tests are run by using the timing function as defined in [`utils.py`](https://github.com/Global-Health-Engineering/ghe_transcribe/blob/main/utils.py), and the audio file [`media/testing_audio_01.mp3`](https://github.com/Global-Health-Engineering/ghe_transcribe/blob/main/media/testing_audio_01.mp3)

| Device     | Time (sec) |
|-------------------|------------------|
| Euler Cluster (16 CPU cores, 16GB RAM) - `cpu`  | 67.4988 |
| Euler Cluster (32 CPU cores, 16GB RAM) - `cpu`  | 44.3622 |
| MacOS (Apple M2, 16GB RAM) - `mps`  | 41.2122 |
| MacOS (Apple M2, 16GB RAM) - `cpu`  | 64.7549 |

## Tools for...

### Transcription
Why Whisper? See [Whisper, wav2vec2 and Kaldi](https://deepgram.com/learn/benchmarking-top-open-source-speech-models).
- [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) by Guillaume Klein, builds on OpenAI's open source transcription model [`Whisper`](https://github.com/openai/whisper).

### Diarization
Why Pyannote? See [Pyannote vs NeMo](https://lajavaness.medium.com/comparing-state-of-the-art-speaker-diarization-frameworks-pyannote-vs-nemo-31a191c6300).
- [`pyannote.audio`](https://github.com/pyannote/pyannote-audio) by Hervé Bredin, open source diarization model of [pyannoteAI](https://www.pyannote.ai/), gated by HuggingFace access token [https://hf.co/settings/tokens](https://hf.co/settings/tokens).
- [`NeMo`](https://github.com/NVIDIA/NeMo) by Nvidia, open source diarization model.

### Transcription + Diarization 
- [`WhisperX`](https://github.com/m-bain/whisperX) &larr; `faster-whisper`+`pyannote.audio`
- [`whisper-diarization`](https://github.com/MahmoudAshraf97/whisper-diarization) &larr; `faster-whisper`+`NeMo`
- [`insanely-fast-whisper`](https://github.com/Vaibhavs10/insanely-fast-whisper) &larr; `insanely-faster-whisper`+`pyannote.audio`

### GUI
- [`wscribe-editor`](https://github.com/geekodour/wscribe-editor), works with wordlevel timestamps in a .json formatted like so [sample.json](https://github.com/geekodour/wscribe/blob/main/examples/output/sample.json).
- [`QualCoder`](https://github.com/ccbogel/QualCoder), a qualitative data analysis application written in Python.

### Transcription + Diarization + GUI
- [`noScribe`](https://github.com/kaixxx/noScribe) &larr; `faster-whisper`+`pyannote.audio`
- [`TranscriboZH`](https://github.com/machinelearningZH/audio-transcription) &larr; `WhisperX`