# ghe_transcribe: A Tool to Transcribe Audio Files with Speaker Diarization

This repository contains a Python script called `ghe_transcribe` that transcribes audio files into text using **Faster Whisper** (a fast reimplementation of OpenAI's Whisper model) and **Pyannote** (for speaker diarization). This tool is especially useful for transcribing long audio recordings, improving transcription accuracy, and separating the audio into individual speakers.

## Table of Contents
1. [Requirements](#requirements)
3. [Usage](#usage)
4. [Research](#research)

## Requirements

This tool has been tested on Euler. To set up the environment, you will need to install the following dependencies:


### Euler Cluster

#### First time environment setup:
Open [https://jupyter.euler.hpc.ethz.ch/](https://jupyter.euler.hpc.ethz.ch/) and login with your @ethz.ch account. We can use the default modules loaded in the JupyterHub instance, running `module list` on a terminal should return:
```bash
1) stack/2024-05   2) gcc/13.2.0   3) cuda/12.2.1   4) python/3.11.6_cuda   5) eth_proxy   6) r/4.3.2   7) hdf5/1.14.3   8) julia/1.10.3
```
the modules `stack/2024-05  gcc/13.2.0  cuda/12.2.1 python/3.11.6_cuda` are what we are interested in.
#### Create a Python environment and create a kernel:
```bash
python3.11 -m venv venv3.11_ghe_transcribe --system-site-packages
source venv3.11_ghe_transcribe/bin/activate
pip3.11 install faster-whisper pyannote.audio ffmpeg-python huggingface-hub
ipython kernel install --user --name=venv3.11_ghe_transcribe
```
#### Setup JupyterHub starting configuration:
To have all new JupyterHub instanced with the `venv3.11_ghe_transcribe` Python environment,
```bash
nano .config/euler/jupyterhub/jupyterlabrc
```
and write:
```bash
source venv3.11_ghe_transcribe/bin/activate
```

### Mac OS
```bash
brew install ffmpeg cmake python3.11
```

```bash
python3.11 -m venv venv3.11_ghe_transcribe --system-site-packages
source venv3.11_ghe_transcribe/bin/activate
pip3.11 install faster-whisper pyannote.audio ffmpeg-python huggingface-hub
ipython kernel install --user --name=venv3.11_ghe_transcribe
```

## Usage

### Quick Start

Let's say you have an audio file called `testing_audio_01.mp3`, in the `media` folder, that you want to transcribe into a `.csv` and `.md` file. Then, run the following command:
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
- `whisper_model` (optional): The size of the Faster Whisper model to use for transcription. Available options include `small.en`, `base.en`, `medium.en`, `small`, `base`, `medium`, `large`, `turbo`. By default, the English model `medium.en` is used.
- `pyannote_model` (optional): The Pyannote model, defaults to `pyannote/speaker-diarization-3.1`.
- `save_output` (optional): Default is `True`. It will create both `output.csv` and `output.md`. If `output = None`, the transcription will only be returned as a list of strings.
- `semicolon` (optional): Specify whether to use semicolons or commas as the column separator in the CSV output. The default is commas.
- `info` (optional): If you want the transcription tool to print additional information about the detected language and its probability.

### Timings

Euler Cluster (16 CPU cores, 16GB RAM)
- `func:'transcribe' args:[('media/241118_1543.mp3',), {'device': 'cpu'}] took: 67.4988 sec`

Euler Cluster (32 CPU cores, 16GB RAM)
- `func:'transcribe' args:[('media/241118_1543.mp3',), {'device': 'cpu'}] took: 44.3622 sec`

MacOS (Apple M2, 16GB RAM)
- `func:'transcribe' args:[('media/241118_1543.mp3',), {'device': 'mps'}] took: 41.2122 sec`

MacOS (Apple M2, 16GB RAM)
- `func:'transcribe' args:[('media/241118_1543.mp3',), {'device': 'cpu'}] took: 64.7549 sec`

## Research

### Transcription
- [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) by Guillaume Klein, builds on OpenAI's open source transcription model [`Whisper`](https://github.com/openai/whisper).

[comparison](https://deepgram.com/learn/benchmarking-top-open-source-speech-models)

### Diarization

- [`pyannote.audio`](https://github.com/pyannote/pyannote-audio) by Hervé Bredin, open source diarization model of [pyannoteAI](https://www.pyannote.ai/), gated by HuggingFace access token [https://hf.co/settings/tokens](https://hf.co/settings/tokens).
- [`NeMo`](https://github.com/NVIDIA/NeMo) by Nvidia, open source diarization model.

[comparison](https://lajavaness.medium.com/comparing-state-of-the-art-speaker-diarization-frameworks-pyannote-vs-nemo-31a191c6300)

### Transcription + Diarization
- [`WhisperX`](https://github.com/m-bain/whisperX) &larr; `faster-whisper`+`pyannote.audio`
- [`whisper-diarization`](https://github.com/MahmoudAshraf97/whisper-diarization) &larr; `faster-whisper`+`NeMo`
- [`insanely-fast-whisper`](https://github.com/Vaibhavs10/insanely-fast-whisper) &larr; `insanely-faster-whisper`+`pyannote.audio`

### Editors
- [`wscribe-editor`](https://github.com/geekodour/wscribe-editor), works with wordlevel timestamps in a .json formatted like so [sample.json](https://github.com/geekodour/wscribe/blob/main/examples/output/sample.json).
- [`QualCoder`](https://github.com/ccbogel/QualCoder), a qualitative data analysis application written in Python.

### Existing Tools
- [`noScribe`](https://github.com/kaixxx/noScribe) &larr; `faster-whisper`+`pyannote.audio`
- [`TranscriboZH`](https://github.com/machinelearningZH/audio-transcription) &larr; `WhisperX`