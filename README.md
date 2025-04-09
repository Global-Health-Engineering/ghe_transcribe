# ghe_transcribe: A Tool to Transcribe Audio Files with Speaker Diarization

This repository contains a Python script called `ghe_transcribe` that transcribes audio files into text using **Faster Whisper** (a fast reimplementation of OpenAI's Whisper model) and **Pyannote** (for speaker diarization). This tool is especially useful for transcribing long audio recordings, improving transcription accuracy, and separating the audio into individual speakers.

## Table of Contents
1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Example Usage](#example-usage)

## Requirements

This tool has been tested on Euler. To set up the environment, you will need to install the following dependencies:


### Euler Cluster

First time environment setup, first load system modules:
```bash
module load stack/2024-06 python/3.11.6
```
then create a Python environment and create a kernel:
```bash
python3.11 -m venv venv3.11_ghe_transcribe --system-site-packages
source venv3.11_ghe_transcribe/bin/activate
pip3.11 install faster-whisper pyannote.audio ffmpeg-python huggingface-hub
ipython kernel install --user --name=venv3.11_ghe_transcribe
```
setup JupyterHub starting configuration, open:

```bash
nano .config/euler/jupyterhub/jupyterlabrc
```
and paste
```bash
module load stack/2024-06 python_cuda/3.11.6
source venv3.11_ghe_transcribe/bin/activate
```
these commands will be run every session

For Mac OS users,
```bash
brew install ffmpeg cmake python3.12
```

## Usage

### Quick Start

Let's say you have an audio file called `audio.mp3` that you want to transcribe into a `.csv` and `.md` file. Open 

```bash
cd /path/to/ghe_transcribe
```

Then, run the following command:

```bash
python ghe_transcribe.py example/241118_1543.mp3 output.csv --device='cpu'
```

### Timing

Euler Cluster (12 CPU cores, 16GB RAM)
```
func:'transcribe' args:[('media/241118_1543.mp3',), {'device': 'cpu'}] took: 51.2057 sec
```

MacOS (Apple M2, 16GB RAM)
```
func:'transcribe' args:[('media/241118_1543.mp3',), {'device': 'mps'}] took: 41.2122 sec
```

MacOS (Apple M2, 16GB RAM)
```
func:'transcribe' args:[('media/241118_1543.mp3',), {'device': 'cpu'}] took: 64.7549 sec
```

### Options

Options for `ghe_transcribe`:

```python
ghe_transcribe(audio_file,
               output_file,
               device='cpu'|'cuda'|'mps',
               whisper_model='small.en'|'base.en'|'medium.en'|'small'|'base'|'medium'|'large'|'turbo',
               semicolon=True|False,
               info=True|False
)
```

- `audio_file`: The path to the audio file you want to transcribe. Accepted formats are .mp3, .wav.
- `output` (optional): Default is `True`. It will create both `output.csv` and `output.md`. If `output = None`, the transcription will only be returned as a list of strings.
- `device` (optional): The device on which to run the model (`cpu`|`cuda`|`mps`). By default, the device is automatically detected based on whether CUDA or MPS is available.
- `whisper_model` (optional): The size of the Faster Whisper model to use for transcription. Available options include `small`, `base`, `medium`, `large`, `turbo`. By default, the English model `medium.en` is used.
- `semicolon` (optional): Specify whether to use semicolons or commas as the column separator in the CSV output. The default is commas.
- `info` (optional): If you want the transcription tool to print additional information about the detected language and its probability.