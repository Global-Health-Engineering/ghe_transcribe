# ghe_transcribe: A Tool to Transcribe Audio Files with Speaker Diarization

This repository contains a Python script called `ghe_transcribe` that transcribes audio files into text using **Faster Whisper** (a fast reimplementation of OpenAI's Whisper model) and **Pyannote** (for speaker diarization). This tool is especially useful for transcribing long audio recordings, improving transcription accuracy, and separating the audio into individual speakers.

## Table of Contents
1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Example Usage](#example-usage)

## Requirements

This tool has been tested on Mac OS. To set up the environment, you will need to install the following dependencies:

### Mac OS system requirements
```bash
brew install ffmpeg cmake python3.12
```

### Python libraries
```bash
python3.12 -m venv venv3.12_ghe_transcribe
source venv3.12_ghe_transcribe/bin/activate
pip3.12 install faster-whisper pyannote.audio torch ipython ipywidgets ipykernel ffmpeg-python
ipython kernel install --user --name=venv3.12_ghe_transcribe
```

For Euler Cluster users, you will need to load the appropriate modules:
```bash
module load stack/2024-06 python/3.12.8
```

**Note:** Unlike the original `openai/whisper`, **Faster Whisper** does not require FFmpeg to be installed on the system separately, as it bundles the FFmpeg libraries via the `PyAV` package (which is a dependency of `faster-whisper`).

## Usage

To use `ghe_transcribe`, you can run the following command in your terminal:

```bash
python ghe_transcribe.py [audio_file] [output_file] [--device='cpu'|'cuda'|'mps'] [--whisper_model='small'|'base'|'medium'|'large'|'turbo'] [--semicolumn=True|False] [--info=True|False]
```

- `[audio_file]`: The path to the audio file you want to transcribe. Accepted formats are .mp3, .wav.
- `[output_file] (optional)`: If provided, the output will be saved as a CSV and Markdown file in the specified location. For example, if you provide `output.csv`, it will create both `output.csv` and `output.md`. If not provided, the transcription will only be returned as a list of strings.
- `--device` (optional): The device on which to run the model (`cpu`|`cuda`|`mps`). By default, the device is automatically detected based on whether CUDA or MPS is available.
- `--whisper_model` (optional): The size of the Faster Whisper model to use for transcription. Available options include `small`, `base`, `medium`, `large`, `turbo`. By default, the multilingual model `medium` is used. When English is detected `medium.en` is used.
- `--semicolumn` (optional): Specify whether to use semicolons or commas as the column separator in the CSV output. The default is commas.
- `--info` (optional): If you want the transcription tool to print additional information about the detected language and its probability, include the `--info=True` flag.

## Example Usage

Let's say you have an audio file called `example/241118_1543.mp3` that you want to transcribe into a CSV file called `output.csv`. First, navigate to the directory containing your script and the audio file:

```bash
cd /path/to/ghe_transcribe
```

Then, run the following command:

```bash
python ghe_transcribe.py example/241118_1543.mp3 output.csv --device='cpu'
```