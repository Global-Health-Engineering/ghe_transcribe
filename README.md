# ghe_transcribe: A Tool to Transcribe Audio Files with Speaker Diarization

[![Python Versions](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/downloads/)
[![Python application](https://github.com/Global-Health-Engineering/ghe_transcribe/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/Global-Health-Engineering/ghe_transcribe/actions/workflows/python-app.yml)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](https://github.com/Global-Health-Engineering/ghe_transcribe/blob/main/CONTRIBUTING.md)

This repository hosts `ghe_transcribe`, a powerful Python script designed to transcribe audio files and perform speaker diarization. It leverages the speed and accuracy of **Faster Whisper** (a highly optimized reimplementation of OpenAI's Whisper) for transcription and **Pyannote** for identifying and separating speakers within the audio. This tool is ideal for handling long recordings, enhancing transcription quality, and automatically segmenting audio by speaker.

## Table of Contents

1.  [**Installation**](#installation)
      * [macOS](#macos)
      * [Euler Cluster](#euler-cluster)
2.  [**Usage**](#usage)
      * [Quickstart](#quickstart)
      * [Python Integration](#python-integration)
      * [Command-Line Interface (CLI)](#command-line-interface-cli)
4.  [**Contributing**](#contributing)
5.  [**License**](#license)

## Installation

Choose the installation method that suits your environment.

### macOS

Run the following commands to install on macOS:

```bash
python -m venv venv_ghe_transcribe
source venv_ghe_transcribe/bin/activate
git clone https://github.com/Global-Health-Engineering/ghe_transcribe.git
cd ghe_transcribe
pip install -r requirements.txt
pip install -e .
ipython kernel install --user --name=venv_ghe_transcribe
```

### Euler Cluster

Follow these steps to set up `ghe_transcribe` on the Euler cluster at ETH Zurich.

#### First login to Euler

> [!IMPORTANT]
> Refer to the official [Euler wiki on getting started](https://scicomp.ethz.ch/wiki/Getting_started_with_clusters) if you are a first-time user.

#### Open a terminal in JupyterHub

1.  Navigate to [https://jupyter.euler.hpc.ethz.ch/](https://jupyter.euler.hpc.ethz.ch/) and log in with your ETHZ account.
2.  Click on "Terminal" in the JupyterLab interface.

#### Load necessary modules

Execute the following command to load the required software modules:

```bash
module load stack/2024-06 python_cuda/3.11.6
```

#### Create a Python virtual environment and kernel

It's recommended to create a dedicated virtual environment to manage dependencies:

```bash
python -m venv venv_ghe_transcribe --system-site-packages
source venv_ghe_transcribe/bin/activate
git clone https://github.com/Global-Health-Engineering/ghe_transcribe.git
cd ghe_transcribe
pip install -e .
ipython kernel install --user --name=venv_ghe_transcribe
```

#### Configure JupyterHub to use the environment

To ensure your JupyterHub instances automatically use the created environment, edit the JupyterLab configuration file:

```bash
echo "module load stack/2024-06 python_cuda/3.11.6 && source venv_ghe_transcribe/bin/activate" >> ~/.config/euler/jupyterhub/jupyterlabrc
```

## Usage

To transcribe an audio file:

### Quickstart
For the quickest start, use the user interface (UI) tool, `app.ipynb`.

1. Open `app.ipynb`
2. Select kernel `venv_ghe_transcribe`
3. Run the cell

### Python Integration

```python
from ghe_transcribe.core import transcribe

result = transcribe("media/YOUR_AUDIO_FILE.mp3")
```

### Command-Line Interface (CLI)

1.  **Place your audio file:** Upload the audio file you want to transcribe. 
> [!TIP]
> Drop the file into the `media` folder.
2.  **Run the transcription script:** Execute the `transcribe` command in the terminal:
    ```bash
    transcribe media/YOUR_AUDIO_FILE.mp3
    ```
> [!IMPORTANT]
> This will work only if you are in the `ghe_transcribe` directory. Otherwise, change path to the correct `path/to/YOUR_AUDIO_FILE.mp3`.

> **Usage**:
> 
> ```console
> $ transcribe [OPTIONS] FILE
> ```
> 
> **Arguments**:
> 
> * `FILE`: Path to the audio file.  [required]
> 
> **Options**:
> 
> * `--trim FLOAT`: Trim the audio file from 0 to the specified number of seconds.
> * `--device [auto|cuda|mps|cpu]`: Device to use.  [default: auto]
> * `--cpu-threads INTEGER`: Number of CPU threads to use.
> * `--whisper-model [tiny.en|tiny|base.en|base|small.en|small|medium.en|medium|large-v1|large-v2|large-v3|large|> distil-large-v2|distil-medium.en|distil-small.en|distil-large-v3|large-v3-turbo|turbo]`: Faster Whisper, model to > use.  [default: large-v3-turbo]
> * `--device-index INTEGER`: Faster Whisper, index of the device to use.  [default: 0]
> * `--compute-type [float32|float16|int8]`: Faster Whisper, compute type.  [default: float32]
> * `--beam-size INTEGER`: Faster Whisper, beam size for decoding.  [default: 5]
> * `--temperature FLOAT`: Faster Whisper, sampling temperature.  [default: 0.0]
> * `--word-timestamps / --no-word-timestamps`: Faster Whisper, enable word timestamps in the output.
> * `--vad-filter / --no-vad-filter`: Faster Whisper, enable voice activity detection.  [default: no-vad-filter]
> * `--min-silence-duration-ms INTEGER`: Faster Whisper, minimum silence duration detected by VAD in milliseconds.  > [default: 2000]
> * `--num-speakers INTEGER`: pyannote.audio, number of speakers.
> * `--min-speakers INTEGER`: pyannote.audio, minimum number of speakers.
> * `--max-speakers INTEGER`: pyannote.audio, maximum number of speakers.
> * `--save-output / --no-save-output`: Save output to .csv and .md files.  [default: save-output]
> * `--info / --no-info`: Print detected language information.  [default: info]
> * `--install-completion`: Install completion for the current shell.
> * `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
> * `--help`: Show this message and exit.

## Contributing

Contributions to `ghe_transcribe` are welcome! 

> [!NOTE]
> Please follow these guidelines:
> ### [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)
> Commits are structure like so, `<type>: <description>`. For example, `fix: typo in README.md`. 

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).