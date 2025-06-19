# ghe_transcribe: A Tool to Transcribe Audio Files with Speaker Diarization

[![Python Versions](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/downloads/)
[![Python application](https://github.com/Global-Health-Engineering/ghe_transcribe/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/Global-Health-Engineering/ghe_transcribe/actions/workflows/python-app.yml)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](https://github.com/Global-Health-Engineering/ghe_transcribe/blob/main/CONTRIBUTING.md)

This repository hosts `ghe_transcribe`, a powerful Python script designed to transcribe audio files and perform speaker diarization. It leverages the speed and accuracy of **Faster Whisper** (a highly optimized reimplementation of OpenAI's Whisper) for transcription and **Pyannote** for identifying and separating speakers within the audio. This tool is ideal for handling long recordings, enhancing transcription quality, and automatically segmenting audio by speaker.

## Table of Contents

1.  [**Getting Started**](#getting-started)
    * [Installation](#installation)
        * [Euler Cluster](#euler-cluster)
        * [macOS](#macos)
    * [Basic Usage](#basic-usage)
        * [Quick Start](#quick-start)
        * [Python Integration](#python-integration)
2.  [**Command-Line Interface**](#command-line-interface)
    * [Options](#options)
3.  [**Performance Benchmarks**](#performance-benchmarks)
    * [Execution Time](#execution-time)
4.  [**Related Projects**](#related-projects)
    * [Transcription Libraries](#transcription-libraries)
    * [Speaker Diarization Libraries](#speaker-diarization-libraries)
    * [Combined Transcription and Diarization Tools](#combined-transcription-and-diarization-tools)
    * [Graphical User Interfaces](#graphical-user-interfaces)
    * [GUI Applications with Transcription and Diarization](#gui-applications-with-transcription-and-diarization)
5.  [**Contributing**](#contributing)
6.  [**License**](#license)

## Getting Started

### Installation

Choose the installation method that suits your environment.

#### Euler Cluster

Follow these steps to set up `ghe_transcribe` on the Euler cluster at ETH Zurich.

##### First login to Euler

Refer to the official [Euler wiki on getting started](https://scicomp.ethz.ch/wiki/Getting_started_with_clusters) if you are a first-time user.

##### Open a terminal in JupyterHub

1.  Navigate to [https://jupyter.euler.hpc.ethz.ch/](https://jupyter.euler.hpc.ethz.ch/) and log in with your ETHZ account.
2.  Click on "Terminal" in the JupyterLab interface.

##### Load necessary modules

Execute the following command to load the required software modules:

```bash
module load stack/2024-06 python_cuda/3.11.6
```

##### Create a Python virtual environment and kernel

It's recommended to create a dedicated virtual environment to manage dependencies:

```bash
python -m venv venv_ghe_transcribe --system-site-packages
source venv_ghe_transcribe/bin/activate
git clone https://github.com/Global-Health-Engineering/ghe_transcribe.git
cd ghe_transcribe
pip install -r requirements.txt
pip install -e .
ipython kernel install --user --name=venv_ghe_transcribe
```

##### Configure JupyterHub to use the environment

To ensure your JupyterHub instances automatically use the created environment, edit the JupyterLab configuration file:

```bash
echo "module load stack/2024-06 python_cuda/3.11.6 && source venv_ghe_transcribe/bin/activate" >> .config/euler/jupyterhub/jupyterlabrc
```

#### macOS

Use the following commands to install the necessary dependencies on macOS:

```bash
python -m venv venv_ghe_transcribe
source venv_ghe_transcribe/bin/activate
git clone https://github.com/Global-Health-Engineering/ghe_transcribe.git
cd ghe_transcribe
pip install -r requirements.txt
pip install -e .
```

### Basic Usage

#### Quick Start

To transcribe an audio file:

1.  **Place your audio file:** Upload the audio file you want to transcribe, e.g.`testing_audio.mp3`. **(Recommended)** Drop the file into the `media` folder.
2.  **Run the transcription script:** Execute the transcribe command in the terminal:
    ```bash
    transcribe media/testing_audio.mp3
    ```
    * **(!)** Make sure you are in the correct directory, `ghe_transcribe`. Otherwise change path to the correct `path/to/your/audio/file.mp3`.

#### Python Integration
**Example:**
```python
from ghe_transcribe.core import transcribe

result = transcribe("media/testing_audio.mp3")
```

## Command-Line Interface

### Options

Transcribe and diarize an audio file.

**Usage**:

```console
$ transcribe [OPTIONS] FILE
```

**Arguments**:

* `FILE`: Path to the audio file.  [required]

**Options**:

* `--huggingface-token TEXT`: Hugging Face token for authentication.
* `--trim FLOAT`: Trim the audio file from 0 to the specified number of seconds.
* `--device [auto|cuda|mps|cpu]`: Device to use.  [default: auto]
* `--cpu-threads INTEGER`: Number of CPU threads to use.
* `--whisper-model [tiny.en|tiny|base.en|base|small.en|small|medium.en|medium|large-v1|large-v2|large-v3|large|distil-large-v2|distil-medium.en|distil-small.en|distil-large-v3|large-v3-turbo|turbo]`: Faster Whisper, model to use.  [default: large-v3-turbo]
* `--device-index INTEGER`: Faster Whisper, index of the device to use.  [default: 0]
* `--compute-type [float32|float16|int8]`: Faster Whisper, compute type.  [default: float32]
* `--beam-size INTEGER`: Faster Whisper, beam size for decoding.  [default: 5]
* `--temperature FLOAT`: Faster Whisper, sampling temperature.  [default: 0.0]
* `--word-timestamps / --no-word-timestamps`: Faster Whisper, enable word timestamps in the output.
* `--vad-filter / --no-vad-filter`: Faster Whisper, enable voice activity detection.  [default: no-vad-filter]
* `--min-silence-duration-ms INTEGER`: Faster Whisper, minimum silence duration detected by VAD in milliseconds.  [default: 2000]
* `--pyannote-model TEXT`: pyannote.audio, speaker diarization model to use.  [default: pyannote/speaker-diarization-3.1]
* `--num-speakers INTEGER`: pyannote.audio, number of speakers.
* `--min-speakers INTEGER`: pyannote.audio, minimum number of speakers.
* `--max-speakers INTEGER`: pyannote.audio, maximum number of speakers.
* `--save-output / --no-save-output`: Save output to .csv and .md files.  [default: save-output]
* `--info / --no-info`: Print detected language information.  [default: info]
* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.

## Performance Benchmarks

### Execution Time

The following table shows the execution time of `transcribe media/testing_audio.mp3` across different environments. These tests were conducted using the `timing` function defined in `utils.py`.

| Device                                      | Time (sec) |
| :------------------------------------------ | :--------- |
| Euler Cluster (16 CPU cores, 16GB RAM) - `cpu` | 67.4988    |
| Euler Cluster (32 CPU cores, 16GB RAM) - `cpu` | 44.3622    |
| MacOS (Apple M2, 16GB RAM) - `mps`           | 41.2122    |
| MacOS (Apple M2, 16GB RAM) - `cpu`           | 64.7549    |

## Related Projects

Explore these related projects and libraries for more advanced functionalities or alternative approaches.

### Transcription Libraries

  - **Whisper:** ([https://github.com/openai/whisper](https://github.com/openai/whisper)) - OpenAI's original speech-to-text model.
  - **faster-whisper:** ([https://github.com/SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper)) - The fast implementation used in `ghe_transcribe`, offering significant speed improvements. See [benchmarks](https://github.com/SYSTRAN/faster-whisper/issues/1030).
  - **Deepgram Benchmarks:** ([https://deepgram.com/learn/benchmarking-top-open-source-speech-models](https://deepgram.com/learn/benchmarking-top-open-source-speech-models)) - A comparison of various open-source speech models, including Whisper.

### Speaker Diarization Libraries

  - **pyannote.audio:** ([https://github.com/pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio)) - The speaker diarization toolkit by pyannoteAI ([https://www.pyannote.ai/](https://www.pyannote.ai/)), requiring a Hugging Face access token ([https://hf.co/settings/tokens](https://hf.co/settings/tokens)).
  - **NeMo:** ([https://github.com/NVIDIA/NeMo](https://github.com/NVIDIA/NeMo)) - NVIDIA's open-source framework for conversational AI, including diarization capabilities.
  - **Pyannote vs NeMo Comparison:** ([https://lajavaness.medium.com/comparing-state-of-the-art-speaker-diarization-frameworks-pyannote-vs-nemo-31a191c6300](https://lajavaness.medium.com/comparing-state-of-the-art-speaker-diarization-frameworks-pyannote-vs-nemo-31a191c6300)) - A blog post comparing the two frameworks.

### Combined Transcription and Diarization Tools

  - **WhisperX:** ([https://github.com/m-bain/whisperX](https://github.com/m-bain/whisperX)) - Combines `faster-whisper` with `pyannote.audio` for transcription and diarization.
  - **whisper-diarization:** ([https://github.com/MahmoudAshraf97/whisper-diarization](https://github.com/MahmoudAshraf97/whisper-diarization)) - Integrates `faster-whisper` with `NeMo` for speaker diarization.
  - **insanely-fast-whisper:** ([https://github.com/Vaibhavs10/insanely-fast-whisper](https://github.com/Vaibhavs10/insanely-fast-whisper)) - Another optimized Whisper implementation combined with `pyannote.audio`.

### Graphical User Interfaces

  - **wscribe-editor:** ([https://github.com/geekodour/wscribe-editor](https://github.com/geekodour/wscribe-editor)) - An editor for transcriptions with word-level timestamps, supporting a specific JSON format (see [sample.json](https://github.com/geekodour/wscribe/blob/main/examples/output/sample.json)).
  - **QualCoder:** ([https://github.com/ccbogel/QualCoder](https://github.com/ccbogel/QualCoder)) - A general-purpose qualitative data analysis tool with some audio transcription features.

### GUI Applications with Transcription and Diarization
  - **noScribe:** ([https://github.com/kaixxx/noScribe](https://github.com/kaixxx/noScribe)) - A GUI application leveraging `faster-whisper` and `pyannote.audio`.
  - **TranscriboZH:** ([https://github.com/machinelearningZH/audio-transcription](https://github.com/machinelearningZH/audio-transcription)) - Another GUI tool based on `WhisperX`.

## Contributing

Contributions to `ghe_transcribe` are welcome! Please follow the following guidelines:
- Follow the commit message format (`<type>: <description>`), eg. `fix: typo in README.md`. Check out the [conventional commits guide](https://www.conventionalcommits.org/en/v1.0.0/) for more details.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).