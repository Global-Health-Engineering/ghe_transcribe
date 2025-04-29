# ghe_transcribe: A Tool to Transcribe Audio Files with Speaker Diarization

[![Python Versions](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue)](https://www.python.org/downloads/)
[![Python application](https://github.com/Global-Health-Engineering/ghe_transcribe/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/Global-Health-Engineering/ghe_transcribe/actions/workflows/python-app.yml)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](https://github.com/Global-Health-Engineering/ghe_transcribe/blob/main/CONTRIBUTING.md)

This repository hosts `ghe_transcribe`, a powerful Python script designed to transcribe audio files and perform speaker diarization. It leverages the speed and accuracy of **Faster Whisper** (a highly optimized reimplementation of OpenAI's Whisper) for transcription and **Pyannote** for identifying and separating speakers within the audio. This tool is ideal for handling long recordings, enhancing transcription quality, and automatically segmenting audio by speaker.

## Table of Contents

1.  [**Installation**](#installation)
    * [Installation on Euler](#installation-on-euler)
    * [Installation on MacOS](#installation-on-macos)
2.  [**Usage**](#usage)
    * [Quick Start](#quick-start)
    * [Command-Line Options](#command-line-options)
3.  [**Performance**](#performance)
    * [Timing Tests](#timing-tests)
4.  [**Related Tools and Resources**](#related-tools-and-resources)
    * [Transcription](#transcription)
    * [Diarization](#diarization)
    * [Transcription + Diarization](#transcription--diarization)
    * [Graphical User Interfaces (GUIs)](#graphical-user-interfaces-guis)
    * [Transcription + Diarization + GUI](#transcription--diarization--gui)
5.  [**Contributing**](#contributing)
6.  [**License**](#license)

## Installation

Choose the installation method that suits your environment.

### Installation on Euler

Follow these steps to set up `ghe_transcribe` on the Euler cluster at ETH Zurich.

#### First login to Euler

Refer to the official [Euler wiki on getting started](https://scicomp.ethz.ch/wiki/Getting_started_with_clusters) and the [GDC's documentation on accessing Euler](https://www.gdc-docs.ethz.ch/EulerManual/site/access/) if you are a first-time user.

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
python3.11 -m venv venv3.11_ghe_transcribe --system-site-packages
source venv3.11_ghe_transcribe/bin/activate
pip3.11 install faster-whisper pyannote.audio huggingface-hub
ipython kernel install --user --name=venv3.11_ghe_transcribe
```

#### Configure JupyterHub to use the environment

To ensure your JupyterHub instances automatically use the created environment, edit the JupyterLab configuration file:

```bash
nano .config/euler/jupyterhub/jupyterlabrc
```

Add the following lines to the file:

```bash
module load stack/2024-06 python_cuda/3.11.6
source venv3.11_ghe_transcribe/bin/activate
```

### Installation on MacOS

Use the following commands to install the necessary dependencies on macOS:

```bash
brew install cmake python@3.11
```

```bash
python3.11 -m venv venv3.11_ghe_transcribe --system-site-packages
source venv3.11_ghe_transcribe/bin/activate
pip3.11 install torch torchaudio av faster-whisper pyannote.audio huggingface-hub
ipython kernel install --user --name=venv3.11_ghe_transcribe
```

## Usage

### Quick Start

To transcribe an audio file with speaker diarization:

1.  **Accept Hugging Face Model Licenses:**
    * Visit and accept the user conditions for [`pyannote/segmentation-3.0`](https://hf.co/pyannote/segmentation-3.0).
    * Visit and accept the user conditions for [`pyannote/speaker-diarization-3.1`](https://hf.co/pyannote/speaker-diarization-3.1).
2.  **Clone the repository and enter:**
    ```bash
    git clone https://github.com/Global-Health-Engineering/ghe_transcribe.git
    cd ghe_transcribe
    ```
3.  **Place your audio file:** Upload the audio file you want to transcribe (e.g., `my_audio.mp3`) into the `media` folder.
4.  **Edit `script.py` to include audio file:** To transcribe a different audio file, you will need to edit the `script.py` file and change the audio file path (currently "media/testing_audio_01.mp3") to the path of your desired audio file in the media folder (e.g., "media/my_audio.mp3"). You can also adjust the num_speakers argument in `script.py` and many more options.
5.  **Run the transcription script:** Execute the main script. You will be prompted to log in to Hugging Face if you haven't already.
    ```bash
    python script.py
    ```
### (Recommended) Save your Hugging Face token

1.  **Generate a Hugging Face Access Token:**
    * Create a new access token at [`hf.co/settings/tokens`](https://hf.co/settings/tokens).
2.  **Configure Access Token:**
    * You can edit `config.json` file in the root directory to save your access token for future use, avoiding repeated logins. The content of the file should be:
        ```json
        {
            "HF_TOKEN": "YOUR_HUGGING_FACE_ACCESS_TOKEN"
        }
        ```
    * Replace `"YOUR_HUGGING_FACE_ACCESS_TOKEN"` with the token you used during the login prompt.

### Command-Line Options

The `ghe_transcribe.core` module accepts the following command-line arguments for customization:

```
usage: python -m ghe_transcribe.core [-h] [--snip SNIP] [--device {cuda,mps,cpu}]
                        [--whisper_model {tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2,large-v3,large,distil-large-v2,distil-medium.en,distil-small.en,distil-large-v3,large-v3-turbo,turbo}]
                        [--device_index DEVICE_INDEX]
                        [--compute_type {float32,float16,int8}]
                        [--cpu_threads CPU_THREADS] [--beam_size BEAM_SIZE]
                        [--temperature TEMPERATURE] [--word_timestamps]
                        [--vad_filter]
                        [--min_silence_duration_ms MIN_SILENCE_DURATION_MS]
                        [--pyannote_model PYANNOTTE_MODEL]
                        [--num_speakers NUM_SPEAKERS]
                        [--min_speakers MIN_SPEAKERS]
                        [--max_speakers MAX_SPEAKERS] [--save_output]
                        [--info]
                        audio_file

Transcribe and diarize an audio file.

positional arguments:
  audio_file            Path to the audio file.

options:
  -h, --help            show this help message and exit
  --device {cuda,mps,cpu}
                        Device to use (cuda, mps, or cpu). Defaults to auto-
                        detection.
  --whisper_model {tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2,large-v3,large,distil-large-v2,distil-medium.en,distil-small.en,distil-large-v3,large-v3-turbo,turbo}
                        Faster Whisper model to use. Defaults to
                        large-v3-turbo.
  --device_index DEVICE_INDEX
                        Index of the device to use. Defaults to 0.
  --compute_type {float32,float16,int8}
                        Compute type for Whisper model. Defaults to float32.
  --cpu_threads CPU_THREADS
                        Number of CPU threads to use for Whisper.
  --beam_size BEAM_SIZE
                        Beam size for Whisper decoding. Defaults to 5.
  --temperature TEMPERATURE
                        Temperature for Whisper sampling. Defaults to 0.0.
  --word_timestamps     Enable word timestamps in Whisper output.
  --vad_filter          Enable voice activity detection in Whisper.
  --min_silence_duration_ms MIN_SILENCE_DURATION_MS
                        Minimum silence duration for VAD (ms). Defaults to
                        2000.
  --pyannote_model PYANNOTTE_MODEL
                        Pyannote speaker diarization model to use. Defaults
                        to pyannote/speaker-diarization-3.1.
  --num_speakers NUM_SPEAKERS
                        Number of speakers for diarization (if known).
  --min_speakers MIN_SPEAKERS
                        Minimum number of speakers for diarization.
  --max_speakers MAX_SPEAKERS
                        Maximum number of speakers for diarization.
  --save_output         Save output to .csv and .md files. (default: True)
  --info                Print detected language information. (default: True)
```

  - `audio_file`: **Required.** The path to the audio file you wish to transcribe. Supported formats include `.mp3` and `.wav`.
  - `--device` (`'cpu'`, `'cuda'`, `'mps'`, optional): Specifies the device for processing. If not provided, the script automatically detects and uses CUDA or MPS if available, otherwise defaults to CPU.
  - `--whisper_model` (string, optional): Selects the Faster Whisper model size. Available options are: `tiny.en`, `tiny`, `base.en`, `base`, `small.en`, `small`, `medium.en`, `medium`, `large-v1`, `large-v2`, `large-v3`, `large`, `distil-large-v2`, `distil-medium.en`, `distil-small.en`, `distil-large-v3`, `large-v3-turbo`, `turbo`. Defaults to `large-v3-turbo`.
  - `--device_index` (integer, optional): The index of the device to use if multiple GPUs are available. Defaults to `0`.
  - `--compute_type` (`'float32'`, `'float16'`, `'int8'`, optional): Sets the computation precision for the Whisper model. Defaults to `float32`.
  - `--cpu_threads` (integer, optional): Limits the number of CPU threads used by Faster Whisper.
  - `--beam_size` (integer, optional): The beam size for the decoding process in Whisper. Defaults to `5`.
  - `--temperature` (float, optional): The temperature parameter for Whisper sampling. Defaults to `0.0`.
  - `--word_timestamps` (flag, optional): If set, the output will include timestamps for each word.
  - `--vad_filter` (flag, optional): Enables Voice Activity Detection (VAD) to filter out silent parts of the audio before transcription.
  - `--min_silence_duration_ms` (integer, optional): The minimum duration of silence (in milliseconds) to consider for VAD. Defaults to `2000`.
  - `--pyannote_model` (string, optional): Specifies the Pyannote speaker diarization model to use. Defaults to `pyannote/speaker-diarization-3.1`.
  - `--num_speakers` (integer, optional): If the number of speakers is known, you can specify it to guide the diarization process.
  - `--min_speakers` (integer, optional): The minimum expected number of speakers.
  - `--max_speakers` (integer, optional): The maximum expected number of speakers.
  - `--save_output` (bool, optional): If set (default), the transcription with speaker labels will be saved to both `.csv` and `.md` files in the `output` directory.
  - `--info` (bool, optional): When enabled (default), the script will print information about the detected language and its confidence score.

From terminal, run:
  ```bash
  python -m ghe_transcribe.core media/testing_audio_01.mp3 --num_speakers 2
  ```

## Performance

### Timing Tests

The following table shows the execution time of `python -m ghe_transcribe.core` on the audio file [`media/testing_audio_01.wav`](https://github.com/Global-Health-Engineering/ghe_transcribe/blob/main/media/testing_audio_01.mp3) across different environments. These tests were conducted using the `timing` function defined in [`utils.py`](https://github.com/Global-Health-Engineering/ghe_transcribe/blob/main/utils.py).

| Device                                      | Time (sec) |
| :------------------------------------------ | :--------- |
| Euler Cluster (16 CPU cores, 16GB RAM) - `cpu` | 67.4988    |
| Euler Cluster (32 CPU cores, 16GB RAM) - `cpu` | 44.3622    |
| MacOS (Apple M2, 16GB RAM) - `mps`           | 41.2122    |
| MacOS (Apple M2, 16GB RAM) - `cpu`           | 64.7549    |

## Related Tools and Resources

Explore these related projects and libraries for more advanced functionalities or alternative approaches.

### Transcription

  - **Whisper:** ([https://github.com/openai/whisper](https://github.com/openai/whisper)) - OpenAI's original speech-to-text model.
  - **faster-whisper:** ([https://github.com/SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper)) - The fast implementation used in `ghe_transcribe`, offering significant speed improvements. See [benchmarks](https://github.com/SYSTRAN/faster-whisper/issues/1030).
  - **Deepgram Benchmarks:** ([https://deepgram.com/learn/benchmarking-top-open-source-speech-models](https://deepgram.com/learn/benchmarking-top-open-source-speech-models)) - A comparison of various open-source speech models, including Whisper.

### Diarization

  - **pyannote.audio:** ([https://github.com/pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio)) - The speaker diarization toolkit by pyannoteAI ([https://www.pyannote.ai/](https://www.pyannote.ai/)), requiring a Hugging Face access token ([https://hf.co/settings/tokens](https://hf.co/settings/tokens)).
  - **NeMo:** ([https://github.com/NVIDIA/NeMo](https://github.com/NVIDIA/NeMo)) - NVIDIA's open-source framework for conversational AI, including diarization capabilities.
  - **Pyannote vs NeMo Comparison:** ([https://lajavaness.medium.com/comparing-state-of-the-art-speaker-diarization-frameworks-pyannote-vs-nemo-31a191c6300](https://lajavaness.medium.com/comparing-state-of-the-art-speaker-diarization-frameworks-pyannote-vs-nemo-31a191c6300)) - A blog post comparing the two frameworks.

### Transcription + Diarization

  - **WhisperX:** ([https://github.com/m-bain/whisperX](https://github.com/m-bain/whisperX)) - Combines `faster-whisper` with `pyannote.audio` for transcription and diarization.
  - **whisper-diarization:** ([https://github.com/MahmoudAshraf97/whisper-diarization](https://github.com/MahmoudAshraf97/whisper-diarization)) - Integrates `faster-whisper` with `NeMo` for speaker diarization.
  - **insanely-fast-whisper:** ([https://github.com/Vaibhavs10/insanely-fast-whisper](https://github.com/Vaibhavs10/insanely-fast-whisper)) - Another optimized Whisper implementation that can be combined with `pyannote.audio`.

### Graphical User Interfaces (GUIs)

  - **wscribe-editor:** ([https://github.com/geekodour/wscribe-editor](https://github.com/geekodour/wscribe-editor)) - An editor for transcriptions with word-level timestamps, supporting a specific JSON format (see [sample.json](https://github.com/geekodour/wscribe/blob/main/examples/output/sample.json)).
  - **QualCoder:** ([https://github.com/ccbogel/QualCoder](https://github.com/ccbogel/QualCoder)) - A general-purpose qualitative data analysis tool with some audio transcription features.

### Transcription + Diarization + GUI

  - **noScribe:** ([https://github.com/kaixxx/noScribe](https://github.com/kaixxx/noScribe)) - A GUI application leveraging `faster-whisper` and `pyannote.audio`.
  - **TranscriboZH:** ([https://github.com/machinelearningZH/audio-transcription](https://github.com/machinelearningZH/audio-transcription)) - Another GUI tool based on `WhisperX`.

## Contributing

Contributions to `ghe_transcribe` are welcome! Please see the [Contributing Guidelines](https://github.com/Global-Health-Engineering/ghe_transcribe/blob/main/CONTRIBUTING.md) for more information on how to contribute.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).