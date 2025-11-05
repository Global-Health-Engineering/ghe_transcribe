# ghe_transcribe

[![Python Versions](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Run on RenkuLab](https://img.shields.io/badge/Run%20on-RenkuLab-orange)](https://renkulab.io/p/nmassari/ghe-transcribe)

A tool to transcribe audio files with speaker diarization using **Faster Whisper** and **Pyannote**.
- **Fast transcription** with optimized Whisper models
- **Speaker diarization** to identify different speakers
- **Multiple output formats** (TXT, SRT)
- **Jupyter interface** for interactive use
- **CLI tool** for global compatibility

<details>
<summary>Interface Preview</summary>

The Jupyter-based interface provides an intuitive way to upload audio files, configure transcription settings, and download results in multiple formats.

![GHE Transcribe App Interface](docs/screenshot_ghe_transcribe_app.png)

</details>

## Installation

### System Dependencies

This tool requires FFmpeg for audio processing:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# Windows
choco install ffmpeg
```

### ghe_transcribe

```bash
git clone https://github.com/Global-Health-Engineering/ghe_transcribe.git
cd ghe_transcribe
python -m venv venv
source venv/bin/activate
pip install -e .
```

> [!NOTE]
> See the [detailed installation guide](docs/INSTALLATION.md).

### Hugging Face Authentication

This tool uses gated models from Hugging Face that require authentication. You need to:

1. **Join Hugging Face**, to access Pyannote
	- [https://hf.co/join](https://hf.co/join)
2. **Accept User Conditions**, to use Pyannote
    - [https://hf.co/pyannote/speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1)
    - [https://hf.co/pyannote/segmentation-3.0](https://hf.co/pyannote/segmentation-3.0)
3. **Create Access Token**, to use ghe_transcribe
	- [https://hf.co/settings/tokens](https://hf.co/settings/tokens)

## Usage

### Renkulab
See the [detailed documentation for Renkulab](docs/renkulab.md)

### Jupyter Interface (Local)
Open `app.ipynb` and run the cell:
```python
from ghe_transcribe.app import execute
execute()
```

### Python API
```python
from ghe_transcribe.core import transcribe
result = transcribe("media/test01.mp3")
```

### Command Line
```bash
# Simplest call
transcribe media/test01.mp3

# Multiple files
transcribe media/test01.mp3 media/test02.m4a --trim 5

# See all options
transcribe --help 
```

## Editors

- **For SRT files** [subtitle-editor.org/](https://subtitle-editor.org/), runs locally on your browser
- **For TXT files** note-taking apps, Word, MAXQDA, QualCoder, ...

## Contributing

We welcome contributions! Please use [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/).
See our [contributions guidelines](docs/CONTRIBUTING.md).

## License

MIT License - see [LICENSE](LICENSE).
