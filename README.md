# ghe_transcribe

[![Python Versions](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A tool to transcribe audio files with speaker diarization using **Faster Whisper** and **Pyannote**.

## Installation

```bash
git clone https://github.com/Global-Health-Engineering/ghe_transcribe.git
cd ghe_transcribe
pip install -e .
```

For Euler cluster or development setup, see the [detailed installation guide](docs/INSTALLATION.md).

## Hugging Face Authentication

This tool uses gated models from Hugging Face that require authentication. You need to:

1. **Join Hugging Face**: [huggingface.co/join](https://huggingface.co/join)
1. **Accept User Conditions**: You must accept conditions for BOTH models:
   - [https://hf.co/pyannote/speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1)
   - [https://hf.co/pyannote/speaker-diarization-community-1](https://hf.co/pyannote/speaker-diarization-community-1)
2. **Create Access Token**: Visit [https://hf.co/settings/tokens](https://hf.co/settings/tokens) to create a new access token with read permissions


## Usage

### Jupyter Interface (Recommended)
Open `app.ipynb` and run the cell:
```python
from ghe_transcribe.app import execute
execute()
```

### Python API
```python
from ghe_transcribe.core import transcribe
result = transcribe("path/to/audio.mp3")
```

### Command Line
```bash
transcribe path/to/audio.mp3
transcribe --help  # See all options
```

## Key Features

- **Fast transcription** with optimized Whisper models
- **Speaker diarization** to identify different speakers
- **Multiple output formats** (TXT, SRT)
- **Jupyter interface** for interactive use
- **CLI tool** for batch processing

## Editors

- **For SRT files** [subtitle-editor.org/](https://subtitle-editor.org/), runs locally on your browser
- **For TXT files** note-taking apps, Word, MAXQDA, QualCoder, ...

## Contributing

We welcome contributions! Please use [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/).

## License

MIT License - see [LICENSE](LICENSE) for details.
