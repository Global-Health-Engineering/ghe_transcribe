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

## Usage

### Jupyter Interface (Recommended)
Open `app.ipynb` and run the cell:
```python
from ghe_transcribe.app import execute
execute()
```

### Python API
```python
from ghe_transcribe.core import transcribe_core
result = transcribe_core("path/to/audio.mp3")
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

## Dependencies
```bash
pip install ipykernel>=6.30.0 jupyter-client>=8.6.0 ipywidgets>=8.0.0 IPython>=8.0.0 torch>=2.0.0 torchaudio>=2.0.0 typer>=0.9.0 av>=11.0.0 faster-whisper>=1.0.0 pyannote.audio>=3.0.0 PyYAML>=6.0
```

## Contributing

We welcome contributions! Please use [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/).

## License

MIT License - see [LICENSE](LICENSE) for details.
