# Detailed Installation Guide

This guide covers installation for different environments and use cases.

## Environment-Specific Installation

### macOS

#### Using pip (Recommended)
```bash
python -m venv venv_ghe_transcribe
source venv_ghe_transcribe/bin/activate
git clone https://github.com/Global-Health-Engineering/ghe_transcribe.git
cd ghe_transcribe
pip install -e ".[ui]"
ipython kernel install --user --name=venv_ghe_transcribe
```

#### Using uv (Alternative)
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/Global-Health-Engineering/ghe_transcribe.git
cd ghe_transcribe
uv sync --extra ui
uv run ipython kernel install --user --name=venv_ghe_transcribe
```

### Euler Cluster (ETH Zurich)

#### Prerequisites
1. Navigate to [https://jupyter.euler.hpc.ethz.ch/](https://jupyter.euler.hpc.ethz.ch/)
2. Log in with your ETHZ account
3. Open a terminal in JupyterLab

#### Setup
```bash
# Load required modules
module load stack/2024-06 python_cuda/3.11.6

# Create virtual environment
python -m venv venv_ghe_transcribe --system-site-packages
source venv_ghe_transcribe/bin/activate

# Install package
git clone https://github.com/Global-Health-Engineering/ghe_transcribe.git
cd ghe_transcribe
pip install -e ".[ui]"

# Install Jupyter kernel
ipython kernel install --user --name=venv_ghe_transcribe
```

#### JupyterHub Configuration
```bash
echo "module load stack/2024-06 python_cuda/3.11.6 && source venv_ghe_transcribe/bin/activate" >> ~/.config/euler/jupyterhub/jupyterlabrc
```

### Other Linux Systems
```bash
# Standard installation
python3 -m venv venv_ghe_transcribe
source venv_ghe_transcribe/bin/activate
git clone https://github.com/Global-Health-Engineering/ghe_transcribe.git
cd ghe_transcribe
pip install -e ".[ui]"
```

## Package Extras

The package includes several optional dependency groups:

- **Core only**: `pip install -e .`
- **UI support**: `pip install -e ".[ui]"` (includes Jupyter widgets)
- **Development**: `pip install -e ".[dev]"` (includes testing and linting tools)
- **Everything**: `pip install -e ".[all]"`

## Using uv (Modern Package Manager)

### Installation
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Linux, add to PATH
export PATH="$HOME/.local/bin:$PATH"

# On macOS, source cargo env
source ~/.cargo/env
```

### Project Setup
```bash
git clone https://github.com/Global-Health-Engineering/ghe_transcribe.git
cd ghe_transcribe

# Install all dependencies (uses uv.lock for reproducible builds)
uv sync

# Install with specific extras
uv sync --extra ui
uv sync --extra dev
```

## Troubleshooting

### Common Issues

#### Module Not Found Error
If you get `ModuleNotFoundError: No module named 'ghe_transcribe'`:

1. **In Jupyter**: Restart your kernel after installation
2. **In Terminal**: Make sure you're in the correct virtual environment
3. **Check installation**: Run `pip list | grep ghe` to verify

#### Permission Errors on Clusters
If uv fails with permission errors, use the pip method instead.

#### Python Version Issues
This package requires Python 3.10-3.12. Check your version:
```bash
python --version
```

### Verification
Test your installation:
```bash
# Test CLI
transcribe --help

# Test Python import
python -c "from ghe_transcribe.core import transcribe_core; print('Success!')"

# Test Jupyter integration
python -c "from ghe_transcribe.app import execute; print('UI available!')"
```

## Development Setup

For contributing to the project:

```bash
git clone https://github.com/Global-Health-Engineering/ghe_transcribe.git
cd ghe_transcribe

# Install with development dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest tests/

# Run linting
ruff check src/ tests/

# Format code
ruff format src/ tests/
```

## Performance Considerations

- **GPU Support**: The package will automatically use CUDA if available
- **Memory**: Large audio files may require significant RAM
- **CPU Threads**: Use `--cpu-threads` to control parallel processing