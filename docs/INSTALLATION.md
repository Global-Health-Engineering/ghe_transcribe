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
pip install -e .
ipython kernel install --user --name=venv_ghe_transcribe
```

#### Using uv (Alternative)
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/Global-Health-Engineering/ghe_transcribe.git
cd ghe_transcribe
uv sync
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
pip install -e .

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
pip install -e .
```

## Package Extras

The package includes several optional dependency groups:

- **CLI and UI support**: `pip install -e .`
- **Development**: `pip install -e ".[dev]"` (includes testing and linting tools)
- **Everything**: `pip install -e ".[all]"`
