# Development Guide

## Prerequisites

- Python 3.10-3.12
- pip-tools

## Setup

1. **Clone and setup environment:**
   ```bash
   git clone https://github.com/Global-Health-Engineering/ghe_transcribe.git
   cd ghe_transcribe
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install development dependencies:**
   ```bash
   pip install -e .[dev]
   ```

## Dependency Management

We use `pip-tools` and `uv` to manage dependencies with `requirements.in` → `requirements.txt` → `pyproject.toml` workflow.

### Adding Dependencies

1. **Add to requirements.in:**
   ```bash
   echo "new-package>=1.0.0" >> requirements.in
   ```

2. **Compile requirements to requirements.txt:**
   ```bash
   pip-compile requirements.in
   ```

3. **Add requirements to pyproject.toml:**
   ```bash
   uv add -r requirements.txt --frozen
   ```

## Testing

```bash
pytest -v
```

## Code Quality

### Format Code
```bash
ruff format .
```

### Lint Code
```bash
ruff check .
```

### Type Check
```bash
mypy src/ghe_transcribe/
```
