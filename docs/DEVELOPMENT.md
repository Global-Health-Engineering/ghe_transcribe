# Development Guide

## Prerequisites

- Python 3.10-3.12
- pip-tools

## Setup

1. **Clone and setup environment:**
   ```bash
   git clone <repository-url>
   cd ghe_transcribe
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install development dependencies:**
   ```bash
   pip install pip-tools
   pip install -e .[dev]
   ```

## Dependency Management

We use `pip-tools` to manage dependencies with `requirements.in` → `requirements.txt` workflow.

### Adding Dependencies

1. **Add to requirements.in:**
   ```bash
   echo "new-package>=1.0.0" >> requirements.in
   ```

2. **Compile requirements:**
   ```bash
   pip-compile requirements.in
   ```

3. **Install updated requirements:**
   ```bash
   pip install -r requirements.txt
   ```

### Updating Dependencies

```bash
pip-compile --upgrade requirements.in
pip install -r requirements.txt
```

## Testing

### Run Tests
```bash
pytest tests/test_core.py -v
```

### Run All Tests
```bash
pytest -v
```

### Run with Coverage
```bash
pytest --cov=ghe_transcribe tests/
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
