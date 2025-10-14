# Generic Python base image - works anywhere
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install the package (this reads pyproject.toml and installs everything)
RUN pip install --no-cache-dir -e .

# Default command
CMD ["python"]