# Use Jupyter's scipy-notebook as base (has jovyan user, JupyterHub, everything needed)
FROM quay.io/jupyter/scipy-notebook:python-3.10

# Switch to root for system packages
USER root

# Install ffmpeg for audio processing
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Switch back to jovyan user (required by RenkuLab)
USER ${NB_USER}

# Set working directory
WORKDIR ${HOME}

# Copy project files
COPY --chown=${NB_UID}:${NB_GID} . /tmp/ghe_transcribe/

# Install the package
RUN pip install --no-cache-dir -e /tmp/ghe_transcribe/

# Expose Jupyter port
EXPOSE 8888

# CMD will be overridden by RenkuLab, but provide a default
CMD ["start-notebook.py"]