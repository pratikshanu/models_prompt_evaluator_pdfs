# Dockerfile

# Stage 1: Base image with NVIDIA CUDA support
# Using a specific CUDA image that's commonly stable for AI/ML tasks on Ubuntu 22.04
FROM nvidia/cuda:12.3.2-base-ubuntu22.04

# Set custom model storage path for Ollama. This will be where models are stored.
ENV OLLAMA_MODELS=/mnt/ollama-data

# Install system-level dependencies required for Python, Ollama, Tesseract,
# PDF processing (PyMuPDF), and image manipulation (Pillow).
# 'procps' is included for 'pkill' used during model pre-pulling.
RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
        python3 \
        python3-pip \
        curl \
        git \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        poppler-utils \
        tesseract-ocr \
        tesseract-ocr-eng \
        fonts-liberation \
        libatlas-base-dev \
        libopenjp2-7 \
        libtiff-dev \
        libjpeg-dev \
        zlib1g-dev \
        libfreetype6-dev \
        liblcms2-dev \
        libwebp-dev \
        libharfbuzz-dev \
        libfribidi-dev \
        libimage-exiftool-perl \
        libxcb1 \
        libxcb-render0 \
        libxcb-shape0 \
        libxcb-xfixes0 \
        libfontconfig1 \
        libpangocairo-1.0-0 \
        procps && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# Install Ollama by downloading and executing its official install script.
# The -L flag is added for following redirects.
RUN curl -L https://ollama.ai/install.sh | sh

# Create the persistent model directory for Ollama.
RUN mkdir -p ${OLLAMA_MODELS}

# --- Robust Ollama Model Pre-pulling during Build ---
# This block ensures Ollama is fully installed and running before attempting to pull models.
# It starts Ollama in the background, waits for its API to be responsive,
# pulls the specified models, and then gracefully shuts down the temporary Ollama server.
RUN bash -c '\
    echo "Starting Ollama server for model pre-pulling..." && \
    ollama serve > /dev/null 2>&1 & \
    OLLAMA_PID=$! && \
    echo "Waiting for Ollama API to become available..." && \
    until curl -s http://localhost:11434 > /dev/null; do sleep 5; done && \
    echo "✅ Ollama server is ready. Beginning model pulls..." && \
    \
    # Pull the required models. Added "|| true" to prevent build failure if a pull fails.
    ollama pull llava:13b || echo "WARNING: Failed to pull llava:13b. Continuing build." && \
    ollama pull gemma3:12b || echo "WARNING: Failed to pull gemma3:12b. Continuing build." && \
    ollama pull llama3.2-vision:11b || echo "WARNING: Failed to pull llama3.2-vision:11b. Continuing build." && \
    ollama pull mistral-small3.2:24b || echo "WARNING: Failed to pull mistral-small3.2:24b. Continuing build." && \
    ollama pull qwen2.5vl:7b || echo "WARNING: Failed to pull qwen2.5vl:7b. Continuing build." && \
    \
    echo "Model pulling complete. Shutting down build-time Ollama server..." && \
    pkill ollama || true && \
    wait $OLLAMA_PID || true && \
    echo "Build-time Ollama server shut down." \
'

# Set the working directory inside the container.
WORKDIR /app

# Copy all application files from the current directory on the host to /app in the container.
COPY . /app

# Install Python packages required by your application.
# --no-cache-dir reduces image size.
# --break-system-packages is used on newer Python/pip versions within system environments.
RUN pip3 install --no-cache-dir \
    streamlit \
    Pillow \
    PyMuPDF \
    pytesseract \
    requests \
    pdf2image

# Expose the ports that will be used by Streamlit and Ollama (if external access is desired).
EXPOSE 8501
EXPOSE 11434

# Entrypoint for running your application.
# It starts the Ollama server in the background, binding to all interfaces (0.0.0.0),
# waits for it to be ready, and then launches the Streamlit application.
ENTRYPOINT ["bash", "-c", "\
    echo \"Starting Ollama server for application runtime...\" && \
    OLLAMA_HOST=0.0.0.0 ollama serve > /dev/null 2>&1 & \
    echo \"Waiting for Ollama API to become available for the application...\" && \
    until curl -s http://localhost:11434 > /dev/null; do sleep 2; done && \
    echo \"✅ Ollama is ready for the application! Launching Streamlit...\" && \
    streamlit run app.py --server.port=8501 --server.address=0.0.0.0 \
"]