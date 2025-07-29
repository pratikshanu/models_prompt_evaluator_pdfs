# Stage 1: Base image with NVIDIA CUDA support
# Using a specific CUDA image that's commonly stable for AI/ML tasks on Ubuntu 22.04
FROM nvidia/cuda:12.3.2-base-ubuntu22.04

# Set custom model storage path for Ollama to a persistent location.
ENV OLLAMA_MODELS=/persistent/ollama-data

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

# Set the working directory inside the container.
WORKDIR /app

# Copy all application files from the current directory on the host to /app in the container.
COPY . /app

# --- Make the warmup script executable ---
RUN chmod +x /app/warmup.sh

# --- Robust Ollama Model Pre-pulling ---
# This block ensures Ollama is fully installed and running before attempting to pull models.
RUN bash -c '\
    echo "Starting Ollama server for model setup..." && \
    ollama serve > /dev/null 2>&1 & \
    OLLAMA_PID=$! && \
    echo "Waiting for Ollama API to become available..." && \
    until curl -s http://localhost:11434 > /dev/null; do sleep 5; done && \
    echo "✅ Ollama server is ready. Beginning model pulls..." && \
    \
    # Pull the required models.
    ollama pull llava:13b || echo "WARNING: Failed to pull llava:13b." && \
    ollama pull gemma3:12b || echo "WARNING: Failed to pull gemma3:12b." && \
    ollama pull llama3.2-vision:11b || echo "WARNING: Failed to pull llama3.2-vision:11b." && \
    ollama pull mistral-small3.2:24b || echo "WARNING: Failed to pull mistral-small3.2:24b." && \
    ollama pull qwen2.5vl:7b || echo "WARNING: Failed to pull qwen2.5vl:7b." && \
    \
    echo "Model setup complete. Shutting down build-time Ollama server..." && \
    pkill ollama || true && \
    wait $OLLAMA_PID || true && \
    echo "Build-time Ollama server shut down." \
'

# Install Python packages required by your application.
RUN pip3 install --no-cache-dir \
    streamlit \
    Pillow \
    PyMuPDF \
    pytesseract \
    requests \
    pdf2image

# Expose the ports that will be used by Streamlit and Ollama.
EXPOSE 8501
EXPOSE 11434

# Entrypoint for running your application.
# This now starts Ollama, runs the warmup script in the background, and then starts Streamlit.
ENTRYPOINT ["bash", "-c", "\
    echo \"Starting Ollama server for application runtime...\" && \
    OLLAMA_HOST=0.0.0.0 ollama serve > /dev/null 2>&1 & \
    echo \"Waiting for Ollama API to become available for the application...\" && \
    until curl -s http://localhost:11434 > /dev/null; do sleep 2; done && \
    echo \"✅ Ollama is ready! Running warmup script in the background...\" && \
    ./warmup.sh & \
    echo \"✅ Warmup script initiated. Launching Streamlit...\" && \
    streamlit run app.py --server.port=8501 --server.address=0.0.0.0 \
"]