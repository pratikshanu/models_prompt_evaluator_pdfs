# Stage 1: Base image with NVIDIA CUDA support
FROM nvidia/cuda:12.3.2-base-ubuntu22.04

ENV OLLAMA_MODELS=/persistent/ollama-data

# Install system-level dependencies
RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
        python3 python3-pip curl git libglib2.0-0 libsm6 libxrender1 libxext6 poppler-utils \
        tesseract-ocr tesseract-ocr-eng fonts-liberation libatlas-base-dev libopenjp2-7 \
        libtiff-dev libjpeg-dev zlib1g-dev libfreetype6-dev liblcms2-dev libwebp-dev \
        libharfbuzz-dev libfribidi-dev libimage-exiftool-perl libxcb1 libxcb-render0 \
        libxcb-shape0 libxcb-xfixes0 libfontconfig1 libpangocairo-1.0-0 procps && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -L https://ollama.ai/install.sh | sh
RUN mkdir -p ${OLLAMA_MODELS}

WORKDIR /app
COPY . /app

# Make the new warmup script executable
RUN chmod +x /app/warmup.mistral.sh

# --- Robust Ollama Model Pre-pulling (MISTRAL ONLY) ---
RUN bash -c '\
    echo "Starting Ollama server for model setup..." && \
    ollama serve > /dev/null 2>&1 & \
    OLLAMA_PID=$! && \
    echo "Waiting for Ollama API to become available..." && \
    until curl -s http://localhost:11434 > /dev/null; do sleep 5; done && \
    echo "✅ Ollama server is ready. Pulling Mistral model..." && \
    ollama pull mistral-small3.2:24b || echo "WARNING: Failed to pull mistral-small3.2:24b." && \
    echo "Model setup complete. Shutting down build-time Ollama server..." && \
    pkill ollama || true && \
    wait $OLLAMA_PID || true && \
    echo "Build-time Ollama server shut down." \
'

# Install Python packages
RUN pip3 install --no-cache-dir streamlit Pillow PyMuPDF pytesseract requests pdf2image

# Expose ports
EXPOSE 8501
EXPOSE 11434

# Entrypoint for running your application
ENTRYPOINT ["bash", "-c", "\
    echo \"Starting Ollama server for application runtime...\" && \
    OLLAMA_HOST=0.0.0.0 ollama serve > /dev/null 2>&1 & \
    echo \"Waiting for Ollama API to become available...\" && \
    until curl -s --fail http://localhost:11434/api/tags > /dev/null; do \
        echo \"- Waiting for Ollama...\" ; \
        sleep 2; \
    done && \
    echo \"✅ Ollama is ready.\" && \
    echo \"Running Mistral warmup script...\" && \
    ./warmup.sh && \
    echo \"✅ Warmup complete.\" && \
    echo \"Launching Streamlit...\" && \
    streamlit run app.py --server.port=8501 --server.address=0.0.0.0 \
"]