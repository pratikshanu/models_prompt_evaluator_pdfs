# ✅ Use NVIDIA base image with CUDA support
FROM nvidia/cuda:12.9.1-base-ubuntu24.04

# 📦 Set custom model storage path
ENV OLLAMA_MODELS=/mnt/ollama-data

# 🧰 System-level dependencies
RUN apt update && \
    apt install -y \
        python3 \
        python3-pip \
        curl \
        git \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        poppler-utils \
        ttf-mscorefonts-installer && \
    apt clean

# 🧠 Install Ollama
RUN curl https://ollama.ai/install.sh | sh

# 📁 Create persistent model directory
RUN mkdir -p /mnt/ollama-data

# ⏳ Start Ollama and preload models
RUN ollama serve & \
    bash -c 'until curl -s http://localhost:11434/api/tags > /dev/null; do sleep 2; done' && \
    echo "✅ Ollama is ready during build!" && \
    ollama pull gemma3:12b && \
    ollama pull llama3.2-vision:11b && \
    ollama pull qwen2.5vl:7b && \
    ollama pull mistral-small3.2:24b

# 📁 Set up workspace
WORKDIR /app
COPY . /app

# 🐍 Python packages
RUN pip3 install --no-cache-dir \
    streamlit \
    pillow \
    ollama \
    pdf2image --break-system-packages

# 🌐 Streamlit port
EXPOSE 8501

# 🚀 Entrypoint for running your app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
