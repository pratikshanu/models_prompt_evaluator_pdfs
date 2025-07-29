#!/bin/bash

# --- Configuration ---
OLLAMA_HOST="http://localhost:11434" # Default Ollama host in Docker
MODELS=(
    "mistral-small3.2:24b"
    "llama3.2-vision:11b" 
    "gemma3:12b"
)
MAX_RETRIES=30 # Max attempts to check if Ollama is ready
SLEEP_TIME=5   # Seconds to wait between retries

# --- Functions ---

# Function to check if Ollama server is running
check_ollama_ready() {
    echo "Checking if Ollama server is ready at $OLLAMA_HOST..."
    for i in $(seq 1 $MAX_RETRIES); do
        if curl -s -o /dev/null "$OLLAMA_HOST/api/tags"; then
            echo "Ollama server is ready!"
            return 0
        fi
        echo "Attempt $i/$MAX_RETRIES: Ollama not ready yet. Waiting $SLEEP_TIME seconds..."
        sleep $SLEEP_TIME
    done
    echo "Error: Ollama server did not become ready after $MAX_RETRIES attempts."
    return 1
}

# Function to warm up a single model
warm_up_model() {
    local model_name=$1
    echo "Warming up model: $model_name..."
    # Send a dummy generate request.
    # stream: false ensures a single response, keep_alive: -1 reiterates the indefinite load.
    curl -X POST "$OLLAMA_HOST/api/generate" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$model_name\",
            \"prompt\": \"Hi\",
            \"stream\": false,
            \"options\": { \"keep_alive\": \"-1\" }
        }" > /dev/null 2>&1 & # Run in background to not block the script
    echo "Initiated warm-up for $model_name in the background."
}

# --- Main Script Execution ---

echo "Starting Ollama model warm-up script..."

# 1. Wait for Ollama server to be ready
if ! check_ollama_ready; then
    echo "Exiting warm-up script due to Ollama server not being available."
    exit 1
fi

# 2. Iterate through models and warm them up
for model in "${MODELS[@]}"; do
    warm_up_model "$model"
done

echo "All specified Ollama models are being warmed up in the background."
echo "You can check their status with 'docker exec <ollama_container_name> ollama ps' or 'nvidia-smi'."

# The script exits here, but the background curl commands continue to run.
# Ensure your main application (e.g., Streamlit) starts after