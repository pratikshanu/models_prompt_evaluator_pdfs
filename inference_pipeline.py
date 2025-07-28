import requests
import json
import time
import base64
import io
from PIL import Image

# This module now handles two types of inference calls.
OLLAMA_ENDPOINT = "http://localhost:11434/api/chat"

def _image_to_base64(image_bytes: bytes) -> str:
    """Helper function to convert image bytes to a base64 string."""
    buffered = io.BytesIO(image_bytes)
    img = Image.open(buffered)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def _call_ollama_api(model: str, messages: list, max_tokens: int, temperature: float, top_k: int, top_p: float) -> dict:
    """
    Generic internal function to call the local Ollama API, now with advanced parameters.
    """
    headers = {'Content-Type': 'application/json'}
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p
        }
    }
    
    start_time = time.time()
    try:
        response = requests.post(OLLAMA_ENDPOINT, headers=headers, data=json.dumps(payload), timeout=1500)
        response.raise_for_status()
        result = response.json()
        end_time = time.time()

        if 'message' in result and 'content' in result['message']:
            return {
                "content": result['message']['content'].strip(),
                "elapsed": end_time - start_time,
                "tokens": result.get("eval_count", 0) 
            }
        else:
            error_info = result.get('error', 'Unknown error format.')
            return {"error": error_info}
            
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def run_inference_on_text(model: str, final_context: str, user_prompt: str, max_tokens: int, temperature: float, top_k: int, top_p: float, prompt_template: str) -> dict:
    """
    Runs inference for a single model using text context.
    """
    # CRUCIAL FIX: Replaced faulty Template substitution with .format()
    prompt_with_context = prompt_template.format(final_context=final_context, user_prompt=user_prompt)
    messages = [{"role": "user", "content": prompt_with_context}]
    return _call_ollama_api(model, messages, max_tokens, temperature, top_k, top_p)

def run_inference_on_image(model: str, image_bytes: bytes, user_prompt: str, max_tokens: int, temperature: float, top_k: int, top_p: float, prompt_template: str) -> dict:
    """
    Runs inference for a single model using an image and a prompt, skipping extraction.
    """
    base64_image = _image_to_base64(image_bytes)
    # CRUCIAL FIX: Replaced faulty Template substitution with .format()
    # Provide a default value for final_context for image-only tasks
    prompt_for_image_qa = prompt_template.format(final_context="N/A", user_prompt=user_prompt)
    messages = [{
        "role": "user",
        "content": prompt_for_image_qa,
        "images": [base64_image]
    }]
    return _call_ollama_api(model, messages, max_tokens, temperature, top_k, top_p)