import ollama
import time
from io import BytesIO

def image_to_bytes(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return buffered.getvalue()

def query_model_with_image(model, image, prompt, max_tokens=500, temperature=0.2):
    img_bytes = image_to_bytes(image)
    start = time.perf_counter()
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt, "images": [img_bytes]}],
        options={"num_predict": max_tokens, "temperature": temperature}
    )
    end = time.perf_counter()
    token_count = response.get("eval_count", None)
    return {
        "elapsed": end - start,
        "tokens": token_count,
        "content": response["message"]["content"]
    }

def query_model_with_text(model, context, user_prompt, max_tokens=500, temperature=0.2):
    start = time.perf_counter()
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": user_prompt}
        ],
        options={"num_predict": max_tokens, "temperature": temperature}
    )
    end = time.perf_counter()
    token_count = response.get("eval_count", None)
    return {
        "elapsed": end - start,
        "tokens": token_count,
        "content": response["message"]["content"]
    }
