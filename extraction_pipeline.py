import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import base64
import requests
import json
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# This module is self-contained. You can configure the model and endpoint here.
OLLAMA_ENDPOINT = "http://localhost:11434/api/chat"

# --- Helper Functions ---

def image_to_base64(image: Image.Image) -> str:
    """Converts a PIL Image to a base64 encoded string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def call_ollama_api(model: str, messages: list) -> dict:
    """
    Generic function to call the local Ollama API.
    Returns a dictionary with content, tokens, elapsed time, or an error.
    """
    headers = {'Content-Type': 'application/json'}
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": -1, "top_p": 0.1,}
    }
    try:
        response = requests.post(OLLAMA_ENDPOINT, headers=headers, data=json.dumps(payload), timeout=1800)
        response.raise_for_status()
        result = response.json()
        if 'message' in result and 'content' in result['message']:
            content = result['message']['content'].strip()
            # Clean up potential markdown/json code blocks
            if content.startswith("```"):
                content = '\n'.join(content.split('\n')[1:])
            if content.endswith("```"):
                content = content[:-3]
            
            return {
                "content": content.strip(),
                "tokens": result.get('eval_count', 0),
                "elapsed": result.get('total_duration', 0) / 1_000_000_000
            }
        else:
            error_info = result.get('error', 'Unknown error format.')
            return {"error": f"API call failed. Details: {error_info}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"HTTP Request failed: {e}"}

# --- Internal Pipeline Functions ---

def _get_raw_text(page: fitz.Page) -> str:
    """Performs a basic text extraction, with OCR fallback for scanned pages."""
    text = page.get_text().strip()
    if not text:
        try:
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            return pytesseract.image_to_string(img, lang='eng')
        except Exception:
            return ""
    return text

def _get_raw_text_from_image(page_image: Image.Image) -> str:
    """Performs OCR on a given PIL Image to get a 'rough draft'."""
    try:
        return pytesseract.image_to_string(page_image, lang='eng')
    except Exception as e:
        return f"[OCR Failed: {e}]"

def _correct_text_with_vision(model: str, raw_text: str, page_image: Image.Image, prompt_template: str) -> dict:
    """Uses a multimodal LLM to correct and structure the raw text against the page image."""
    base64_image = image_to_base64(page_image)
    
    # Escape all braces in the prompt except for the one we want to format
    prompt = prompt_template.replace('{', '{{').replace('}', '}}').replace('{{raw_text}}', '{raw_text}')
    
    try:
        prompt = prompt.format(raw_text=raw_text)
    except KeyError as e:
        return {"error": f"Prompt formatting error. Ensure only '{{raw_text}}' is a variable. Invalid key: {e}"}

    messages = [{"role": "user", "content": prompt, "images": [base64_image]}]
    return call_ollama_api(model, messages)

# --- Worker for Parallel Processing ---
def _process_page_worker(args):
    """
    A worker function to process a single page of a PDF.
    This function is designed to be called by a thread pool executor.
    """
    page_num, file_bytes, model_for_extraction, extraction_prompt = args
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    page = doc[page_num]
    
    pix = page.get_pixmap(dpi=300)
    page_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    raw_text = _get_raw_text(page)
    
    correction_result = _correct_text_with_vision(model_for_extraction, raw_text, page_image, extraction_prompt)
    
    doc.close()
    return page_num, correction_result

# --- Main Public Function ---
def extract_and_correct_document(file_bytes: bytes, file_type: str, model_for_extraction: str, extraction_prompt: str) -> tuple[str | None, int, float]:
    """
    Orchestrates the full extraction and AI-powered correction pipeline for either a PDF or an image.
    For PDFs, it processes pages in parallel to leverage multiple GPUs.
    """
    if not file_bytes:
        return None, 0, 0.0

    total_tokens = 0
    start_time = time.time()
    
    try:
        if file_type == "application/pdf":
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            num_pages = len(doc)
            doc.close()

            page_results = [None] * num_pages
            
            # Use 4 workers, assuming 4 GPUs for optimal performance.
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Prepare arguments for each page
                tasks = [(i, file_bytes, model_for_extraction, extraction_prompt) for i in range(num_pages)]
                
                # Submit tasks to the executor
                future_to_page = {executor.submit(_process_page_worker, task): task[0] for task in tasks}
                
                for future in as_completed(future_to_page):
                    page_num = future_to_page[future]
                    try:
                        _, correction_result = future.result()
                        if "error" in correction_result:
                            st.error(f"Page {page_num + 1} correction failed: {correction_result['error']}")
                            page_results[page_num] = {"content": f"[ERROR PROCESSING PAGE {page_num + 1}]"}
                        else:
                            page_results[page_num] = correction_result
                            total_tokens += correction_result.get('tokens', 0)
                    except Exception as exc:
                        st.error(f"Page {page_num + 1} generated an exception: {exc}")
                        page_results[page_num] = {"content": f"[EXCEPTION ON PAGE {page_num + 1}]"}

            full_corrected_context = ""
            for i, result in enumerate(page_results):
                if result:
                    full_corrected_context += f"\n\n--- Page {i+1} ---\n\n{result.get('content', '')}"
            
            total_elapsed = time.time() - start_time
            return full_corrected_context.strip(), total_tokens, total_elapsed

        elif "image" in file_type:
            page_image = Image.open(io.BytesIO(file_bytes))
            raw_text = _get_raw_text_from_image(page_image)
            correction_result = _correct_text_with_vision(model_for_extraction, raw_text, page_image, extraction_prompt)
            
            if "error" in correction_result:
                return None, 0, 0.0

            full_corrected_context = correction_result.get('content', '')
            total_tokens = correction_result.get('tokens', 0)
            total_elapsed = correction_result.get('elapsed', 0.0)
            return full_corrected_context, total_tokens, total_elapsed

    except Exception as e:
        st.error(f"❌ An unexpected error occurred during extraction: {e}")
        import traceback
        traceback.print_exc()
        return None, 0, 0.0
