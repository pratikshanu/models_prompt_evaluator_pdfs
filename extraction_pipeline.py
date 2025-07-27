import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import base64
import requests
import json
import streamlit as st

# This module is self-contained. You can configure the model and endpoint here.
OLLAMA_ENDPOINT = "http://localhost:11434/api/chat"

# --- Helper Functions ---

def image_to_base64(image: Image.Image) -> str:
    """Converts a PIL Image to a base64 encoded string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def call_ollama_api(model: str, messages: list) -> str:
    """Generic function to call the local Ollama API."""
    headers = {'Content-Type': 'application/json'}
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.0}
    }
    try:
        response = requests.post(OLLAMA_ENDPOINT, headers=headers, data=json.dumps(payload), timeout=300)
        response.raise_for_status()
        result = response.json()
        if 'message' in result and 'content' in result['message']:
            content = result['message']['content'].strip()
            if content.startswith("```markdown"):
                content = content[10:]
            if content.endswith("```"):
                content = content[:-3]
            return content.strip()
        else:
            error_info = result.get('error', 'Unknown error format.')
            st.error(f"API call failed. Reason: {error_info}")
            return f"[ERROR: API call failed. Details: {error_info}]"
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred during the API request: {e}")
        return f"[ERROR: HTTP Request failed: {e}]"

# --- Internal Pipeline Functions ---

def _get_raw_text(page: fitz.Page) -> str:
    """Performs a basic text extraction, with OCR fallback for scanned pages."""
    st.write("  - Attempting direct text extraction...")
    text = page.get_text().strip()
    if not text:
        st.warning("  - No text found. Switching to OCR...")
        try:
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            return pytesseract.image_to_string(img, lang='eng')
        except Exception as e:
            st.error(f"    - OCR failed: {e}")
            return ""
    st.success("  - Text extracted successfully.")
    return text

def _get_raw_text_from_image(page_image: Image.Image) -> str:
    """Performs OCR on a given PIL Image to get a 'rough draft'."""
    st.write("- **Step 2:** Performing OCR to get initial text.")
    try:
        text = pytesseract.image_to_string(page_image, lang='eng')
        st.success("  - OCR completed.")
        return text
    except pytesseract.TesseractNotFoundError:
        st.error("  - ERROR: Tesseract is not installed or not in your PATH.")
        return "[Tesseract Not Found - OCR Skipped]"
    except Exception as e:
        st.error(f"  - OCR Failed: {e}")
        return f"[OCR Failed: {e}]"

def _correct_text_with_vision(model: str, raw_text: str, page_image: Image.Image, prompt_template: str) -> str:
    """Uses a multimodal LLM to correct and structure the raw text against the page image."""
    st.write(f"- **Step 3:** Sending page to `{model}` for correction and structuring.")
    base64_image = image_to_base64(page_image)
    prompt = prompt_template.format(raw_text=raw_text)
    messages = [{"role": "user", "content": prompt, "images": [base64_image]}]
    corrected_text = call_ollama_api(model, messages)
    st.success("  - AI correction complete.")
    return corrected_text

# --- Main Public Function ---
def extract_and_correct_document(file_bytes: bytes, file_type: str, model_for_extraction: str, extraction_prompt: str) -> str | None:
    """
    Orchestrates the full extraction and AI-powered correction pipeline for either a PDF or an image.
    """
    if not file_bytes:
        return None

    # The spinner is now handled in the main app.py file.
    # This function now just contains the extraction logic.
    full_corrected_context = ""
    try:
        if file_type == "application/pdf":
            st.write("- **Step 1:** Opened PDF document.")
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                for i, page in enumerate(doc):
                    st.markdown(f"--- \n- **Step 2:** Processing Page {i+1}/{len(doc)}")
                    pix = page.get_pixmap(dpi=300)
                    page_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    raw_text = _get_raw_text(page)
                    corrected_text = _correct_text_with_vision(model_for_extraction, raw_text, page_image, extraction_prompt)
                    if corrected_text.startswith("[ERROR"):
                        return None
                    full_corrected_context += f"\n\n--- Page {i+1} ---\n\n{corrected_text}"
        
        elif "image" in file_type:
            st.write("- **Step 1:** Opened image file.")
            page_image = Image.open(io.BytesIO(file_bytes))
            raw_text = _get_raw_text_from_image(page_image)
            corrected_text = _correct_text_with_vision(model_for_extraction, raw_text, page_image, extraction_prompt)
            if corrected_text.startswith("[ERROR"):
                return None
            full_corrected_context = corrected_text

        else:
            st.error(f"Unsupported file type: {file_type}")
            return None
        st.markdown("---")
        st.success("✅ **Definitive extraction complete!**")
        return full_corrected_context
    except Exception as e:
        st.error(f"❌ An unexpected error occurred during extraction: {e}")
        import traceback
        traceback.print_exc()
        return None