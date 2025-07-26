import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import base64
import requests
import json
import streamlit as st # Import streamlit for caching

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
        response = requests.post(OLLAMA_ENDPOINT, headers=headers, data=json.dumps(payload))
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
            print(f"API call failed. Reason: {error_info}")
            return f"[ERROR: API call failed. Details: {error_info}]"
    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the API request: {e}")
        return f"[ERROR: HTTP Request failed: {e}]"

# --- Internal Pipeline Functions ---

def _get_raw_text(page: fitz.Page) -> str:
    """Performs a basic text extraction, with OCR fallback for scanned pages."""
    print(f"    - Performing initial raw text extraction for page {page.number + 1}...")
    text = page.get_text()
    if len(text.strip()) < 100:
        print(f"    - Scanned page detected, falling back to OCR for page {page.number + 1}...")
        try:
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            return pytesseract.image_to_string(img, lang='eng')
        except Exception as e:
            print(f"    - OCR failed: {e}")
            return ""
    return text

# **FIX:** Added the missing helper function for image OCR
def _get_raw_text_from_image(page_image: Image.Image) -> str:
    """Performs OCR on a given PIL Image to get a 'rough draft'."""
    print("    - Performing OCR on image...")
    try:
        return pytesseract.image_to_string(page_image, lang='eng')
    except pytesseract.TesseractNotFoundError:
        print("    - ERROR: Tesseract is not installed or not in your PATH.")
        return "[Tesseract Not Found - OCR Skipped]"
    except Exception as e:
        return f"[OCR Failed: {e}]"

def _correct_text_with_vision(model: str, raw_text: str, page_image: Image.Image) -> str:
    """Uses a multimodal LLM to correct and structure the raw text against the page image."""
    print(f"    - Sending page to {model} for correction and structuring...")
    base64_image = image_to_base64(page_image)
    
    prompt = f"""
    You are a world-class document transcription expert. Your single most important job is to create a perfect, 1-to-1 transcription of the provided document image into a single block of Markdown text.

    **CRITICAL INSTRUCTIONS:**
    1.  **COMPLETE TRANSCRIPTION IS MANDATORY:** You must transcribe ALL text from the image, including every paragraph, heading, list, and footnote. DO NOT OMIT ANY TEXT.
    2.  **UNIFY TABLES:** If the document contains tables with complex, multi-line headers, you MUST render them as a SINGLE, UNIFIED Markdown table with merged headers.
    3.  **DIRECT OUTPUT ONLY:** Your entire response must be ONLY the transcribed content. Do not include any introductory phrases or explanations.

    **ROUGH EXTRACTED TEXT (for context and error-checking):**
    ---
    {raw_text}
    ---
    """
    
    messages = [{"role": "user", "content": prompt, "images": [base64_image]}]
    return call_ollama_api(model, messages)

# --- Main Public Function ---

@st.cache_data
def extract_and_correct_document(file_bytes: bytes, file_type: str, model_for_extraction: str) -> str | None:
    """
    Orchestrates the full extraction and AI-powered correction pipeline for either a PDF or an image.
    This function is cached to avoid re-running on the same file.
    """
    if not file_bytes:
        return None

    print(f"🚀 Starting AI-Powered Extraction for a {file_type} file with model '{model_for_extraction}'")
    full_corrected_context = ""

    try:
        if file_type == "application/pdf":
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                for i, page in enumerate(doc):
                    print(f"\n📄 Processing Page {i+1}/{len(doc)}...")
                    pix = page.get_pixmap(dpi=300)
                    page_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    raw_text = _get_raw_text(page)
                    
                    corrected_text = _correct_text_with_vision(model_for_extraction, raw_text, page_image)
                    if corrected_text.startswith("[ERROR"):
                        return None
                    full_corrected_context += f"\n\n--- Page {i+1} ---\n\n{corrected_text}"
        
        elif "image" in file_type:
            print("\n📄 Processing single image file...")
            page_image = Image.open(io.BytesIO(file_bytes))
            raw_text = _get_raw_text_from_image(page_image)
            
            corrected_text = _correct_text_with_vision(model_for_extraction, raw_text, page_image)
            if corrected_text.startswith("[ERROR"):
                return None
            full_corrected_context = corrected_text

        else:
            print(f"Unsupported file type: {file_type}")
            return None

        with open("definitive_extraction_output.md", "w", encoding="utf-8") as f:
            f.write(full_corrected_context)
        print("\n✅ Definitive extraction complete.")
        print("   - Corrected context saved to 'definitive_extraction_output.md'")
        
        return full_corrected_context

    except Exception as e:
        print(f"❌ An unexpected error occurred during extraction: {e}")
        import traceback
        traceback.print_exc()
        return None
