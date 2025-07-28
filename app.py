import streamlit as st
from pdf_utils import extract_pdf_pages
from extraction_pipeline import extract_and_correct_document
from inference_pipeline import run_inference_on_text, run_inference_on_image
from prompts import EXTRACTION_PROMPT, INFERENCE_PROMPT
from PIL import Image
import io
import time

st.set_page_config(page_title="LLM + Prompts Evaluator", layout="wide")

# ——— Session State Init ———
if 'session_started' not in st.session_state:
    st.session_state.session_started = False
if 'qa_mode' not in st.session_state:
    st.session_state.qa_mode = "Run Extraction Pipeline"
if 'last_prompt' not in st.session_state:
    st.session_state.last_prompt = ""
if 'inference_results' not in st.session_state:
    st.session_state.inference_results = {}
# Add new state keys for extraction metrics
if 'extraction_time' not in st.session_state:
    st.session_state.extraction_time = 0.0
if 'extraction_tokens' not in st.session_state:
    st.session_state.extraction_tokens = 0


# --- Helper Function for Report Generation ---
def generate_report_markdown(final_context, prompt, inference_results, extraction_model, file_type, extraction_time, extraction_tokens):
    """Generates a comprehensive Markdown report of the entire analysis session."""
    report = f"# Analysis Report\n\n"
    if final_context:
        report += f"## Extracted Content (using `{extraction_model}`)\n\n"
        # Add extraction metrics to the report
        report += f"**Time Taken:** {extraction_time:.2f}s | **Output Tokens:** {extraction_tokens}\n\n"
        report += f"```markdown\n{final_context}\n```\n\n"
    else:
        report += f"## Direct Q&A on Image\n\n"
        report += f"An analysis was run directly on the uploaded {file_type}.\n\n"
    report += f"---\n\n## Inference Results\n\n**Question Asked:**\n> {prompt}\n\n"
    for model, result in inference_results.items():
        report += f"### Model: `{model}`\n\n"
        if "error" in result:
            report += f"**Error:** {result['error']}\n\n"
        else:
            report += f"**Time Taken:** {result.get('elapsed', 0):.2f}s | **Output Tokens:** {result.get('tokens', 0)}\n\n"
            report += f"**Answer:**\n"
            answer_lines = result.get('content', 'No content returned.').split('\n')
            for line in answer_lines:
                report += f"> {line}\n"
            report += "\n"
        report += "---\n"
    return report

# ——— Sidebar Inputs ———
with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("#### 1. Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF or Image file", type=["pdf", "png", "jpg", "jpeg"], label_visibility="collapsed")
    
    if uploaded_file:
        if "image" in uploaded_file.type:
            st.info("🖼️ **Image Detected!** Please choose a processing mode below.")
            st.session_state.qa_mode = st.radio("Select Mode for Image",("Run Extraction Pipeline", "Ask Question Directly"), index=1, key="qa_mode_radio", horizontal=True, label_visibility="collapsed")
        else:
            st.session_state.qa_mode = "Run Extraction Pipeline"
    
    st.markdown("---")
    st.markdown("#### 2. Select Models")
    is_direct_qa_mode = (uploaded_file is not None and "image" in uploaded_file.type and st.session_state.qa_mode == "Ask Question Directly")
    extraction_model = st.selectbox("Model for Extraction", ["llava:13b", "gemma3:12b", "llama3.2-vision:11b", "mistral-small3.2:latest", "qwen2.5vl:7b"], index=2, help="Choose your best vision model to create the high-quality context. (Disabled in 'Ask Question Directly' mode)", disabled=is_direct_qa_mode)
    models_for_inference = st.multiselect("Models for Inference", ["llava:13b", "gemma3:12b", "llama3.2-vision:11b", "mistral-small3.2:latest", "qwen2.5vl:7b"], default=["llama3.2-vision:11b", "gemma3:12b", "mistral-small3.2:latest"], help="Choose which models to ask questions of.")
    
    st.markdown("---")
    st.markdown("#### 3. Set Parameters")
    max_tokens = st.slider("Max Tokens", 100, 4000, 1000, help="Sets the maximum number of tokens to generate in the response.")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, help="Controls randomness. Lower is more deterministic, higher is more creative.")
    st.markdown("##### Advanced Parameters")
    top_k = st.slider("Top K", 0, 100, 40, help="Reduces the probability of generating nonsense. A higher value (e.g., 100) gives more varied answers.")
    top_p = st.slider("Top P", 0.0, 1.0, 0.9, help="Works with Top K to improve realism. A higher value (e.g., 0.9) gives more varied answers.")

    st.markdown("---")
    if st.button("🚀 Start Session", use_container_width=True, type="primary"):
        if uploaded_file and models_for_inference:
            st.session_state.file_bytes = uploaded_file.read()
            st.session_state.file_type = uploaded_file.type
            st.session_state.extraction_model = extraction_model
            st.session_state.models_for_inference = models_for_inference
            st.session_state.session_started = True
            st.session_state.run_extraction = True
            st.session_state.inference_results = {}
        else:
            st.warning("Please upload a file and select at least one inference model.")

    if st.button("🔄 Reset", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# ——— Main Page Content ———
st.title("LLM + Prompts Evaluator")

if st.session_state.get('run_extraction'):
    is_direct_mode = st.session_state.get('qa_mode') == "Ask Question Directly"
    if "image" in st.session_state.file_type and is_direct_mode:
        st.session_state.pages_for_display = [Image.open(io.BytesIO(st.session_state.file_bytes))]
        st.session_state.final_context = None
        st.session_state.extraction_done = True
        st.success("Direct Q&A mode ready. Ask a question below.")
    else:
        with st.spinner(f"Running extraction with `{st.session_state.extraction_model}`..."), st.expander("Show Extraction Log", expanded=True):
            # Capture the new return values from the extraction function
            final_context, tokens, elapsed = extract_and_correct_document(st.session_state.file_bytes, st.session_state.file_type, st.session_state.extraction_model, EXTRACTION_PROMPT)
            
        if st.session_state.file_type == "application/pdf":
            st.session_state.pages_for_display = extract_pdf_pages(st.session_state.file_bytes)
        else:
            st.session_state.pages_for_display = [Image.open(io.BytesIO(st.session_state.file_bytes))]
            
        if final_context:
            st.session_state.final_context = final_context
            # Store metrics in session state
            st.session_state.extraction_time = elapsed
            st.session_state.extraction_tokens = tokens
            st.session_state.extraction_done = True
            st.success("Extraction complete!")
        else:
            st.error("Extraction failed. Check the log above for errors.")
            st.session_state.extraction_done = False
            
    st.session_state.run_extraction = False
    st.rerun()

if st.session_state.get('session_started') and st.session_state.get('extraction_done'):
    is_direct_mode = st.session_state.get('final_context') is None
    
    if is_direct_mode:
        st.subheader("Direct Q&A on Image")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(st.session_state.pages_for_display[0], caption="Original Image", use_container_width=True)
    else:
        st.subheader("Extracted Content vs. Original Document")
        col_img, col_text = st.columns([2.5, 3])
        with col_text:
            # Display the extraction metrics using columns and st.metric
            st.markdown(f"**Extraction Metrics** (using `{st.session_state.extraction_model}`)")
            metric1, metric2 = st.columns(2)
            metric1.metric("⏱️ Time", f"{st.session_state.get('extraction_time', 0):.2f}s")
            metric2.metric("🧮 Tokens", f"{st.session_state.get('extraction_tokens', 0)}")
            st.text_area(label="Extracted Content", value=st.session_state.final_context, height=550, label_visibility="collapsed")
            
        with col_img:
            page_tabs = st.tabs([f"Page {i+1}" for i, _ in enumerate(st.session_state.pages_for_display)])
            for i, tab in enumerate(page_tabs):
                with tab:
                    st.image(st.session_state.pages_for_display[i], caption=f"Original Page {i+1}", use_container_width=True)

    st.markdown("---")
    st.subheader("❓ Ask a Question")
    
    prompt = st.text_area("Your Question/Prompt", height=100, label_visibility="collapsed")
    col1, col2 = st.columns([5, 1])
    run_inference_clicked = col1.button("🚀 Run Inference")

    if run_inference_clicked and prompt:
        st.session_state.last_prompt = prompt
        st.session_state.inference_results = {}
        st.subheader("🧠 Inference Results")
        cols = st.columns(len(models_for_inference))
        for i, model in enumerate(models_for_inference):
            with cols[i]:
                with st.spinner(f"Running `{model}`..."):
                    if is_direct_mode:
                        result = run_inference_on_image(model, st.session_state.file_bytes, prompt, max_tokens, temperature, top_k, top_p, INFERENCE_PROMPT)
                    else:
                        result = run_inference_on_text(model, st.session_state.final_context, prompt, max_tokens, temperature, top_k, top_p, INFERENCE_PROMPT)
                    st.session_state.inference_results[model] = result
                with st.container(border=True):
                    st.markdown(f"##### **Model:** `{model}`")
                    st.markdown("---")
                    if "error" in result:
                         st.error(f"**Error:** {result['error']}")
                    else:
                        st.markdown(result.get('content', 'No content returned.'))
                        st.markdown("---")
                        st.write(f"⏱️ {result.get('elapsed', 0):.2f}s | 🧮 {result.get('tokens', 0)} tokens")

    if st.session_state.get('inference_results'):
        with col2:
            # Pass the new metrics to the report generator
            report_data = generate_report_markdown(
                st.session_state.get('final_context'),
                st.session_state.last_prompt,
                st.session_state.inference_results,
                st.session_state.extraction_model,
                st.session_state.file_type,
                st.session_state.get('extraction_time', 0),
                st.session_state.get('extraction_tokens', 0)
            )
            st.download_button(label="📥 Download", data=report_data, file_name="llm_analysis_report.md", mime="text/markdown")

elif not st.session_state.get('session_started'):
    st.info("To begin, upload a PDF or Image, configure your settings, then click **Start Session** in the sidebar.")