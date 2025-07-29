import streamlit as st
from pdf_utils import extract_pdf_pages
from extraction_pipeline import extract_and_correct_document
from inference_pipeline import run_inference_on_text, run_inference_on_image
from prompts import EXTRACTION_PROMPT, INFERENCE_PROMPT
from PIL import Image
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

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
if 'extraction_prompt' not in st.session_state:
    st.session_state.extraction_prompt = EXTRACTION_PROMPT
if 'inference_prompt' not in st.session_state:
    st.session_state.inference_prompt = INFERENCE_PROMPT
if 'saved_prompts' not in st.session_state:
    st.session_state.saved_prompts = {"Default": {"extraction": EXTRACTION_PROMPT, "inference": INFERENCE_PROMPT}}
if 'extraction_results' not in st.session_state:
    st.session_state.extraction_results = {}
if 'selected_context_model' not in st.session_state:
    st.session_state.selected_context_model = None

# --- Helper Function for Report Generation ---
def generate_report_markdown(final_context, prompt, inference_results, extraction_model, file_type, extraction_time, extraction_tokens):
    report = f"# Analysis Report\n\n"
    if final_context:
        report += f"## Extracted Content (using `{extraction_model}`)\n\n"
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
            st.info("🖼️ **Image Detected!**")
            st.session_state.qa_mode = st.radio("Select Mode for Image",("Run Extraction Pipeline", "Ask Question Directly"), index=1, key="qa_mode_radio", horizontal=True, label_visibility="collapsed")
        else:
            st.session_state.qa_mode = "Run Extraction Pipeline"
    
    st.markdown("---")
    st.markdown("#### 2. Select Models")
    is_direct_qa_mode = (uploaded_file is not None and "image" in uploaded_file.type and st.session_state.qa_mode == "Ask Question Directly")
    
    extraction_models = st.multiselect("Models for Extraction", ["mistral-small3.2:24b"], default=["mistral-small3.2:24b"], help="Choose one or more models to perform extraction in parallel.", disabled=is_direct_qa_mode)
    models_for_inference = st.multiselect("Models for Inference", ["mistral-small3.2:24b"], default=["mistral-small3.2:24b"], help="Choose which models to ask questions of.")
    
    st.markdown("---")
    st.markdown("#### 3. Set Parameters")
    max_tokens = st.slider("Max Tokens", 100, 4000, 1000)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0)
    st.markdown("##### Advanced Parameters")
    top_k = st.slider("Top K", 0, 100, 40)
    top_p = st.slider("Top P", 0.0, 1.0, 0.9)

    st.markdown("---")
    with st.expander("**Prompt Engineering**", expanded=False):
        st.markdown("###### Prompt Library")
        col1, col2 = st.columns([3, 1])
        def load_prompt():
            name = st.session_state.selected_prompt_name
            if name in st.session_state.saved_prompts:
                st.session_state.extraction_prompt = st.session_state.saved_prompts[name]["extraction"]
                st.session_state.inference_prompt = st.session_state.saved_prompts[name]["inference"]
        prompt_names = list(st.session_state.saved_prompts.keys())
        col1.selectbox("Load Saved Prompt", prompt_names, key="selected_prompt_name", on_change=load_prompt, label_visibility="collapsed")
        if col2.button("Delete", use_container_width=True):
            name_to_delete = st.session_state.selected_prompt_name
            if name_to_delete != "Default" and name_to_delete in st.session_state.saved_prompts:
                del st.session_state.saved_prompts[name_to_delete]
                st.rerun()
        st.markdown("---")
        st.text_area("Extraction Prompt Template", key="extraction_prompt", height=200)
        st.text_area("Inference Prompt Template", key="inference_prompt", height=200)
        st.markdown("---")
        st.markdown("###### Save Current Prompts")
        col1a, col2a = st.columns([3, 1])
        new_prompt_name = col1a.text_input("Save as:", placeholder="e.g., 'Factual QA v2'", label_visibility="collapsed")
        if col2a.button("Save", use_container_width=True):
            if new_prompt_name:
                st.session_state.saved_prompts[new_prompt_name] = {"extraction": st.session_state.extraction_prompt, "inference": st.session_state.inference_prompt}
                st.success(f"Prompt '{new_prompt_name}' saved!")
                time.sleep(1) 
                st.rerun()
            else:
                st.warning("Please enter a name to save the prompt.")

    st.markdown("---")
    if st.button("🚀 Start Session", use_container_width=True, type="primary"):
        if uploaded_file and extraction_models and models_for_inference:
            st.session_state.file_bytes = uploaded_file.read()
            st.session_state.file_type = uploaded_file.type
            st.session_state.extraction_models = extraction_models
            st.session_state.models_for_inference = models_for_inference
            st.session_state.session_started = True
            st.session_state.run_extraction = True
            st.session_state.inference_results = {}
            st.session_state.extraction_results = {}
            st.rerun()
        else:
            st.warning("Please upload a file and select at least one extraction and inference model.")

    if st.button("🔄 Reset", use_container_width=True):
        for k in list(st.session_state.keys()):
            if k != 'saved_prompts':
                del st.session_state[k]
        st.rerun()

# --- Worker Functions for Parallelism ---
def _run_extraction_worker(args):
    model, file_bytes, file_type, extraction_prompt = args
    return model, extract_and_correct_document(file_bytes, file_type, model, extraction_prompt)

def _run_inference_worker(args):
    model, is_direct_mode, file_bytes, user_prompt, max_tokens, temperature, top_k, top_p, inference_prompt_template, final_context = args
    if is_direct_mode:
        result = run_inference_on_image(model, file_bytes, user_prompt, max_tokens, temperature, top_k, top_p, inference_prompt_template)
    else:
        result = run_inference_on_text(model, final_context, user_prompt, max_tokens, temperature, top_k, top_p, inference_prompt_template)
    return model, result

# ——— Main Page Content ———
st.title("LLM + Prompts Evaluator")

# This block now handles running extraction and displaying results in real-time
if st.session_state.get('run_extraction'):
    is_direct_mode = st.session_state.get('qa_mode') == "Ask Question Directly"
    if "image" in st.session_state.file_type and is_direct_mode:
        st.session_state.pages_for_display = [Image.open(io.BytesIO(st.session_state.file_bytes))]
        st.session_state.extraction_done = True
        st.success("Direct Q&A mode ready. Ask a question below.")
        st.session_state.run_extraction = False
        st.rerun()
    else:
        st.subheader("Extraction Results")
        status = st.status(f"Running extraction for {len(st.session_state.extraction_models)} models...", expanded=True)
        
        placeholders = {}
        with status:
            extraction_tabs = st.tabs([f"📄 {model}" for model in st.session_state.extraction_models])
            for i, model in enumerate(st.session_state.extraction_models):
                with extraction_tabs[i]:
                    placeholders[model] = st.empty()
                    placeholders[model].info(f"Queued extraction for `{model}`...")
        
        with ThreadPoolExecutor() as executor:
            future_to_model = {executor.submit(_run_extraction_worker, (model, st.session_state.file_bytes, st.session_state.file_type, st.session_state.extraction_prompt)): model for model in st.session_state.extraction_models}
            
            pending_models = st.session_state.extraction_models.copy()
            completed_models = []
            for future in as_completed(future_to_model):
                model = future_to_model[future]
                with placeholders[model].container():
                    _, (context, tokens, elapsed) = future.result()
                    st.session_state.extraction_results[model] = {"context": context, "tokens": tokens, "elapsed": elapsed}
                    
                    result = st.session_state.extraction_results[model]
                    col_img, col_text = st.columns([2.5, 3])
                    with col_text:
                        st.markdown(f"**Extraction Metrics**")
                        metric1, metric2 = st.columns(2)
                        metric1.metric("⏱️ Time", f"{result.get('elapsed', 0):.2f}s")
                        metric2.metric("🧮 Tokens", f"{result.get('tokens', 0)}")
                        st.text_area("Extracted Content", result.get('context', "Extraction failed."), height=550, key=f"text_{model}")
                    
                    if st.session_state.get('pages_for_display') is None:
                         if st.session_state.file_type == "application/pdf":
                            st.session_state.pages_for_display = extract_pdf_pages(st.session_state.file_bytes)
                         else:
                            st.session_state.pages_for_display = [Image.open(io.BytesIO(st.session_state.file_bytes))]

                    with col_img:
                        page_tabs_inner = st.tabs([f"Page {p+1}" for p, _ in enumerate(st.session_state.pages_for_display)])
                        for p, tab_inner in enumerate(page_tabs_inner):
                            with tab_inner:
                                st.image(st.session_state.pages_for_display[p], caption=f"Original Page {p+1}", use_container_width=True)
                
                completed_models.append(f"`{model}`")
                pending_models.remove(model)
                pending_str = ", ".join([f"`{p}`" for p in pending_models])
                status.update(label=f"Completed: {', '.join(completed_models)}. Pending: {pending_str if pending_models else 'None'}.")

        st.session_state.extraction_done = True
        st.session_state.run_extraction = False
        status.update(label="All extractions complete!", state="complete", expanded=True)
        st.rerun()

# This block handles displaying the UI after extraction is complete
elif st.session_state.get('session_started'):
    if st.session_state.get('extraction_done'):
        is_direct_mode = st.session_state.get('qa_mode') == "Ask Question Directly"
        
        if is_direct_mode:
            st.subheader("Direct Q&A on Image")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(st.session_state.pages_for_display[0], caption="Original Image", use_container_width=True)
        elif 'extraction_results' in st.session_state and st.session_state.extraction_results:
            st.subheader("Extraction Results")
            extraction_tabs = st.tabs([f"📄 {model}" for model in st.session_state.extraction_models])
            for i, model in enumerate(st.session_state.extraction_models):
                with extraction_tabs[i]:
                    result = st.session_state.extraction_results.get(model, {})
                    col_img, col_text = st.columns([2.5, 3])
                    with col_text:
                        st.markdown(f"**Extraction Metrics**")
                        metric1, metric2 = st.columns(2)
                        metric1.metric("⏱️ Time", f"{result.get('elapsed', 0):.2f}s")
                        metric2.metric("🧮 Tokens", f"{result.get('tokens', 0)}")
                        st.text_area("Extracted Content", result.get('context', "Extraction failed."), height=550, key=f"text_{model}_display")
                    with col_img:
                        page_tabs_inner = st.tabs([f"Page {p+1}" for p, _ in enumerate(st.session_state.pages_for_display)])
                        for p, tab_inner in enumerate(page_tabs_inner):
                            with tab_inner:
                                st.image(st.session_state.pages_for_display[p], caption=f"Original Page {p+1}", use_container_width=True)

        st.markdown("---")
        st.subheader("❓ Ask a Question")
        
        successful_extractions = [m for m, r in st.session_state.extraction_results.items() if r.get('context')]
        if not is_direct_mode and successful_extractions:
            st.session_state.selected_context_model = st.selectbox("Choose context for inference:", successful_extractions)
        
        prompt = st.text_area("Your Question/Prompt", height=100, label_visibility="collapsed")
        
        col1, col2 = st.columns([5, 1])
        if col1.button("🚀 Run Inference"):
            if prompt:
                st.session_state.last_prompt = prompt
                st.session_state.inference_results = {}
                st.session_state.run_inference = True
                st.rerun()
            else:
                st.warning("Please enter a question to run inference.")

    # New block to handle running and displaying inference
    if st.session_state.get('run_inference'):
        final_context = None
        if not is_direct_mode and st.session_state.selected_context_model:
            final_context = st.session_state.extraction_results[st.session_state.selected_context_model]['context']

        st.subheader("🧠 Inference Results")
        status = st.status(f"Running inference for {len(models_for_inference)} models...", expanded=True)
        
        placeholders = {}
        with status:
            cols = st.columns(len(models_for_inference))
            for i, model_name in enumerate(models_for_inference):
                with cols[i]:
                    placeholders[model_name] = st.empty()
                    placeholders[model_name].info(f"Queued inference for `{model_name}`...")

        with ThreadPoolExecutor() as executor:
            future_to_model = {executor.submit(_run_inference_worker, (model, is_direct_mode, st.session_state.file_bytes, st.session_state.last_prompt, max_tokens, temperature, top_k, top_p, st.session_state.inference_prompt, final_context)): model for model in models_for_inference}
            
            pending_models = models_for_inference.copy()
            completed_models = []
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                with placeholders[model_name].container():
                    model, result = future.result()
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
                
                completed_models.append(f"`{model_name}`")
                pending_models.remove(model_name)
                pending_str = ", ".join([f"`{p}`" for p in pending_models])
                status.update(label=f"Completed: {', '.join(completed_models)}. Pending: {pending_str if pending_models else 'None'}.")

        st.session_state.run_inference = False
        status.update(label="All inferences complete!", state="complete", expanded=True)

    elif st.session_state.get('inference_results'):
        # This block displays results from a completed run
        st.subheader("🧠 Inference Results")
        if st.session_state.last_prompt:
             st.write(f"Showing results for question: *\"{st.session_state.last_prompt}\"*")

        cols = st.columns(len(st.session_state.models_for_inference))
        for i, model_name in enumerate(st.session_state.models_for_inference):
            with cols[i]:
                result = st.session_state.inference_results.get(model_name, {})
                with st.container(border=True):
                    st.markdown(f"##### **Model:** `{model_name}`")
                    st.markdown("---")
                    if "error" in result:
                         st.error(f"**Error:** {result['error']}")
                    else:
                        st.markdown(result.get('content', 'No content returned.'))
                        st.markdown("---")
                        st.write(f"⏱️ {result.get('elapsed', 0):.2f}s | 🧮 {result.get('tokens', 0)} tokens")
    
    if st.session_state.get('inference_results'):
        with col2:
            report_data = generate_report_markdown(
                st.session_state.extraction_results.get(st.session_state.selected_context_model, {}).get('context'),
                st.session_state.last_prompt,
                st.session_state.inference_results,
                st.session_state.selected_context_model,
                st.session_state.file_type,
                st.session_state.extraction_results.get(st.session_state.selected_context_model, {}).get('elapsed', 0),
                st.session_state.extraction_results.get(st.session_state.selected_context_model, {}).get('tokens', 0)
            )
            st.download_button(label="📥 Download", data=report_data, file_name="llm_analysis_report.md")

elif not st.session_state.get('session_started'):
    st.info("To begin, upload a PDF or Image, configure your settings, then click **Start Session** in the sidebar.")
