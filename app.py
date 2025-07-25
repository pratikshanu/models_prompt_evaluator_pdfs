import streamlit as st
from pdf_utils import extract_pdf_pages
from inference_pipeline import run_extraction, run_inference
from PIL import Image

st.set_page_config(page_title="📘 PDF Model Comparator", layout="wide")

# ——— Session State Init ———
for key, default in {
    'pages': None,
    'models': None,
    'extracted': False,
    'extraction_done': False,
    'extraction_results': {},
    'inference_results': {}
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ——— Sidebar Inputs ———
with st.sidebar:
    st.title("⚙️ Controls")
    pdf_file = st.file_uploader("📄 Upload PDF", type=["pdf"])
    models = st.multiselect(
        "🤖 Models",
        ["llama3.2-vision:11b", "gemma3:12b", "mistral-small3.2:24b"],
        default=["gemma3:12b", "llama3.2-vision:11b"]
    )
    max_tokens = st.slider("🔧 Max tokens", 100, 2000, 500)
    temperature = st.slider("🌡️ Temperature", 0.1, 1.0, 0.2)

    extract_clicked = st.button("📑 Extract Document")
    reset_clicked = st.button("🔄 Reset")

    if reset_clicked:
        for k in ['pages', 'models', 'extracted', 'extraction_done', 'extraction_results', 'inference_results']:
            st.session_state[k] = False if k in ('extracted', 'extraction_done') else None if k in ('pages', 'models') else {}

    if extract_clicked and pdf_file and models:
        pages = extract_pdf_pages(pdf_file.read())
        st.session_state.pages = pages
        st.session_state.models = models
        st.session_state.extraction_results = {m: [None]*len(pages) for m in models}
        st.session_state.extracted = True
        st.session_state.extraction_done = False

# ——— Main Page ———
st.title("🧠 PDF Model Comparison Tool")

if not st.session_state.extracted:
    st.info("Upload a PDF, select your models, then click **Extract Document** in the sidebar.")
    st.stop()

pages = st.session_state.pages
models = st.session_state.models

# ——— Extraction Phase ———
if not st.session_state.extraction_done:
    st.markdown("---")
    st.subheader("📥 Running Extraction...")
    extraction_prompt = (
        "Extract all text including headings and paragraphs as-is.\n"
        "For tables, provide JSON.\n"
        "For charts, figures, and illustrations, interpret and describe them."
    )
    for i, page in enumerate(pages):
        for model in models:
            with st.spinner(f"🔄 Extracting Page {i+1} with `{model}`…"):
                text = run_extraction(
                    [model], [page], extraction_prompt,
                    max_tokens, temperature
                )[model][0]
                st.session_state.extraction_results[model][i] = text
    st.session_state.extraction_done = True

# ——— Always show extracted content using nested tabs ———
st.markdown("---")
st.subheader("📄 Extracted Content")

page_tabs = st.tabs([f"📄 Page {i+1}" for i in range(len(pages))])
for i, tab in enumerate(page_tabs):
    with tab:
        st.image(pages[i], caption=f"Page {i+1}", use_container_width=True)
        model_tabs = st.tabs(models)
        for model_tab, model in zip(model_tabs, models):
            with model_tab:
                text = st.session_state.extraction_results[model][i]
                st.text_area(
                    label=f"{model} — Page {i+1}",
                    value=text,
                    height=300,
                    key=f"display_{model}_page_{i}"
                )

# ——— Q&A Interface ———
st.markdown("---")
st.header("🔍 Ask Questions on Extracted Content")
prompt = st.text_area("Your Question/Prompt")

if st.button("🚀 Run Inference") and prompt:
    st.session_state.inference_results = {}
    for model in models:
        with st.spinner(f"🔎 Running inference for `{model}`…"):
            res = run_inference(
                [model],
                {model: st.session_state.extraction_results[model]},
                prompt,
                max_tokens,
                temperature
            )[model]
        st.session_state.inference_results[model] = res

# ——— Inference Results ———
if st.session_state.inference_results:
    st.markdown("---")
    st.header("🧠 Inference Results")
    for model, result in st.session_state.inference_results.items():
        st.markdown(
            f"### ✅ `{model}`\n"
            f"⏱️ {result['elapsed']:.2f}s | 🧮 {result['tokens']} tokens\n\n"
            f"{result['content']}"
        )
