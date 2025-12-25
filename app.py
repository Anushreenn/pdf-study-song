# app.py
import streamlit as st
import os
from pypdf import PdfReader
from groq import Groq
import time
import re

from pdf2image import convert_from_path
from PIL import Image
import pytesseract

# ---------------------------
# SAFE SAVE DIR
# ---------------------------
SAVE_DIR = "pdf_songs"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------------------
# 🔐 API KEY (DO NOT HARD-CODE)
# Add in Streamlit Cloud: Settings → Secrets → GROQ_API_KEY
# Add in local system: export GROQ_API_KEY="your_key_here"
# ---------------------------
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Tesseract path NOT needed on cloud; Streamlit Cloud will install via packages.txt

# ---------- Topic heading ----------

def get_topic_heading(chunk: str, lang: str) -> str:
    """Generate a 2–6 word heading ONLY based on given text."""
    if lang == "hindi":
        system_prompt = "दिए गए टेक्स्ट के लिए सिर्फ़ 2-6 शब्दों का छोटा शीर्षक लिखो।"
    else:
        system_prompt = "Write ONLY a 2-6 word short topic heading based on text."

    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chunk[:800]},
        ],
        temperature=0.2,
        max_tokens=20,
    )
    return resp.choices[0].message.content.strip()

# ---------- Language detection ----------

def detect_language(text):
    hindi_chars = len(re.findall(r'[\u0900-\u097F]', text))
    latin_chars = len(re.findall(r'[A-Za-z]', text))
    total = len(text) or 1
    return "hindi" if hindi_chars / total > 0.05 and hindi_chars > latin_chars else "english"

def chunk_text(text, size=1800):
    return [text[i:i+size] for i in range(0, len(text), size)]

# ---------- Song generation ----------

@st.cache_data
def make_song(chunk, lang="auto"):
    if lang == "auto":
        lang = detect_language(chunk[:400])

    chunk = chunk.strip() or "Text missing due to scan. Use visible words only."

    if lang == "hindi":
        system_prompt = """
ONLY use given Hindi chapter content to create a simple Hindi study song.
No new examples or facts. Short lines + chorus. No explanations.
"""
    else:
        system_prompt = """
ONLY use given English chapter content to create a simple study song.
No adding new facts. Short lines + chorus. Student-friendly.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": chunk}],
        temperature=0.5,
        max_tokens=450,
    )
    return response.choices[0].message.content.strip()

# ----------------------------------------------------------------
# 🎨 UI
# ----------------------------------------------------------------
st.set_page_config(page_title="PDF Study Song Generator", layout="wide")
st.title("🎵 PDF to Study Song Generator (Hindi/English + OCR)")

st.caption("📄 Works for scanned PDFs using OCR • No API key is shown")

lang_mode = st.radio(
    "Language Mode:",
    ["🚀 Auto-detect", "🇺🇸 Force English", "🇮🇳 Force Hindi"],
    index=0
)

uploaded_file = st.file_uploader("📁 Upload PDF", type="pdf")

if uploaded_file:
    tmp_pdf_path = uploaded_file.name
    with open(tmp_pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("🔍 Extracting text & OCR scanning when needed..."):
        reader = PdfReader(tmp_pdf_path)
        text = ""
        page_count = len(reader.pages)
        prog = st.progress(0.0)

        for i, page in enumerate(reader.pages):
            t = (page.extract_text() or "").strip()

            if len(t) < 20:
                images = convert_from_path(tmp_pdf_path, dpi=300, first_page=i+1, last_page=i+1)
                img = images[0]
                t = pytesseract.image_to_string(img, lang="hin+eng")

            text += t + "\n"
            prog.progress((i + 1) / page_count)

    detected_lang = detect_language(text[:2000])
    final_lang = (
        detected_lang if lang_mode == "🚀 Auto-detect"
        else "hindi" if "Hindi" in lang_mode
        else "english"
    )

    st.success(f"🎯 Detected Language: **{detected_lang.upper()}** → Output: **{final_lang.upper()}**")

    if st.button("🎶 Generate Song", type="primary"):
        chunks = chunk_text(text)
        st.info(f"✍️ Creating {len(chunks)} verse(s)...")

        bar = st.progress(0.0)
        final_song = ""

        for i, chunk in enumerate(chunks):
            topic = get_topic_heading(chunk, final_lang)
            verse = make_song(chunk, final_lang)

            final_song += f"**🧾 Topic:** {topic}\n\n**🎵 Verse {i+1}**\n{verse}\n\n---\n\n"
            bar.progress((i + 1) / len(chunks))
            time.sleep(0.2)

        st.subheader("✨ Final Song")
        st.markdown(final_song)

        file_name = uploaded_file.name.replace(".pdf", f"_{final_lang}_song.txt")
        save_path = os.path.join(SAVE_DIR, file_name)

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(final_song)

        st.download_button(
            "📥 Download Song",
            data=final_song,
            file_name=file_name,
            mime="text/plain",
            use_container_width=True
        )

        st.success("🚀 Done! Ready to Deploy!")

