# app.py
import streamlit as st
import os
from pypdf import PdfReader
from groq import Groq
import time, re, random

from pdf2image import convert_from_path
from PIL import Image
import pytesseract

# ---------------------------
# SAFE SAVE DIR
# ---------------------------
SAVE_DIR = "pdf_songs"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------------------
# 🔐 API KEY
# ---------------------------
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------------------------
# RETRY (ANTI-RATE LIMIT)
# ---------------------------
def safe_groq_call(messages, max_tokens=250, tries=4):
    """Retry request if GROQ gives rate limit error."""
    for attempt in range(tries):
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                temperature=0.5,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            if "rate" in str(e).lower() or "limit" in str(e).lower():
                wait = 2 * (attempt + 1)  # exponential backoff
                st.warning(f"⏳ GROQ Rate limit! retrying in {wait}s...")
                time.sleep(wait)
                continue
            return f"⚠️ Error: {e}"
    return "❌ Rate limit reached repeatedly. Try again in 1-2 minutes."

# ---------- SONG TOPIC ----------
def get_topic_heading(chunk, lang):
    if lang == "hindi":
        system = "दिए गए टेक्स्ट के लिए सिर्फ़ 2-6 शब्दों का छोटा विषय शीर्षक लिखो।"
    else:
        system = "Write ONLY a 2-6 word short heading from the text."

    messages = [
        {"role":"system","content":system},
        {"role":"user","content":chunk[:800]}
    ]
    return safe_groq_call(messages, max_tokens=30)

# ---------- LANGUAGE DETECT ----------
def detect_language(text):
    hindi_chars = len(re.findall(r'[\u0900-\u097F]', text))
    latin_chars = len(re.findall(r'[A-Za-z]', text))
    total = len(text) or 1
    return "hindi" if (hindi_chars/total > 0.05 and hindi_chars > latin_chars) else "english"

# ---------- SPLIT PDF TEXT ----------
def chunk_text(text, size=1700):
    return [text[i:i+size] for i in range(0, len(text), size)]

# ---------- SONG MAKER ----------
def make_song(chunk, lang="auto"):
    if lang == "auto":
        lang = detect_language(chunk[:500])

    if lang == "hindi":
        system = """
ONLY use given Hindi text to make a study song.
Short lines, rhyming simple.
Make 6–10 lines + 1 chorus. No new facts.
"""
    else:
        system = """
ONLY use given English textbook content to make a study song.
Simple, rhyming, student-friendly, 6–10 lines + chorus.
No extra information.
"""

    messages = [
        {"role":"system","content":system},
        {"role":"user","content":chunk}
    ]
    return safe_groq_call(messages, max_tokens=450)

# ----------------------------------------------------------------
# 🎨 UI
# ----------------------------------------------------------------
st.set_page_config(page_title="PDF Study Song Generator", layout="wide")
st.title("🎵 PDF to Study Song Generator (Hindi/English + OCR)")
st.caption("📄 Scanned PDFs supported (OCR). No API key shown.")

lang_mode = st.radio(
    "Language Mode:",
    ["🚀 Auto-detect", "🇺🇸 Force English", "🇮🇳 Force Hindi"],
    index=0
)

uploaded_file = st.file_uploader("📁 Upload PDF", type="pdf")

# ----------------------------------------------------------------
# 📥 Process PDF
# ----------------------------------------------------------------
if uploaded_file:
    tmp_pdf_path = uploaded_file.name
    with open(tmp_pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("🔍 Extracting text & running OCR..."):
        reader = PdfReader(tmp_pdf_path)
        text = ""
        progress = st.progress(0)
        page_count = len(reader.pages)

        for i, page in enumerate(reader.pages):
            raw = (page.extract_text() or "").strip()

            # If not enough text → OCR
            if len(raw) < 25:
                img = convert_from_path(tmp_pdf_path, dpi=300, first_page=i+1, last_page=i+1)[0]
                raw = pytesseract.image_to_string(img, lang="hin+eng")

            text += raw + "\n"
            progress.progress((i+1)/page_count)

    # Decide language
    detected = detect_language(text[:2000])
    final_lang = detected if lang_mode == "🚀 Auto-detect" else ("hindi" if "Hindi" in lang_mode else "english")

    st.success(f"🎯 Detected: {detected.upper()} → Output: {final_lang.upper()}")

    # ----------------------------------------------------------------
    # 🎶 Generate Song
    # ----------------------------------------------------------------
    if st.button("🎶 Generate Song", type="primary"):
        chunks = chunk_text(text)
        st.info(f"✍️ Making {len(chunks)} verse(s)...")

        bar = st.progress(0)
        final_song = ""

        for i, chunk in enumerate(chunks):
            topic = get_topic_heading(chunk, final_lang)
            verse = make_song(chunk, final_lang)

            final_song += f"## 🧾 {topic}\n\n🎵 **Verse {i+1}**\n{verse}\n\n---\n\n"
            bar.progress((i + 1) / len(chunks))

        st.subheader("✨ Final Song")
        st.markdown(final_song)

        # Save + download
        fn = uploaded_file.name.replace(".pdf", f"_{final_lang}_song.txt")
        with open(os.path.join(SAVE_DIR, fn), "w", encoding="utf-8") as f:
            f.write(final_song)

        st.download_button(
            "📥 Download Song",
            data=final_song,
            file_name=fn,
            mime="text/plain",
            use_container_width=True
        )

        st.balloons()
        st.success("🚀 Done! Your Study Song is Ready 🎤")
