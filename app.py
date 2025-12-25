%%writefile app.py
import streamlit as st
import os
from pypdf import PdfReader
from groq import Groq
import time
import re
import io
try:
    import numpy as np
    import cv2
    from pdf2image import convert_from_path
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    st.warning("⚠️ OCR disabled (install pdf2image+pytesseract locally)")

# Config
st.set_page_config(page_title="PDF Study Song (HI+EN+OCR)", layout="wide")

# Groq client (env var for Streamlit Cloud)
@st.cache_resource
def get_client():
    return Groq(api_key=os.environ.get("GROQ_API_KEY", "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"))

client = get_client()

SAVE_DIR = "pdf_songs"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------- YOUR EXACT HELPERS ----------
def preprocess_for_ocr(pil_img):
    if not OCR_AVAILABLE: return pil_img
    img = np.array(pil_img.convert("L"))
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    img = cv2.medianBlur(img, 3)
    return Image.fromarray(img)

def looks_noisy(t: str) -> bool:
    t = t.strip()
    if len(t) < 80: return True
    letters_spaces = sum(c.isalpha() or c.isspace() for c in t)
    ratio = letters_spaces / max(len(t), 1)
    return ratio < 0.5

# ---------- YOUR EXACT FUNCTIONS ----------
def get_topic_heading(chunk: str, lang: str) -> str:
    if lang == "hindi":
        system_prompt = "दिए गए अध्ययन सामग्री के लिए सिर्फ़ 2-6 शब्दों का छोटा टॉपिक/शीर्षक लिखो। पूरा वाक्य नहीं, कोई व्याख्या नहीं, सिर्फ़ शीर्षक।"
    else:
        system_prompt = "For the given study text, write ONLY a very short topic heading (2-6 words). No sentence, no explanation, just the heading."

    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": chunk[:800]}],
        temperature=0.2, max_tokens=20,
    )
    return resp.choices[0].message.content.strip()

def detect_language(text):
    hindi_chars = len(re.findall(r'[\u0900-\u097F]', text))
    latin_chars = len(re.findall(r'[A-Za-z]', text))
    total = len(text) or 1
    if hindi_chars / total > 0.05 and hindi_chars > latin_chars: return "hindi"
    return "english"

def chunk_text(text, size=1800):
    return [text[i:i+size] for i in range(0, len(text), size)]

@st.cache_data
def make_song(chunk, lang="auto"):
    if lang == "auto": lang = detect_language(chunk[:400])
    safe_chunk = chunk.strip() or f"Text is almost empty; use: {chunk[:200]}"

    if lang == "hindi":
        system_prompt = """तुम्हें नीचे दिए गए टेक्स्ट (chapter content) को ही लेकर
एक छोटा, सरल और याद रखने लायक हिंदी स्टडी गीत बनाना है।
सख्त नियम: सिर्फ़ दिए गए टेक्स्ट में जो concepts, facts हैं वही इस्तेमाल करो।"""
    else:
        system_prompt = """You are given ONLY textbook content. Turn ONLY this into study song.
STRICT: Use ONLY concepts from given text. Student-friendly, short lines, chorus."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": safe_chunk}],
        temperature=0.5, max_tokens=450
    )
    return response.choices[0].message.content.strip()

# ---------- YOUR EXACT PDF PROCESSING ----------
def process_pdf(uploaded_file):
    uploaded_file.seek(0)
    tmp_pdf_path = f"/tmp/{uploaded_file.name}"
    with open(tmp_pdf_path, "wb") as f: f.write(uploaded_file.read())
    
    noisy_pages = 0
    text = ""
    
    try:
        reader = PdfReader(tmp_pdf_path)
        page_count = len(reader.pages)
        prog = st.progress(0.0)
        
        for i, page in enumerate(reader.pages):
            t = page.extract_text() or ""
            clean = t.strip()
            
            if len(clean) < 40 and OCR_AVAILABLE:
                try:
                    images = convert_from_path(tmp_pdf_path, dpi=300, first_page=i+1, last_page=i+1)
                    img = preprocess_for_ocr(images[0])
                    ocr_text = pytesseract.image_to_string(img, lang="hin+eng")
                    if not looks_noisy(ocr_text): t = ocr_text
                    else: noisy_pages += 1
                except: pass
            
            text += t + "\n"
            prog.progress((i + 1) / page_count)
        
        if noisy_pages > page_count * 0.6:
            st.error("❌ Too much noisy OCR. Try clearer PDF.")
            st.stop()
            
    except: text = ""
    
    return text.strip(), page_count, noisy_pages

# ---------- YOUR EXACT UI ----------
st.title("🎵 PDF to Study Song Generator 🎵")
st.markdown("**English / हिंदी PDFs (printed + scanned). Handwritten experimental.**")

col1, col2 = st.columns([1, 3])
with col1:
    lang_mode = st.radio("🌐 Language Mode:", ["🚀 Auto-detect", "🇺🇸 Force English", "🇮🇳 Force Hindi", "both english and hindi"], index=0)
with col2:
    st.info("📚 Printed/scanned PDFs best. Handwritten OCR may be imperfect.")

uploaded_files = st.file_uploader("📁 Upload PDF(s)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        with st.spinner(f"🔍 Reading {uploaded_file.name}..."):
            text, page_count, noisy_pages = process_pdf(uploaded_file)
            
            detected_lang = detect_language(text[:2000])
            lang_display = "🇮🇳 Hindi" if detected_lang == "hindi" else "🇺🇸 English"
            
            m1, m2, m3 = st.columns(3)
            with m1: st.metric("📄 Pages", page_count)
            with m2: st.metric("🔤 Characters", len(text))
            with m3: st.metric("🗣️ Detected", lang_display)
            
            if noisy_pages: st.warning(f"⚠️ {noisy_pages} noisy OCR page(s)")
            st.success(f"✅ {uploaded_file.name} ready! ({lang_display})")
            
            final_lang = detected_lang if lang_mode == "🚀 Auto-detect" else "hindi" if "Hindi" in lang_mode else "english"
            
            if st.button(f"🎶 Generate Song: {uploaded_file.name}", key=f"gen_{uploaded_file.name}"):
                chunks = chunk_text(text)[:15]  # Max 15 verses
                st.info(f"🎼 Creating {len(chunks)} verses in **{final_lang.upper()}**...")
                
                bar = st.progress(0.0)
                status = st.empty()
                final_song = ""
                
                for i, chunk in enumerate(chunks):
                    status.text(f"✍️ Verse {i+1}/{len(chunks)}...")
                    topic = get_topic_heading(chunk, final_lang)
                    verse = make_song(chunk, final_lang)
                    
                    final_song += f"**🧾 Topic:** {topic}\n\n**🎵 Verse {i+1} 🎵**\n\n{verse}\n\n---\n\n"
                    bar.progress((i + 1) / len(chunks))
                    time.sleep(0.8)  # Rate limit protection
                
                st.subheader("🎤 Complete Study Song")
                st.markdown(final_song)
                
                fname = uploaded_file.name.replace(".pdf", f"_{final_lang}_study_song.txt")
                st.download_button("📥 Download", final_song, fname, "text/plain", use_container_width=True)

else:
    st.info("📁 Upload PDF to start!")
