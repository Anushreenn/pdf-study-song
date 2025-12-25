import streamlit as st
import os
from pypdf import PdfReader
from groq import Groq
import time
import re
import io
from PIL import Image
import pytesseract

# Page config
st.set_page_config(page_title="PDF Study Song (HI+EN+OCR)", layout="wide")

# Groq client
@st.cache_resource
def get_client():
    return Groq(api_key=os.environ.get("GROQ_API_KEY"))

client = get_client()

SAVE_DIR = "pdf_songs"

# ---------- Helpers ----------
def preprocess_for_ocr(pil_img):
    """Light preprocessing to help Tesseract on scans/handwritten."""
    from PIL import ImageEnhance
    img = pil_img.convert("L")  # grayscale
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2)
    return img

def looks_noisy(t: str) -> bool:
    """Return True if text looks like OCR garbage."""
    t = t.strip()
    if len(t) < 80:
        return True
    letters_spaces = sum(c.isalpha() or c.isspace() for c in t)
    ratio = letters_spaces / max(len(t), 1)
    return ratio < 0.5

# ---------- Topic heading ----------
def get_topic_heading(chunk: str, lang: str) -> str:
    if lang == "hindi":
        system_prompt = (
            "दिए गए अध्ययन सामग्री के लिए सिर्फ़ 2-6 शब्दों का छोटा टॉपिक/शीर्षक लिखो। "
            "पूरा वाक्य नहीं, कोई व्याख्या नहीं, सिर्फ़ शीर्षक।"
        )
    else:
        system_prompt = (
            "For the given study text, write ONLY a very short topic heading "
            "(2-6 words). No sentence, no explanation, just the heading."
        )

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
    if hindi_chars / total > 0.05 and hindi_chars > latin_chars:
        return "hindi"
    return "english"

def chunk_text(text, size=1800):
    return [text[i:i+size] for i in range(0, len(text), size)]

# ---------- Song generation ----------
@st.cache_data
def make_song(chunk, lang="auto"):
    if lang == "auto":
        lang = detect_language(chunk[:400])

    safe_chunk = chunk.strip()
    if not safe_chunk:
        safe_chunk = "Text is almost empty and noisy; use only these few visible words:\n" + chunk[:200]

    if lang == "hindi":
        system_prompt = """तुम्हें नीचे दिए गए टेक्स्ट (chapter content) को ही लेकर
एक छोटा, सरल और याद रखने लायक हिंदी स्टडी गीत बनाना है।

सख्त नियम:
- सिर्फ़ दिए गए टेक्स्ट में जो concepts, facts, definitions, examples हैं, वही इस्तेमाल करो
- कोई नया example, जगह, कहानी, व्यक्ति, organization खुद से मत बनाओ
- अगर कुछ समझ में नहीं आता, उसे छोड़ दो; अपने से कुछ मत जोड़ो
- छात्रों के लिए आसान हिंदी, छोटी लाइनें, कोरस और दोहराव
- आउटपुट सिर्फ़ गीत के बोल हो; explanation या "मैं नहीं कर सकता" मत लिखो
"""
    else:
        system_prompt = """You are given ONLY textbook content for a specific chapter.
Turn ONLY this content into a short, simple, easy-to-memorize study song.

STRICT rules:
- Use ONLY concepts, terms, definitions, and examples that appear in the given text
- Do NOT add any new topics, places, names, stories, or facts that are not clearly present
- If something is unclear or missing, SKIP it instead of inventing details
- Student-friendly, short lines with a small chorus
- Output ONLY song lyrics, never explanations or meta-comments
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": safe_chunk}
        ],
        temperature=0.5,
        max_tokens=450
    )
    return response.choices[0].message.content.strip()

# ---------- PDF Processing with OCR ----------
def extract_pdf_text(uploaded_file):
    text = ""
    noisy_pages = 0
    page_count = 0
    
    # Save uploaded file temporarily
    tmp_pdf_path = f"/tmp/{uploaded_file.name}"
    with open(tmp_pdf_path, "wb") as f:
        f.write(uploaded_file.read())
        uploaded_file.seek(0)  # Reset for later use
    
    try:
        reader = PdfReader(tmp_pdf_path)
        page_count = len(reader.pages)
        
        for i, page in enumerate(reader.pages):
            t = page.extract_text() or ""
            clean = t.strip()
            
            # If almost no text, try OCR
            if len(clean) < 40:
                try:
                    images = convert_from_path(tmp_pdf_path, dpi=200, first_page=i+1, last_page=i+1)
                    img = preprocess_for_ocr(images[0])
                    ocr_text = pytesseract.image_to_string(img, lang='hin+eng')
                    t = ocr_text if not looks_noisy(ocr_text) else t
                    if looks_noisy(ocr_text):
                        noisy_pages += 1
                except:
                    pass
            
            text += t + "\n"
    except:
        text = ""
    
    return text.strip(), page_count, noisy_pages

# ---------- Main UI ----------
st.title("🎵 PDF to Study Song Generator 🎵")
st.markdown("**English / हिंदी PDFs के लिए काम करता है (printed + scanned). Handwritten is experimental.**")

col1, col2 = st.columns([1, 3])
with col1:
    lang_mode = st.radio(
        "🌐 Language Mode:",
        ["🚀 Auto-detect", "🇺🇸 Force English", "🇮🇳 Force Hindi", "both english and hindi"],
        index=0
    )

with col2:
    st.info("📚 Printed / scanned textbook PDFs पर best results. Handwritten notes पर OCR हमेशा accurate नहीं होगा।")

# Multi-file upload
uploaded_files = st.file_uploader(
    "📁 Upload PDF(s)", 
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    all_songs = ""
    
    for uploaded_file in uploaded_files:
        with st.spinner(f"🔍 Processing {uploaded_file.name}..."):
            raw_text, page_count, noisy_pages = extract_pdf_text(uploaded_file)
            
            detected_lang = detect_language(raw_text[:2000])
            lang_display = "🇮🇳 Hindi" if detected_lang == "hindi" else "🇺🇸 English"
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📄 Pages", page_count)
            with col2:
                st.metric("🔤 Characters", len(raw_text))
            with col3:
                st.metric("🗣️ Detected", lang_display)
            
            if noisy_pages:
                st.warning(f"⚠️ {noisy_pages} noisy page(s) in {uploaded_file.name}")
            
            st.success(f"✅ {uploaded_file.name} ready! ({lang_display})")
            
            # Language selection
            if lang_mode == "🚀 Auto-detect":
                final_lang = detected_lang
            elif "Hindi" in lang_mode:
                final_lang = "hindi"
            else:
                final_lang = "english"
            
            if st.button(f"🎼 Generate Song for {uploaded_file.name}", key=f"gen_{uploaded_file.name}"):
                chunks = chunk_text(raw_text)
                st.info(f"🎼 Creating {min(20, len(chunks))} verse(s) from {uploaded_file.name}...")
                
                bar = st.progress(0.0)
                status = st.empty()
                pdf_song = f"\n# 📚 {uploaded_file.name}\n\n"
                
                for i, chunk in enumerate(chunks[:20]):  # Max 20 verses
                    status.text(f"✍️ {uploaded_file.name}: Verse {i+1}/{min(20, len(chunks))}")
                    
                    topic = get_topic_heading(chunk, final_lang)
                    verse = make_song(chunk, final_lang)
                    
                    pdf_song += (
                        f"**🧾 Topic:** {topic}\n\n"
                        f"**🎵 Verse {i+1} 🎵**\n\n"
                        f"{verse}\n\n"
                        f"---\n\n"
                    )
                    
                    bar.progress((i + 1) / min(20, len(chunks)))
                    time.sleep(0.5)
                
                all_songs += pdf_song
                
                st.markdown("### 🎤 Study Song Generated!")
                st.markdown(pdf_song)
    
    # Final download
    if all_songs:
        st.download_button(
            "📥 Download All Songs",
            data=all_songs,
            file_name="study_songs.txt",
            mime="text/plain"
        )

else:
    st.info("📁 Upload PDF(s) to generate study songs!")
