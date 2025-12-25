import streamlit as st
import os
from pypdf import PdfReader
from groq import Groq
import time
import re

# Page config
st.set_page_config(page_title="PDF Study Song (HI+EN)", layout="wide")

# SAFE Groq client (env var ONLY)
@st.cache_resource
def get_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.error("❌ Set GROQ_API_KEY in Streamlit Cloud Settings!")
        st.stop()
    return Groq(api_key=api_key)

client = get_client()

SAVE_DIR = "pdf_songs"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------- Language detection ----------
def detect_language(text):
    hindi_chars = len(re.findall(r'[\u0900-\u097F]', text))
    latin_chars = len(re.findall(r'[A-Za-z]', text))
    total = len(text) or 1
    if hindi_chars / total > 0.05 and hindi_chars > latin_chars:
        return "hindi"
    return "english"

def chunk_text(text, size=2500):  # Larger chunks = fewer calls
    return [text[i:i+size] for i in range(0, len(text), size)]

# ---------- SINGLE API CALL: Topic + Song ----------
@st.cache_data
def make_song_with_topic(chunk, lang="auto"):
    if lang == "auto":
        lang = detect_language(chunk[:400])
    
    safe_chunk = chunk.strip()[:5000]  # Truncate for token limit
    
    if lang == "hindi":
        system_prompt = """नीचे दिए गए टेक्स्ट से EXACTLY:
1. पहली लाइन: 2-6 शब्दों का टॉपिक शीर्षक (कोई नंबर नहीं)
2. खाली लाइन  
3. हिंदी स्टडी गीत (छोटी लाइनें, कोरस, छात्रों के लिए आसान)

सिर्फ़ दिए गए टेक्स्ट के concepts/facts इस्तेमाल करो। कोई नया content नहीं।"""
    else:
        system_prompt = """From given textbook text, output EXACTLY:
1. First line: 2-6 word topic heading (NO numbers/brackets)
2. Blank line
3. English study song lyrics only (short lines, chorus, student-friendly)

Use ONLY concepts from this text. NO new facts/examples."""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": safe_chunk}
            ],
            temperature=0.5,
            max_tokens=450
        )
        full = response.choices[0].message.content.strip()
        
        lines = full.splitlines()
        if lines:
            topic = lines[0].strip()
            verse = "\n".join(lines[1:]).strip()
            return topic, verse or full
        return "Geography Topic", full
    except Exception as e:
        if "429" in str(e):
            return "Rate Limited", "Daily limit reached - wait 5min or upgrade!"
        return "Error", "Generation failed"

# ---------- PDF Processing ----------
def extract_pdf_text(uploaded_file):
    uploaded_file.seek(0)
    reader = PdfReader(uploaded_file)
    text = ""
    page_count = len(reader.pages)
    
    for page in reader.pages:
        t = page.extract_text() or ""
        text += t + "\n"
    
    # Clean text
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'[^\w\s\n।।।।।।]', ' ', text)
    return text.strip(), page_count

# ---------- Main UI ----------
st.title("🎵 PDF to Study Song Generator 🎵")
st.markdown("**English / हिंदी PDFs के लिए काम करता है (printed + scanned)**")

col1, col2 = st.columns([1, 3])

with col1:
    lang_mode = st.radio(
        "🌐 Language Mode:",
        ["🚀 Auto-detect", "🇺🇸 Force English", "🇮🇳 Force Hindi"],
        index=0
    )

with col2:
    st.info("📚 **Printed/scanned textbook PDFs** पर best results")

# SINGLE PDF upload
uploaded_file = st.file_uploader("📁 Upload PDF", type="pdf")

if uploaded_file is not None:
    with st.spinner("🔍 Reading PDF..."):
        text, page_count = extract_pdf_text(uploaded_file)

    detected_lang = detect_language(text[:2000])
    lang_display = "🇮🇳 Hindi" if detected_lang == "hindi" else "🇺🇸 English"

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("📄 Pages", page_count)
    with m2:
        st.metric("🔤 Characters", len(text))
    with m3:
        st.metric("🗣️ Detected", lang_display)

    st.success(f"✅ PDF ready! Detected: **{lang_display}**")

    if st.button("🎶 Generate Study Song", type="primary", use_container_width=True):
        if lang_mode == "🚀 Auto-detect":
            final_lang = detected_lang
        elif lang_mode == "🇺🇸 Force English":
            final_lang = "english"
        else:
            final_lang = "hindi"

        chunks = chunk_text(text)[:12]  # MAX 12 VERSES
        st.info(f"🎼 Creating **{len(chunks)} verse(s)** in **{final_lang.upper()}** …")

        bar = st.progress(0.0)
        status = st.empty()
        final_song = f"# 📚 Study Song - {lang_display}\n\n"

        rate_limit_reached = False
        
        for i, chunk in enumerate(chunks):
            if rate_limit_reached:
                final_song += f"**Verse {i+1}:** (⏳ Rate limit reached)\n\n---\n\n"
                continue
                
            status.text(f"✍️ Generating verse {i+1}/{len(chunks)} …")
            
            topic, verse = make_song_with_topic(chunk, final_lang)
            
            final_song += (
                f"**🧾 Topic:** {topic}\n\n"
                f"**🎵 Verse {i+1} 🎵**\n\n"
                f"{verse}\n\n"
                f"---\n\n"
            )
            
            bar.progress((i + 1) / len(chunks))
            time.sleep(1.0)  # PERFECT RATE LIMIT PROTECTION

        st.subheader("🎤 Your Complete Study Song")
        st.markdown(final_song)

        fname = uploaded_file.name.replace(".pdf", f"_{final_lang}_study_song.txt")
        st.download_button(
            "📥 Download Song",
            data=final_song,
            file_name=fname,
            mime="text/plain",
            use_container_width=True
        )
        st.balloons()

else:
    st.info("📁 **Upload PDF** to generate study songs!")

st.markdown("---")
st.markdown("*Powered by Groq + Streamlit* 🚀")
