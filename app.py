import streamlit as st
import os
from pypdf import PdfReader
from groq import Groq
import time
import re
import io

# Config
st.set_page_config(page_title="PDF Study Song (HI+EN+OCR)", layout="wide")

# 🚨 SAFE Groq client (env var ONLY)
@st.cache_resource
def get_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.error("❌ **Set GROQ_API_KEY in Streamlit Cloud Settings!**")
        st.stop()
    return Groq(api_key=api_key)

client = get_client()

# ---------- Helpers ----------
def looks_noisy(t: str) -> bool:
    t = t.strip()
    if len(t) < 80: return True
    letters_spaces = sum(c.isalpha() or c.isspace() for c in t)
    ratio = letters_spaces / max(len(t), 1)
    return ratio < 0.5

# ---------- Topic heading ----------
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

# ---------- Language detection ----------
def detect_language(text):
    hindi_chars = len(re.findall(r'[\u0900-\u097F]', text))
    latin_chars = len(re.findall(r'[A-Za-z]', text))
    total = len(text) or 1
    if hindi_chars / total > 0.05 and hindi_chars > latin_chars: return "hindi"
    return "english"

def chunk_text(text, size=2500):  # Larger = fewer API calls
    return [text[i:i+size] for i in range(0, len(text), size)]

# ---------- Song generation ----------
@st.cache_data
def make_song(chunk, lang="auto"):
    if lang == "auto": lang = detect_language(chunk[:400])
    safe_chunk = chunk.strip() or f"Text is almost empty; use: {chunk[:200]}"

    if lang == "hindi":
        system_prompt = """तुम्हें नीचे दिए गए टेक्स्ट (chapter content) को ही लेकर
एक छोटा, सरल और याद रखने लायक हिंदी स्टडी गीत बनाना है।

सख्त नियम:
- सिर्फ़ दिए गए टेक्स्ट में जो concepts, facts, definitions, examples हैं, वही इस्तेमाल करो
- कोई नया example, जगह, कहानी, व्यक्ति, organization खुद से मत बनाओ
- अगर कुछ समझ में नहीं आता, उसे छोड़ दो; अपने से कुछ मत जोड़ो
- छात्रों के लिए आसान हिंदी, छोटी लाइनें, कोरस और दोहराव
- आउटपुट सिर्फ़ गीत के बोल हो; explanation या "मैं नहीं कर सकता" मत लिखो"""
    else:
        system_prompt = """You are given ONLY textbook content for a specific chapter.
Turn ONLY this content into a short, simple, easy-to-memorize study song.

STRICT rules:
- Use ONLY concepts, terms, definitions, and examples that appear in the given text
- Do NOT add any new topics, places, names, stories, or facts that are not clearly present
- If something is unclear or missing, SKIP it instead of inventing details
- Student-friendly, short lines with a small chorus
- Output ONLY song lyrics, never explanations or meta-comments"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": safe_chunk}],
            temperature=0.5, max_tokens=450
        )
        return response.choices[0].message.content.strip()
    except:
        return "Song generation failed (rate limit?)"

# ---------- PDF Processing ----------
def extract_pdf_text(uploaded_file):
    uploaded_file.seek(0)
    reader = PdfReader(uploaded_file)
    text = ""
    
    for page in reader.pages:
        t = page.extract_text() or ""
        text += t + "\n"
    
    # Clean text
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'[^\w\s\n।।।।।।]', ' ', text)
    return text.strip()

# ---------- Main UI ----------
st.title("🎵 PDF to Study Song Generator 🎵")
st.markdown("**English / हिंदी PDFs के लिए काम करता है (printed + scanned). Handwritten experimental.**")

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
    accept_multiple_files=True,
    help="Drag & drop multiple PDFs (Max 200MB each)"
)

if uploaded_files:
    st.warning("⚠️ **Free limit: 12 verses/PDF**. Upgrade Groq for unlimited!")
    
    all_songs = ""
    
    for uploaded_file in uploaded_files:
        with st.spinner(f"🔍 Reading {uploaded_file.name}..."):
            raw_text = extract_pdf_text(uploaded_file)
            uploaded_file.seek(0)
            reader = PdfReader(uploaded_file)
            page_count = len(reader.pages)
            
            detected_lang = detect_language(raw_text[:2000])
            lang_display = "🇮🇳 Hindi" if detected_lang == "hindi" else "🇺🇸 English"
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📄 Pages", page_count)
            with col2:
                st.metric("🔤 Characters", len(raw_text))
            with col3:
                st.metric("🗣️ Detected", lang_display)
            
            st.success(f"✅ {uploaded_file.name} ready! ({lang_display})")
            
            # Language mapping
            if lang_mode == "🚀 Auto-detect":
                final_lang = detected_lang
            elif "Hindi" in lang_mode:
                final_lang = "hindi"
            else:
                final_lang = "english"
            
            # Generate button per PDF
            if st.button(f"🎼 Generate Study Song: {uploaded_file.name}", key=f"gen_{uploaded_file.name}", type="primary"):
                chunks = chunk_text(raw_text)
                MAX_VERSES = 12  # Rate limit safe
                
                st.info(f"🎼 Creating {min(MAX_VERSES, len(chunks))} verse(s) in **{final_lang.upper()}** …")
                
                bar = st.progress(0.0)
                status = st.empty()
                pdf_song = f"# 📚 {uploaded_file.name}\n\n"
                
                for i, chunk in enumerate(chunks[:MAX_VERSES]):
                    status.text(f"✍️ {uploaded_file.name}: Verse {i+1}/{min(MAX_VERSES, len(chunks))} …")
                    
                    try:
                        topic = get_topic_heading(chunk, final_lang)
                        verse = make_song(chunk, final_lang)
                        
                        pdf_song += (
                            f"**🧾 Topic:** {topic}\n\n"
                            f"**🎵 Verse {i+1} 🎵**\n\n"
                            f"{verse}\n\n"
                            f"---\n\n"
                        )
                    except Exception as e:
                        pdf_song += f"**Verse {i+1}:** (Skipped - rate limit)\n\n---\n\n"
                    
                    bar.progress((i + 1) / min(MAX_VERSES, len(chunks)))
                    time.sleep(0.8)  # Rate limit protection
                
                all_songs += pdf_song
                
                st.markdown("### 🎤 Your Complete Study Song")
                st.markdown(pdf_song)
                
                fname = uploaded_file.name.replace(".pdf", f"_{final_lang}_study_song.txt")
                st.download_button(
                    "📥 Download Song",
                    data=pdf_song,
                    file_name=fname,
                    mime="text/plain",
                    use_container_width=True
                )
    
    # All PDFs download
    if all_songs and len(uploaded_files) > 1:
        st.download_button(
            "📥 Download All Songs",
            data=all_songs,
            file_name="all_study_songs.txt",
            mime="text/plain",
            use_container_width=True
        )

else:
    st.info("📁 Upload PDF(s) to generate study songs!")
