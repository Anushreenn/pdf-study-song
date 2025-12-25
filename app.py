import streamlit as st
import PyPDF2
import fitz  # PyMuPDF
from openai import OpenAI
import time
import re
from langdetect import detect, LangDetectException
import os

# Page config
st.set_page_config(
    page_title="PDF to Study Song Generator",
    page_icon="🎵",
    layout="wide"
)

# Groq client
@st.cache_resource
def get_client():
    return OpenAI(
        api_key=os.environ.get("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )

client = get_client()

# Language detection
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

# Chunk text into larger pieces to reduce API calls
def chunk_text(text, size=8000):
    return [text[i:i+size] for i in range(0, len(text), size)]

# Combined function: topic + song verse
@st.cache_data
def make_song_with_topic(chunk, lang="auto"):
    if lang == "auto":
        lang = detect_language(chunk[:400])

    safe_chunk = chunk.strip()
    if not safe_chunk:
        safe_chunk = "Text is almost empty and noisy; use only these few visible words:\n" + chunk[:200]

    if lang == "hindi":
        system_prompt = """तुम्हें नीचे दिए गए टेक्स्ट (chapter content) से ही
एक छोटा, सरल और याद रखने लायक हिंदी स्टडी गीत बनाना है।

आउटपुट फॉर्मेट:
पहली लाइन: केवल 2-6 शब्दों का छोटा टॉपिक/शीर्षक (कोई नंबर, कोई ब्रैकेट नहीं)
फिर एक खाली लाइन
फिर केवल गीत के बोल।

सख्त नियम:
- सिर्फ़ दिए गए टेक्स्ट में जो concepts, facts, definitions, examples हैं, वही इस्तेमाल करो
- कोई नया example, जगह, कहानी, व्यक्ति खुद से मत बनाओ
- अगर कुछ समझ में नहीं आता, उसे छोड़ दो; अपने से कुछ मत जोड़ो
- छात्रों के लिए आसान हिंदी, छोटी लाइनें, कोरस और दोहराव
"""
    else:
        system_prompt = """You are given ONLY textbook content for a specific chapter.
From this text, create:
- First line: a VERY short topic heading (2-6 words, no numbering, no brackets)
- Then a blank line
- Then ONLY the study song lyrics.

STRICT rules:
- Use ONLY concepts, terms, definitions, and examples that appear in the given text
- Do NOT add any new topics, places, names, stories, or facts that are not clearly present
- If something is unclear or missing, SKIP it instead of inventing details
- Student-friendly, short lines with a small chorus
- Output ONLY: heading line + blank line + lyrics
"""

    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": safe_chunk}
        ],
        temperature=0.5,
        max_tokens=500
    )
    full = resp.choices[0].message.content.strip()

    lines = full.splitlines()
    if not lines:
        return "Unknown topic", full

    topic = lines[0].strip()
    rest = "\n".join(lines[1:]).lstrip()
    verse = rest if rest else full
    return topic, verse

# Extract text from PDF
def extract_pdf_text(file):
    text = ""
    try:
        # Try PyMuPDF first (better for scanned PDFs)
        doc = fitz.open(stream=file.read(), filetype="pdf")
        for page in doc:
            text += page.get_text()
        doc.close()
    except:
        # Fallback to PyPDF2
        file.seek(0)
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    
    # Clean up extra whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'[^\w\s\n।।।।।।]', ' ', text)
    return text.strip()

# Main app
st.title("🎵 PDF to Study Song Generator 🎵")
st.markdown("English / हिंदी PDFs के लिए काम करता है (printed + scanned). Handwritten is experimental.")

# Sidebar
with st.sidebar:
    st.header("🌐 Language Mode")
    lang_option = st.radio(
        "Select language:",
        ["🚀 Auto-detect", "🇺🇸 Force English", "🇮🇳 Force Hindi"],
        index=0
    )

# File upload
uploaded_file = st.file_uploader(
    "📁 Upload PDF",
    type="pdf",
    help="Drag and drop file here (Limit 200MB per file • PDF)"
)

if uploaded_file:
    with st.spinner("Reading PDF..."):
        raw_text = extract_pdf_text(uploaded_file)
        page_count = len(fitz.open(stream=uploaded_file.read(), filetype="pdf"))
        uploaded_file.seek(0)  # Reset for potential re-read
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 Pages", page_count)
        with col2:
            st.metric("🔤 Characters", len(raw_text))
        with col3:
            detected_lang = detect_language(raw_text[:1000])
            st.metric("🗣️ Detected", "🇺🇸 English" if detected_lang == "en" else "🇮🇳 Hindi")

    if raw_text:
        st.success("✅ PDF ready!")
        st.info(f"Detected: {'🇺🇸 English' if detected_lang == 'en' else '🇮🇳 Hindi'}")

        # Language mapping
        if lang_option == "🚀 Auto-detect":
            final_lang = "auto"
        elif lang_option == "🇺🇸 Force English":
            final_lang = "en"
        else:
            final_lang = "hi"

        # Generate button
        if st.button("🎼 Generate Study Songs", type="primary"):
            with st.spinner("Processing..."):
                chunks = chunk_text(raw_text, size=8000)
                MAX_VERSES = 40
                if len(chunks) > MAX_VERSES:
                    st.warning(f"Book is large; generating only first {MAX_VERSES} verses.")
                    chunks = chunks[:MAX_VERSES]

                st.info(f"🎼 Creating {len(chunks)} verse(s) in **{final_lang.upper() if final_lang != 'auto' else 'AUTO'}** …")

                bar = st.progress(0.0)
                status = st.empty()
                final_song = ""

                for i, chunk in enumerate(chunks):
                    status.text(f"✍️ Generating verse {i+1}/{len(chunks)} …")
                    
                    try:
                        topic, verse = make_song_with_topic(chunk, final_lang)
                        
                        final_song += (
                            f"**🧾 Topic:** {topic}\n\n"
                            f"**🎵 Verse {i+1} 🎵**\n\n"
                            f"{verse}\n\n"
                            f"---\n\n"
                        )
                    except Exception as e:
                        st.error(f"Error on verse {i+1}: {str(e)}")
                        final_song += f"**Verse {i+1}:** (Skipped due to error)\n\n---\n\n"

                    bar.progress((i + 1) / len(chunks))
                    time.sleep(0.5)  # Rate limiting

                # Display results
                st.markdown("## 🎤 Complete Study Song")
                st.markdown(final_song)

                # Download button
                st.download_button(
                    label="💾 Download Song Lyrics",
                    data=final_song,
                    file_name="study_song_lyrics.txt",
                    mime="text/plain"
                )
    else:
        st.info("👆 Upload a PDF and click 'Generate Study Songs'")
else:
    st.info("📁 Please upload a PDF file to get started!")
