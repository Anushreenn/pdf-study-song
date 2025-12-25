import streamlit as st
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

# Chunk text into larger pieces
def chunk_text(text, size=20000):
    return [text[i:i+size] for i in range(0, len(text), size)]

# Combined function: topic + song verse
@st.cache_data
def make_song_with_topic(chunk, lang="auto"):
    if lang == "auto":
        lang = detect_language(chunk[:400])

    safe_chunk = chunk.strip()
    if not safe_chunk:
        safe_chunk = "Text is almost empty and noisy; use only these few visible words:\n" + chunk[:200]

    if lang == "hindi" or lang == "hi":
        system_prompt = """तुम्हें नीचे दिए गए टेक्स्ट (chapter content) से ही
एक छोटा, सरल और याद रखने लायक हिंदी स्टडी गीत बनाना है।

आउटपुट फॉर्मेट:
पहली लाइन: केवल 2-6 शब्दों का छोटा टॉपिक/शीर्षक (कोई नंबर, कोई ब्रैकेट नहीं)
फिर एक खाली लाइन
फिर केवल गीत के बोल।

सख्त नियम:
- सिर्फ़ दिए गए टेक्स्ट में जो concepts, facts, definitions, examples हैं, वही इस्तेमाल करो
"""
    else:
        system_prompt = """You are given ONLY textbook content for a specific chapter.
From this text, create:
- First line: a VERY short topic heading (2-6 words, no numbering, no brackets)
- Then a blank line
- Then ONLY the study song lyrics.

STRICT rules:
- Use ONLY concepts, terms, definitions, and examples that appear in the given text
- Output ONLY: heading line + blank line + lyrics
"""

    try:
        resp = client.chat.completions.create(
            model="llama3-8b-8192",  # Cheaper + unlimited
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": safe_chunk}
            ],
            temperature=0.5,
            max_tokens=400
        )
        full = resp.choices[0].message.content.strip()

        lines = full.splitlines()
        if not lines:
            return "Unknown topic", full

        topic = lines[0].strip()
        rest = "\n".join(lines[1:]).lstrip()
        verse = rest if rest else full
        return topic, verse
    except Exception as e:
        if "429" in str(e):
            return "Rate Limited", "⏳ Daily limit reached - upgrade Groq!"
        return "Error", f"Generation failed: {str(e)[:50]}"

# Extract text from PDF
def extract_pdf_text(file):
    file.seek(0)
    text = ""
    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        for page in doc:
            text += page.get_text()
        doc.close()
    except:
        text = ""
    
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'[^\w\s\n।।।।।।]', ' ', text)
    return text.strip()

# Main app
st.title("🎵 PDF to Study Song Generator 🎵")
st.markdown("**Upload 1-4 PDFs at once** 👇 English / हिंदी PDFs (printed + scanned)")

# Sidebar
with st.sidebar:
    st.header("🌐 Language Mode")
    lang_option = st.radio(
        "Select language:",
        ["🚀 Auto-detect", "🇺🇸 Force English", "🇮🇳 Force Hindi"],
        index=0
    )
    st.info("💡 **Pro tip:** Larger chunks = fewer API calls = no rate limits!")

# Multi-file upload
uploaded_files = st.file_uploader(
    "📚 Upload PDFs (1-4 files)",
    type="pdf",
    accept_multiple_files=True,
    help="Drag & drop multiple PDFs (Max 200MB each)"
)

if uploaded_files:
    pdf_data = []
    
    # Process each PDF
    for uploaded_file in uploaded_files:
        with st.spinner(f"Reading {uploaded_file.name}..."):
            raw_text = extract_pdf_text(uploaded_file)
            
            # Reset for page count
            uploaded_file.seek(0)
            try:
                doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                page_count = len(doc)
                doc.close()
            except:
                page_count = "Unknown"
            
            detected_lang = detect_language(raw_text[:1000]) if raw_text else "en"
            
            pdf_data.append({
                'name': uploaded_file.name,
                'text': raw_text,
                'pages': page_count,
                'chars': len(raw_text),
                'lang': detected_lang
            })
    
    # Display PDF stats
    st.subheader("📊 PDF Summary")
    for pdf in pdf_data:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"📄 {pdf['name']}", pdf['pages'])
        with col2:
            st.metric("🔤 Chars", pdf['chars'])
        with col3:
            st.metric("🗣️ Lang", "🇺🇸 English" if pdf['lang'] == "en" else "🇮🇳 Hindi")

    total_chars = sum(pdf['chars'] for pdf in pdf_data)
    st.info(f"**Total: {len(pdf_data)} PDFs, {total_chars:,} chars**")

    # Language mapping
    if lang_option == "🚀 Auto-detect":
        final_lang = "auto"
    elif lang_option == "🇺🇸 Force English":
        final_lang = "en"
    else:
        final_lang = "hi"

    # Generate button
    if st.button("🎼 Generate Songs from ALL PDFs", type="primary"):
        all_songs = ""
        total_progress = st.progress(0.0)
        status = st.empty()
        
        for pdf_idx, pdf in enumerate(pdf_data):
            if not pdf['text']:
                continue
                
            st.markdown(f"### 📖 {pdf['name']} ({pdf['pages']} pages)")
            
            chunks = chunk_text(pdf['text'], size=25000)  # HUGE chunks
            MAX_VERSES = 12  # Per PDF
            
            if len(chunks) > MAX_VERSES:
                st.warning(f"📚 {pdf['name']}: First {MAX_VERSES} chapters only")
                chunks = chunks[:MAX_VERSES]
            
            pdf_progress = st.progress(0.0)
            
            for i, chunk in enumerate(chunks):
                status.text(f"✍️ {pdf['name']} - Verse {i+1}/{len(chunks)}")
                
                topic, verse = make_song_with_topic(chunk, final_lang)
                
                all_songs += (
                    f"**📚 {pdf['name']}**\n"
                    f"**🧾 Topic:** {topic}\n\n"
                    f"**🎵 Verse {i+1} 🎵**\n\n"
                    f"{verse}\n\n"
                    f"---\n\n"
                )
                
                pdf_progress.progress((i + 1) / len(chunks))
                time.sleep(0.8)
            
            total_progress.progress((pdf_idx + 1) / len(pdf_data))
        
        # Final results
        st.markdown("## 🎤 Complete Study Songs")
        st.markdown(all_songs)
        
        st.success(f"✅ Generated songs from {len(pdf_data)} PDFs!")
        st.download_button(
            label="💾 Download All Songs",
            data=all_songs,
            file_name="multi_pdf_study_songs.txt",
            mime="text/plain"
        )

else:
    st.info("📚 Upload 1-4 PDFs to generate study songs from all at once!")
