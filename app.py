# app.py
import streamlit as st
import os, time, re
from pypdf import PdfReader
from groq import Groq
from pdf2image import convert_from_path
from PIL import Image
import pytesseract

# ------------------------------------------------
# 🔐 GROQ API - from Streamlit Secrets (Cloud Safe)
# ------------------------------------------------
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

SAVE_DIR = "pdf_songs"
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------------------------------------
# 🛡️ Rate Limit Safe Call
# ------------------------------------------------
def safe_groq(messages, max_tokens=400):
    for a in range(4):
        try:
            r = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                temperature=0.4,
                messages=messages,
                max_tokens=max_tokens,
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            if "rate" in str(e).lower():
                t = 2 * (a + 1)
                st.warning(f"⏳ Groq Rate Limit. Retrying in {t}s...")
                time.sleep(t)
                continue
            return f"❌ Error: {e}"
    return "❌ Server busy. Try again later."

# ------------------------------------------------
# 🔤 Language Detection
# ------------------------------------------------
def detect_lang(text):
    hi = len(re.findall(r'[\u0900-\u097F]', text))
    en = len(re.findall(r'[A-Za-z]', text))
    return "hindi" if hi > en else "english"

# ------------------------------------------------
# 📄 Split Text
# ------------------------------------------------
def chunk_text(t, n=1800):
    return [t[i:i+n] for i in range(0, len(t), n)]

# ------------------------------------------------
# 🎵 Topic Title
# ------------------------------------------------
def topic(chunk, lang):
    sys = "2-5 words heading" if lang == "english" else "2-5 शब्द का शीर्षक"
    msgs = [
        {"role": "system", "content": sys},
        {"role": "user", "content": chunk[:600]}
    ]
    return safe_groq(msgs, 30)

# ------------------------------------------------
# 🎶 Song Generation
# ------------------------------------------------
def make_song(chunk, lang):
    if lang == "hindi":
        sys = "ONLY use Hindi text to make a simple study song. No new info."
    else:
        sys = "ONLY use English text to make a simple study song. No new info."
    
    msgs = [
        {"role": "system", "content": sys},
        {"role": "user", "content": chunk}
    ]
    return safe_groq(msgs, 450)

# ------------------------------------------------
# UI
# ------------------------------------------------
st.set_page_config(page_title="PDF Study Song Generator", layout="wide")
st.title("🎵 PDF → Study Song Generator")
st.caption("📄 Works for scanned and normal PDFs (OCR).")

mode = st.radio("Language Mode:", ["Auto", "English", "Hindi"], horizontal=True)
file = st.file_uploader("📁 Upload PDF", type="pdf")

if file:
    name = file.name
    with open(name, "wb") as f: f.write(file.read())

    st.info("🔍 Reading PDF + OCR if needed...")
    
    reader = PdfReader(name)
    full_text = ""
    bad = 0
    pb = st.progress(0)

    for i, p in enumerate(reader.pages):
        txt = (p.extract_text() or "").strip()

        # 🧠 OCR if no text
        if len(txt) < 40:
            try:
                img = convert_from_path(name, dpi=300, first_page=i+1, last_page=i+1)[0]
                txt = pytesseract.image_to_string(img, lang="hin+eng")
            except:
                txt = ""
        if len(txt) < 20: bad += 1

        full_text += txt + "\n"
        pb.progress((i+1)/len(reader.pages))

    auto = detect_lang(full_text[:1500])
    final = "english" if mode=="English" else "hindi" if mode=="Hindi" else auto
    st.success(f"🎯 Detected: {auto.upper()} → Output: {final.upper()}")

    if bad:
        st.warning(f"⚠️ {bad} page(s) unreadable / handwritten")

    if st.button("🎶 Generate Song"):
        parts = chunk_text(full_text)
        st.info(f"✍️ Making {len(parts)} verse(s)...")
        bar = st.progress(0)
        out = ""

        for i, c in enumerate(parts):
            t = topic(c, final)
            s = make_song(c, final)
            out += f"## {t}\n🎵 Verse {i+1}\n\n{s}\n\n---\n\n"
            bar.progress((i+1)/len(parts))

        st.subheader("🎤 Final Song")
        st.markdown(out)

        fn = name.replace(".pdf", f"_{final}.txt")
        with open(os.path.join(SAVE_DIR, fn), "w", encoding="utf-8") as f:
            f.write(out)

        st.download_button("📥 Download Song", out, fn)
        st.balloons()
