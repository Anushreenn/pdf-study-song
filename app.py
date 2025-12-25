# import streamlit as st
# import os
# from pypdf import PdfReader
# from groq import Groq
# import time
# import re
# import numpy as np
# import cv2

# from pdf2image import convert_from_path
# from PIL import Image
# import pytesseract

# # ---------- Paths ----------

# SAVE_DIR = "pdf_songs"
# os.makedirs(SAVE_DIR, exist_ok=True)

# # ---------- Groq client ----------

# client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

# # Tesseract binary path (works on Colab / many Linux servers)
# pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# # ---------- Helpers ----------

# def preprocess_for_ocr(pil_img):
#     """Light preprocessing to help Tesseract on scans/handwritten."""
#     img = np.array(pil_img.convert("L"))  # grayscale
#     img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
#     img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
#     img = cv2.medianBlur(img, 3)
#     return Image.fromarray(img)

# def looks_noisy(t: str) -> bool:
#     """Return True if text looks like OCR garbage (for handwriting etc.)."""
#     t = t.strip()
#     if len(t) < 80:
#         return True
#     letters_spaces = sum(c.isalpha() or c.isspace() for c in t)
#     ratio = letters_spaces / max(len(t), 1)
#     return ratio < 0.5

# # ---------- Topic heading ----------

# def get_topic_heading(chunk: str, lang: str) -> str:
#     if lang == "hindi":
#         system_prompt = (
#             "दिए गए अध्ययन सामग्री के लिए सिर्फ़ 2-6 शब्दों का छोटा टॉपिक/शीर्षक लिखो। "
#             "पूरा वाक्य नहीं, कोई व्याख्या नहीं, सिर्फ़ शीर्षक।"
#         )
#     else:
#         system_prompt = (
#             "For the given study text, write ONLY a very short topic heading "
#             "(2-6 words). No sentence, no explanation, just the heading."
#         )

#     resp = client.chat.completions.create(
#         model="llama-3.1-8b-instant",
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": chunk[:800]},
#         ],
#         temperature=0.2,
#         max_tokens=20,
#     )
#     return resp.choices[0].message.content.strip()

# # ---------- Language detection ----------

# def detect_language(text):
#     hindi_chars = len(re.findall(r'[\u0900-\u097F]', text))
#     latin_chars = len(re.findall(r'[A-Za-z]', text))
#     total = len(text) or 1
#     if hindi_chars / total > 0.05 and hindi_chars > latin_chars:
#         return "hindi"
#     return "english"

# def chunk_text(text, size=1800):
#     return [text[i:i+size] for i in range(0, len(text), size)]

# # ---------- Song generation (ONLY from extracted text) ----------

# @st.cache_data
# def make_song(chunk, lang="auto"):
#     if lang == "auto":
#         lang = detect_language(chunk[:400])

#     safe_chunk = chunk.strip()
#     if not safe_chunk:
#         safe_chunk = "Text is almost empty and noisy; use only these few visible words:\n" + chunk[:200]

#     if lang == "hindi":
#         system_prompt = """तुम्हें नीचे दिए गए टेक्स्ट (chapter content) को ही लेकर
# एक छोटा, सरल और याद रखने लायक हिंदी स्टडी गीत बनाना है।

# सख्त नियम:
# - सिर्फ़ दिए गए टेक्स्ट में जो concepts, facts, definitions, examples हैं, वही इस्तेमाल करो
# - कोई नया example, जगह, कहानी, व्यक्ति, organization खुद से मत बनाओ
# - अगर कुछ समझ में नहीं आता, उसे छोड़ दो; अपने से कुछ मत जोड़ो
# - छात्रों के लिए आसान हिंदी, छोटी लाइनें, कोरस और दोहराव
# - आउटपुट सिर्फ़ गीत के बोल हो; explanation या "मैं नहीं कर सकता" मत लिखो
# """
#     else:
#         system_prompt = """You are given ONLY textbook content for a specific chapter.
# Turn ONLY this content into a short, simple, easy-to-memorize study song.

# STRICT rules:
# - Use ONLY concepts, terms, definitions, and examples that appear in the given text
# - Do NOT add any new topics, places, names, stories, or facts that are not clearly present
# - If something is unclear or missing, SKIP it instead of inventing details
# - Student-friendly, short lines with a small chorus
# - Output ONLY song lyrics, never explanations or meta-comments
# """

#     response = client.chat.completions.create(
#         model="llama-3.1-8b-instant",
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": safe_chunk}
#         ],
#         temperature=0.5,
#         max_tokens=450
#     )
#     return response.choices[0].message.content.strip()

# # ---------- UI ----------

# st.set_page_config(page_title="PDF Study Song (HI+EN+OCR)", layout="wide")

# st.title("🎵 PDF to Study Song Generator 🎵")
# st.markdown("**English / हिंदी PDFs के लिए काम करता है (printed + scanned). Handwritten is experimental.**")

# col1, col2 = st.columns([1, 3])

# with col1:
#     lang_mode = st.radio(
#         "🌐 Language Mode:",
#         ["🚀 Auto-detect", "🇺🇸 Force English", "🇮🇳 Force Hindi", "both english and hindi"],
#         index=0
#     )

# with col2:
#     st.info("📚 Printed / scanned textbook PDFs पर best results. Handwritten notes पर OCR हमेशा accurate नहीं होगा।")

# uploaded_file = st.file_uploader("📁 Upload PDF", type="pdf")

# if uploaded_file is not None:
#     # Local path works both in Colab and cloud
#     tmp_pdf_path = os.path.join(".", uploaded_file.name)
#     with open(tmp_pdf_path, "wb") as f:
#         f.write(uploaded_file.read())

#     noisy_pages = 0

#     with st.spinner("🔍 Reading PDF (text first, OCR for image/handwritten pages)..."):
#         reader = PdfReader(tmp_pdf_path)
#         text = ""
#         page_count = len(reader.pages)
#         prog = st.progress(0.0)

#         for i, page in enumerate(reader.pages):
#             t = page.extract_text() or ""
#             clean = t.strip()

#             # If almost no text, assume scan / handwritten -> OCR
#             if len(clean) < 40:
#                 images = convert_from_path(
#                     tmp_pdf_path, dpi=300, first_page=i+1, last_page=i+1
#                 )
#                 img = preprocess_for_ocr(images[0])
#                 t = pytesseract.image_to_string(img, lang="hin+eng")
#                 if looks_noisy(t):
#                     noisy_pages += 1

#             text += t + "\n"
#             prog.progress((i + 1) / page_count)

#     if noisy_pages > page_count * 0.6:
#         st.error("❌ Most pages look like unreadable handwriting / noisy OCR. Songs would be nonsense. Try a clearer scan or typed PDF.")
#         st.stop()

#     detected_lang = detect_language(text[:2000])
#     lang_display = "🇮🇳 Hindi" if detected_lang == "hindi" else "🇺🇸 English"

#     m1, m2, m3 = st.columns(3)
#     with m1:
#         st.metric("📄 Pages", page_count)
#     with m2:
#         st.metric("🔤 Characters", len(text))
#     with m3:
#         st.metric("🗣️ Detected", lang_display)

#     if noisy_pages:
#         st.warning(f"⚠️ {noisy_pages} page(s) had very noisy OCR (likely handwritten / bad scan). Output may be imperfect.")

#     st.success(f"✅ PDF ready! Detected: **{lang_display}**")

#     if st.button("🎶 Generate Study Song", type="primary", use_container_width=True):
#         if lang_mode == "🚀 Auto-detect":
#             final_lang = detected_lang
#         elif "Hindi" in lang_mode:
#             final_lang = "hindi"
#         else:
#             final_lang = "english"

#         chunks = chunk_text(text)
#         st.info(f"🎼 Creating {len(chunks)} verse(s) in **{final_lang.upper()}** …")

#         bar = st.progress(0.0)
#         status = st.empty()
#         final_song = ""

#         for i, chunk in enumerate(chunks):
#             status.text(f"✍️ Generating verse {i+1}/{len(chunks)} …")

#             topic = get_topic_heading(chunk, final_lang)
#             verse = make_song(chunk, final_lang)

#             final_song += (
#                 f"**🧾 Topic:** {topic}\n\n"
#                 f"**🎵 Verse {i+1} 🎵**\n\n"
#                 f"{verse}\n\n---\n\n"
#             )

#             bar.progress((i + 1) / len(chunks))
#             time.sleep(0.2)

#         st.subheader("🎤 Your Complete Study Song")
#         st.markdown(final_song)

#         fname = uploaded_file.name.replace(".pdf", f"_{final_lang}_study_song.txt")
#         out_path = os.path.join(SAVE_DIR, fname)
#         with open(out_path, "w", encoding="utf-8") as f:
#             f.write(final_song)

#         st.success(f"💾 Saved at: `{out_path}`")
#         st.download_button(
#             "📥 Download Song",
#             data=final_song,
#             file_name=fname,
#             mime="text/plain",
#             use_container_width=True
#         )



import streamlit as st
import os
from pypdf import PdfReader
from groq import Groq
import time
import re
import numpy as np
import cv2

from pdf2image import convert_from_path
from PIL import Image
import pytesseract

# ---------- Paths ----------

SAVE_DIR = "pdf_songs"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------- Groq client ----------

client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))


# Tesseract binary path (works on Colab / many Linux servers)
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# ---------- Helpers ----------

def preprocess_for_ocr(pil_img):
    """Light preprocessing to help Tesseract on scans/handwritten."""
    img = np.array(pil_img.convert("L"))  # grayscale
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    img = cv2.medianBlur(img, 3)
    return Image.fromarray(img)

def looks_noisy(t: str) -> bool:
    """Return True if text looks like OCR garbage (for handwriting etc.)."""
    t = t.strip()
    if len(t) < 80:
        return True
    letters_spaces = sum(c.isalpha() or c.isspace() for c in t)
    ratio = letters_spaces / max(len(t), 1)
    return ratio < 0.5

# ---------- Language detection ----------

def detect_language(text):
    hindi_chars = len(re.findall(r'[\u0900-\u097F]', text))
    latin_chars = len(re.findall(r'[A-Za-z]', text))
    total = len(text) or 1
    if hindi_chars / total > 0.05 and hindi_chars > latin_chars:
        return "hindi"
    return "english"

def chunk_text(text, size=8000):
    return [text[i:i+size] for i in range(0, len(text), size)]

# ---------- Song + topic generation (ONLY from extracted text) ----------

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

# ---------- UI ----------

st.set_page_config(page_title="PDF Study Song (HI+EN+OCR)", layout="wide")

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

uploaded_file = st.file_uploader("📁 Upload PDF", type="pdf")

if uploaded_file is not None:
    tmp_pdf_path = os.path.join(".", uploaded_file.name)
    with open(tmp_pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    noisy_pages = 0

    with st.spinner("🔍 Reading PDF (text first, OCR for image/handwritten pages)..."):
        reader = PdfReader(tmp_pdf_path)
        text = ""
        page_count = len(reader.pages)
        prog = st.progress(0.0)

        for i, page in enumerate(reader.pages):
            t = page.extract_text() or ""
            clean = t.strip()

            if len(clean) < 40:
                images = convert_from_path(
                    tmp_pdf_path, dpi=300, first_page=i+1, last_page=i+1
                )
                img = preprocess_for_ocr(images[0])
                t = pytesseract.image_to_string(img, lang="hin+eng")
                if looks_noisy(t):
                    noisy_pages += 1

            text += t + "\n"
            prog.progress((i + 1) / page_count)

    if noisy_pages > page_count * 0.6:
        st.error("❌ Most pages look like unreadable handwriting / noisy OCR. Songs would be nonsense. Try a clearer scan or typed PDF.")
        st.stop()

    detected_lang = detect_language(text[:2000])
    lang_display = "🇮🇳 Hindi" if detected_lang == "hindi" else "🇺🇸 English"

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("📄 Pages", page_count)
    with m2:
        st.metric("🔤 Characters", len(text))
    with m3:
        st.metric("🗣️ Detected", lang_display)

    if noisy_pages:
        st.warning(f"⚠️ {noisy_pages} page(s) had very noisy OCR (likely handwritten / bad scan). Output may be imperfect.")

    st.success(f"✅ PDF ready! Detected: **{lang_display}**")

    if st.button("🎶 Generate Study Song", type="primary", use_container_width=True):
        if lang_mode == "🚀 Auto-detect":
            final_lang = detected_lang
        elif "Hindi" in lang_mode:
            final_lang = "hindi"
        else:
            final_lang = "english"

        chunks = chunk_text(text, size=8000)
        MAX_VERSES = 40
        if len(chunks) > MAX_VERSES:
            st.warning(f"Book is large; generating only first {MAX_VERSES} verses.")
            chunks = chunks[:MAX_VERSES]

        st.info(f"🎼 Creating {len(chunks)} verse(s) in **{final_lang.upper()}** …")

        bar = st.progress(0.0)
        status = st.empty()
        final_song = ""

        for i, chunk in enumerate(chunks):
            status.text(f"✍️ Generating verse {i+1}/{len(chunks)} …")

            topic, verse = make_song_with_topic(chunk, final_lang)

            final_song += (
                f"**🧾 Topic:** {topic}\n\n"
                f"**🎵 Verse {i+1} 🎵**\n\n"
                f"{verse}\n\n---\n\n"
            )

            bar.progress((i + 1) / len(chunks))
            time.sleep(0.2)

        st.subheader("🎤 Your Complete Study Song")
        st.markdown(final_song)

        fname = uploaded_file.name.replace(".pdf", f"_{final_lang}_study_song.txt")
        out_path = os.path.join(SAVE_DIR, fname)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(final_song)

        st.success(f"💾 Saved at: `{out_path}`")
        st.download_button(
            "📥 Download Song",
            data=final_song,
            file_name=fname,
            mime="text/plain",
            use_container_width=True
        )
