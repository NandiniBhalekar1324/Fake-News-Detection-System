import streamlit as st
import joblib
import re
import time
import pandas as pd
from datetime import datetime
import os

# -------------------------
# Config
# -------------------------
st.set_page_config(page_title="Bluff — Fake News Detection", layout="wide", page_icon="❌")

# -------------------------
# History CSV
# -------------------------
HISTORY_CSV = "history.csv"
if not os.path.exists(HISTORY_CSV):
    pd.DataFrame(columns=["timestamp", "text_snippet", "label", "confidence"]).to_csv(HISTORY_CSV, index=False)

# -------------------------
# Load model (cached)
# -------------------------
@st.cache_resource
def load_resources():
    model = joblib.load("random_forest_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_resources()

# -------------------------
# Helpers
# -------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def predict_news(news_text):
    cleaned = clean_text(news_text)
    transformed = vectorizer.transform([cleaned])
    proba = model.predict_proba(transformed)[0]
    prediction = model.predict(transformed)[0]
    confidence = proba[1] if prediction == "TRUE" else proba[0]
    return "TRUE" if prediction == "TRUE" else "FAKE", float(confidence)

def history_count():
    try:
        df = pd.read_csv(HISTORY_CSV)
        return len(df)
    except Exception:
        return 0

# -------------------------
# Styling & CSS improvements
# -------------------------
st.markdown(
    """
    <style>
    /* Page */
    .stApp {
        background: radial-gradient(circle at 10% 10%, rgba(86,26,139,0.08), transparent 8%),
                    linear-gradient(180deg, #05050a 0%, #020205 100%);
        color: #dbe9ff;
        font-family: Inter, sans-serif;
    }

    /* Main card */
    .main-card {
        background: linear-gradient(135deg, rgba(18,10,30,0.6), rgba(8,8,18,0.62));
        border: 1px solid rgba(77,166,255,0.06);
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.5);
        backdrop-filter: blur(6px) saturate(120%);
    }

    /* Header / small logo */
    .logo-wrapper { text-align:center; padding:10px 0 12px 0; }
    .neon-title {
        font-size: 24px;
        font-weight: 900;
        line-height: 1.05;
        letter-spacing: 1.6px;
        font-family: 'Audiowide', sans-serif;
        display:inline-block;
    }
    .neon-title.blue {
        color: #00eaff;
        text-shadow:
            0 0 5px #00eaff,
            0 0 12px #00eaff,
            0 0 22px #00eaff,
            0 0 36px #0077ff;
    }

    /* Sidebar neon accent and active item highlight */
    .sidebar .block-container {
        padding-top: 8px;
    }
    .sidebar .stRadio > div {
        margin-top: 10px;
    }
    /* left neon border for sidebar */
    .css-1d391kg { /* container class may vary between Streamlit versions; using fallback below too */
        border-left: 4px solid rgba(0,234,255,0.06);
        box-shadow: inset -6px 0 48px rgba(0,234,255,0.02);
    }
    /* fallback wrapper style to ensure accent */
    .sidebar .element-container:nth-child(1) {
        border-left: 4px solid rgba(0,234,255,0.06);
    }

    /* active radio (highlight row) - depends on Streamlit DOM but this adds a visible effect to selected item */
    .stRadio .stRadio > div[role="radiogroup"] > label[aria-checked="true"], 
    .stRadio label[aria-checked="true"] {
        background: linear-gradient(90deg, rgba(0,234,255,0.03), rgba(107,45,255,0.02));
        border-radius: 8px;
        padding-left: 8px;
        padding-right: 8px;
        box-shadow: 0 6px 20px rgba(0,234,255,0.03);
        color: #dffaff;
    }

    /* nav icons color */
    .stSidebar .stButton>button, .stSidebar .stMarkdown {
        color: #cfeeff;
    }

    /* Neon buttons */
    .neon-btn button {
        background: linear-gradient(90deg,#2db6ff,#6b2bff) !important;
        color: #fff !important;
        font-weight: 800 !important;
        padding: 10px 22px !important;
        border-radius: 10px !important;
        box-shadow: 0 8px 34px rgba(45,182,255,0.12), 0 0 28px rgba(107,45,255,0.06);
    }

    /* small stat cards */
    .stat-card {
        background: linear-gradient(180deg, rgba(14,12,28,0.6), rgba(8,8,18,0.5));
        border-radius: 10px;
        padding: 12px;
        border: 1px solid rgba(77,166,255,0.06);
        text-align: center;
    }
    .stat-number { font-weight:800; color:#a7f3ff; font-size:20px; }
    .stat-label { color:#9fbde6; font-size:12px; }

    /* Live feed placeholder cards */
    .placeholder-card {
        background: linear-gradient(90deg, rgba(30,10,40,0.6), rgba(8,8,18,0.5));
        border-radius: 10px;
        padding: 18px;
        border: 1px solid rgba(255,255,255,0.02);
        box-shadow: 0 8px 30px rgba(0,0,0,0.5);
        color:#dfefff;
    }

    /* footer neon strip */
    .footer-strip {
        height: 4px;
        background: linear-gradient(90deg, rgba(0,234,255,0.25), rgba(107,45,255,0.20));
        border-radius: 4px;
        margin-top: 14px;
        margin-bottom: 6px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Sidebar (smaller neon logo + nav)
# -------------------------
st.sidebar.markdown("<div style='padding:8px 6px 4px 6px;'>", unsafe_allow_html=True)

st.sidebar.markdown("""
<div class="logo-wrapper">
    <span class="neon-title blue">TRUTH<br>SCANNER 2.0</span>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<h3 style='color:#67e8ff;margin:6px 0 2px 0;font-weight:700;'>Bluff</h3>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='color:#9fbde6;margin-top:-6px'>Cyberpunk Fake News</div>", unsafe_allow_html=True)
st.sidebar.markdown("---", unsafe_allow_html=True)

# improved icon choices and order
menu = st.sidebar.radio("Navigation", ["🏡 Home", "🔎 Detector", "📡 Live Feed", "📘 Project Info", "🕒 History"])

st.sidebar.markdown("---", unsafe_allow_html=True)
st.sidebar.markdown("<div style='color:#94a3b8;font-size:13px'>Developed by Nandini Raju Bhalekar</div>", unsafe_allow_html=True)
st.sidebar.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Pages (with improvements)
# -------------------------
def page_home():
    st.markdown('<div class="header main-card">', unsafe_allow_html=True)
    st.markdown(f'<div style="display:flex; align-items:center; justify-content:space-between;"><div><h2 style="margin:0;color:#67e8ff">BLUFF</h2><div style="color:#9fbde6">Neon-assisted Fake News Detection</div></div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="main-card" style="margin-top:14px">', unsafe_allow_html=True)
    st.markdown('<div style="display:flex; gap:14px; align-items:stretch">', unsafe_allow_html=True)

    # stats
    col1, col2, col3, col4 = st.columns([1,1,1,1])
    with col1:
        st.markdown('<div class="stat-card"><div class="stat-number">Random Forest</div><div class="stat-label">Model</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stat-card"><div class="stat-number">96.5%</div><div class="stat-label">Accuracy (example)</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{history_count()}</div><div class="stat-label">Analyses (history)</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="stat-card"><div class="stat-number">~0.6s</div><div class="stat-label">Avg latency</div></div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-card"><b>Quick Start</b><br>Paste an article in Detector → click <b>ANALYZE ARTICLE</b>. Use History to export results.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def page_detector():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown('<div style="padding:6px 0 6px 0"><label style="color:#cfeeff;font-weight:700;">Paste News Article</label></div>', unsafe_allow_html=True)
        user_text = st.text_area("", value="", key="text_area", height=300,
                                 placeholder="Paste full article or paragraph...",
                                 label_visibility="collapsed")
        st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
        analyze_clicked = st.button("ANALYZE ARTICLE", key="analyze_btn")
    with col2:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<div style="color:#9fbde6;font-weight:700">Quick Scanner</div>', unsafe_allow_html=True)
        st.markdown('<div class="scan"></div>', unsafe_allow_html=True)
        st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
        st.markdown('<div style="color:#94a3b8">Model: RandomForest • Vectorizer: TF-IDF</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    if analyze_clicked:
        if not user_text or len(user_text.strip()) < 20:
            st.warning("Text is too short. Add more content.")
        else:
            with st.spinner("Scanning article..."):
                time.sleep(1.0)
            label, conf = predict_news(user_text)
            conf_pct = round(conf * 100, 2)
            if label == "TRUE":
                st.markdown('<div class="true" style="font-size:20px; font-weight:800">✔ VERIFIED — TRUE</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="fake" style="font-size:20px; font-weight:800">✖ FAKE NEWS</div>', unsafe_allow_html=True)
            st.markdown(f"<h3 style='text-align:center;color:#a8d6ff;margin-top:8px'>{conf_pct}% confidence</h3>", unsafe_allow_html=True)

            # Save history
            snippet = (user_text[:300] + "...") if len(user_text) > 300 else user_text
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            try:
                df = pd.read_csv(HISTORY_CSV)
            except Exception:
                df = pd.DataFrame(columns=["timestamp", "text_snippet", "label", "confidence"])
            df.loc[len(df)] = [timestamp, snippet, label, conf]
            df.to_csv(HISTORY_CSV, index=False)

def page_live_feed():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown('<h2 style="color:#67e8ff">Live Feed (Coming Soon)</h2>', unsafe_allow_html=True)
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    # show 3 placeholder cards so page feels alive
    c1, c2, c3 = st.columns(3)
    for c in (c1, c2, c3):
        c.markdown('<div class="placeholder-card"><div style="font-weight:800;color:#a8d6ff">Headline — Loading</div><div style="color:#9fbde6;margin-top:8px">This slot will show live headlines and detection status.</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def page_project_info():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown("### 📘 Project Info", unsafe_allow_html=True)
    st.markdown("""
    ### 🔧 Overview
    Bluff detects fake news using machine learning and a futuristic cyberpunk UI.

    ### 🧠 Model Pipeline
    - **Model:** Random Forest
    - **Vectorizer:** TF-IDF
    - **Preprocessing:** lowercasing, punctuation removal, URL removal, whitespace normalization
    - **Output:** TRUE or FAKE + confidence score

    ### 🧪 Why Random Forest?
    - Strong performance on sparse TF-IDF vectors
    - Handles noisy text well
    - Good generalization and stability

    ### 🗃️ Storage
    - User analysis history saved in: `history.csv`

    ### 💻 Tech Stack
    - Frontend: Streamlit
    - Backend: Python, scikit-learn, joblib
    - UI: Custom Cyberpunk Neon CSS

    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def page_history():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown("### 🕒 Analysis History", unsafe_allow_html=True)
    try:
        df = pd.read_csv(HISTORY_CSV)
    except Exception:
        df = pd.DataFrame(columns=["timestamp", "text_snippet", "label", "confidence"])
    col_a, col_b = st.columns([3,1])
    with col_b:
        if st.button("Download CSV"):
            csv_bytes = df.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download history.csv", data=csv_bytes, file_name="history.csv", mime="text/csv")
        if st.button("Clear History"):
            pd.DataFrame(columns=["timestamp", "text_snippet", "label", "confidence"]).to_csv(HISTORY_CSV, index=False)
            st.success("History cleared.")
            df = pd.read_csv(HISTORY_CSV)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if df.empty:
        st.info("No history yet. Run analyses in Detector to populate history.")
    else:
        st.dataframe(df.sort_values(by="timestamp", ascending=False).head(300), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Router
if menu == "🏡 Home":
    page_home()
elif menu == "🔎 Detector":
    page_detector()
elif menu == "📡 Live Feed":
    page_live_feed()
elif menu == "📘 Project Info":
    page_project_info()
elif menu == "🕒 History":
    page_history()

# Footer + neon strip
st.markdown('<div class="footer-strip"></div>', unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:#8aa6d6;margin-top:8px'>Developed by Nandini Raju Bhalekar — Bluff</div>", unsafe_allow_html=True)




import os
import requests
import streamlit as st

NEWSAPI_KEY = "8bd084d11f16444c97a8b35f930541d0"


# ------------ NewsAPI fetcher (cached with ttl 5 minutes) ------------
@st.cache_data(ttl=300)
def fetch_newsapi(topic: str = "technology", page_size: int = 6):
    # Try Streamlit secrets first, then environment var
    api_key = None
    try:
        api_key = st.secrets.get("NEWSAPI_KEY")  # works on Streamlit sharing
    except Exception:
        api_key = None

    if not api_key:
        api_key = os.getenv("NEWSAPI_KEY", "")

    if not api_key:
        return {"error": "NEWSAPI_KEY not set. Set environment variable or st.secrets."}

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": topic,
        "pageSize": page_size,
        "language": "en",
        "sortBy": "publishedAt",
        "apiKey": api_key
    }

    try:
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        # handle cases where API returns error info
        if data.get("status") != "ok":
            return {"error": data.get("message", "Unknown NewsAPI error")}
        return {"articles": data.get("articles", [])}
    except Exception as e:
        return {"error": str(e)}


# ------------ UI: replace your previous page_live_feed with this ------------
def page_live_feed():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown('<h2 style="color:#67e8ff">Live Feed</h2>', unsafe_allow_html=True)

    # topic input + refresh
    col_t, col_refresh = st.columns([4,1])
    with col_t:
        topic = st.text_input("Topic to search", "technology")
    with col_refresh:
        if st.button("Refresh"):
            # clear cached fetch so fresh headlines load immediately
            try:
                st.cache_data.clear()
            except Exception:
                pass
            st.experimental_rerun()

    stub = st.empty()  # placeholder while loading
    with stub.container():
        stub.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        stub.info("Fetching live headlines...")

    # fetch articles
    res = fetch_newsapi(topic=topic, page_size=6)

    # clear placeholder
    stub.empty()

    if "error" in res:
        st.error(f"News fetch error: {res['error']}")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    articles = res["articles"]

    if not articles:
        st.info("No articles found. Try another topic or refresh.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    cols = st.columns(3)
    for i, art in enumerate(articles):
        col = cols[i % 3]
        title = art.get("title", "No title")
        source = art.get("source", {}).get("name", "Unknown")
        desc = art.get("description") or art.get("content") or ""
        url = art.get("url", "#")
        published = art.get("publishedAt", "")[:16]

        # run your model on description (or title if empty)
        label, conf = predict_news(desc or title)
        conf_pct = round(conf * 100, 1)
        badge_color = "#6bffdf" if label == "TRUE" else "#ff6b7a"

        with col:
            st.markdown(
                f"""
                <div class="placeholder-card">
                  <div style="font-weight:800;color:#a8d6ff">{title}</div>
                  <div style="color:#9fbde6;margin-top:8px">{source} • {published}</div>
                  <div style="height:8px"></div>
                  <div style="font-weight:700;color:{badge_color}">{label} — {conf_pct}%</div>
                  <div style="height:8px"></div>
                  <a href="{url}" target="_blank">Open source</a>
                </div>
                """, unsafe_allow_html=True
            )

    st.markdown("</div>", unsafe_allow_html=True)
