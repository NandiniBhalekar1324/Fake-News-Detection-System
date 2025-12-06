import os
import re
import time
import joblib
import requests
import pandas as pd
import streamlit as st
from datetime import datetime

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Bluff — Fake News Detection",
    layout="wide",
    page_icon="❌"
)

# --------------------------------------------------
# ASSETS FOLDER
# --------------------------------------------------
ASSETS = "assets"

# --------------------------------------------------
# HISTORY CSV
# --------------------------------------------------
HISTORY_CSV = "history.csv"
if not os.path.exists(HISTORY_CSV):
    pd.DataFrame(columns=["timestamp", "text_snippet", "label", "confidence"]).to_csv(HISTORY_CSV, index=False)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_resources():
    model = joblib.load("random_forest_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_resources()

# --------------------------------------------------
# TEXT CLEANING + PREDICTION
# --------------------------------------------------
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
    return prediction, float(confidence)

def history_count():
    try:
        df = pd.read_csv(HISTORY_CSV)
        return len(df)
    except:
        return 0

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0a16, #0e0e20);
    box-shadow: inset -5px 0px 40px rgba(0,255,255,0.08);
}

.stSidebar, .stSidebar div, .stSidebar label, .stSidebar span {
    color: #bfeaff !important;
}

.stSidebar [aria-checked="true"] {
    background: rgba(0,255,255,0.15) !important;
    border: 1px solid rgba(0,255,255,0.3);
    box-shadow: 0 0 8px rgba(0,255,255,0.3);
}

.stApp {
    background: radial-gradient(circle at 10% 10%, rgba(86,26,139,0.08), transparent 8%),
                linear-gradient(180deg, #05050a 0%, #020205 100%);
    color: #dbe9ff;
    font-family: Inter, sans-serif;
}

.main-card {
    background: linear-gradient(135deg, rgba(18,10,30,0.6), rgba(8,8,18,0.62));
    border: 1px solid rgba(77,166,255,0.06);
    border-radius: 12px;
    padding: 18px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.5);
    backdrop-filter: blur(6px) saturate(120%);
}

.stat-card {
    background: linear-gradient(180deg, rgba(14,12,28,0.6), rgba(8,8,18,0.55));
    border-radius: 10px;
    padding: 12px;
    border: 1px solid rgba(77,166,255,0.06);
    text-align: center;
}

.stat-number { font-weight:800; color:#a7f3ff; font-size:20px; }
.stat-label  { color:#9fbde6; font-size:12px; }

.footer-strip {
    height: 4px;
    background: linear-gradient(90deg, rgba(0,234,255,0.25), rgba(107,45,255,0.20));
    border-radius: 4px;
    margin-top: 14px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
# replaced deprecated use_container_width / width=True with width="stretch"
st.sidebar.image(os.path.join(ASSETS, "bluff_logo.png"), width="stretch")

st.sidebar.markdown("---")

menu = st.sidebar.radio("Navigation", ["🏡 Home", "🔎 Detector", "📡 Live Feed", "📘 Project Info", "🕒 History"])

st.sidebar.markdown("---")

# --------------------------------------------------
# HOME
# --------------------------------------------------
def page_home():
    st.markdown("""
        <h1 style='
            text-align:center;
            color:#00eaff;
            font-size:42px;
            font-weight:900;
            margin-top:-35px;
            text-shadow:
                0 0 8px #00eaff,
                0 0 16px #00aaff,
                0 0 30px #0088ff,
                0 0 45px rgba(0,150,255,0.6);
            font-family: "Audiowide", sans-serif;
        '>
            WELCOME TO BLUFF
        </h1>

        <p style='
            text-align:center;
            color:#9bdcff;
            margin-top:-8px;
            font-size:18px;
            text-shadow:0 0 10px rgba(0,200,255,0.35);
        '>
            Fake News Detection Engine
        </p>
    """, unsafe_allow_html=True)

    banner_path = os.path.join(ASSETS, "banner.gif")
    banner_mp4 = os.path.join(ASSETS, "banner.mp4")

    st.markdown("<div style='text-align:center; margin-top:15px;'>", unsafe_allow_html=True)

    if os.path.exists(banner_path):
        st.image(banner_path, width=1100)
    elif os.path.exists(banner_mp4):
        st.video(banner_mp4)
    else:
        st.warning("Banner not found (banner.gif or banner.mp4)")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="main-card" style="margin-top:18px">', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.markdown('<div class="stat-card"><div class="stat-number">Random Forest</div><div class="stat-label">Model</div></div>', unsafe_allow_html=True)
    col2.markdown('<div class="stat-card"><div class="stat-number">96.5%</div><div class="stat-label">Accuracy</div></div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="stat-card"><div class="stat-number">{history_count()}</div><div class="stat-label">Analyses</div></div>', unsafe_allow_html=True)
    col4.markdown('<div class="stat-card"><div class="stat-number">~0.6s</div><div class="stat-label">Latency</div></div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# DETECTOR PAGE
# --------------------------------------------------
def page_detector():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)

    col1, col2 = st.columns([3,1])
    with col2:
        st.image(os.path.join(ASSETS, "scanner.gif"))

    with col1:
        st.markdown("<label style='color:#cfeeff;font-weight:700;'>Paste News Article</label>", unsafe_allow_html=True)
        text = st.text_area(
            "News Article",
            height=300,
            placeholder="Paste full article...",
            label_visibility="collapsed"
        )


        if st.button("ANALYZE ARTICLE"):
            if len(text.strip()) < 20:
                st.warning("Text is too short.")
            else:
                with st.spinner("Scanning article..."):
                    time.sleep(1)

                label, conf = predict_news(text)
                conf_pct = round(conf * 100, 2)

                if label == "TRUE":
                    st.success(f"✔ VERIFIED — TRUE ({conf_pct}% confidence)")
                else:
                    st.error(f"✖ FAKE NEWS ({conf_pct}% confidence)")

                snippet = text[:300] + "..." if len(text) > 300 else text
                timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

                df = pd.read_csv(HISTORY_CSV)
                df.loc[len(df)] = [timestamp, snippet, label, conf]
                df.to_csv(HISTORY_CSV, index=False)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# LIVE FEED (includes fetch function)
# --------------------------------------------------
@st.cache_data(ttl=300)
def fetch_newsapi(topic="technology", page_size=6):
    api_key = os.getenv("NEWSAPI_KEY", "")
    if not api_key:
        return {"error": "NEWSAPI_KEY missing."}

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": topic,
        "pageSize": page_size,
        "language": "en",
        "sortBy": "publishedAt",
        "apiKey": api_key
    }

    try:
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        return {"articles": r.json().get("articles", [])}
    except Exception as e:
        return {"error": str(e)}

def page_live_feed():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown("<h2 style='color:#67e8ff'>Live Feed</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns([4,1])
    topic = col1.text_input("Search Topic", "technology")

    # Refresh button
    if col2.button("Refresh"):
        st.cache_data.clear()
        st.session_state["refresh"] = True

    # Trigger safe rerun
    if st.session_state.get("refresh"):
        st.session_state["refresh"] = False
        st.rerun()

    # Fetch news
    res = fetch_newsapi(topic)
    if "error" in res:
        st.error(res["error"])
        return

    cols = st.columns(3)
    for i, art in enumerate(res["articles"]):
        col = cols[i % 3]

        col.image(os.path.join(ASSETS, "news_icon.png"), width=40)

        title = art.get("title", "No title")
        desc = art.get("description", "")
        url = art.get("url", "#")

        label, conf = predict_news(desc or title)
        conf_pct = round(conf * 100, 1)
        color = "#6bffdf" if label == "TRUE" else "#ff6b7a"

        col.markdown(f"""
        <div style="margin-bottom:12px">
            <div style="font-weight:800;color:#a8d6ff">{title}</div>
            <div style="color:{color};font-weight:700">{label} — {conf_pct}%</div>
            <a href="{url}" target="_blank">Open Source</a>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# PROJECT INFO
# --------------------------------------------------
def page_project_info():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown("<h2 style='color:#67e8ff'>📘 Project Info</h2>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["✨ Features", "🧠 ML Pipeline", "⚙ Tech Stack", "🚀 Future Scope"])

    with tab1:
        st.image(os.path.join(ASSETS, "features_icon.png"), width=90)
        st.markdown("""
        ### ✨ FEATURES
        - Real-time Fake News Detection  
        - Cyberpunk Neon UI  
        - TF-IDF + Random Forest ML Pipeline  
        - Live News Feed Analysis  
        - Confidence Scores  
        - History Export  
        """)

    with tab2:
        st.image(os.path.join(ASSETS, "ml_icon.png"), width=90)
        st.markdown("""
        ### 🧠 ML PIPELINE
        1. Text Cleaning  
        2. TF-IDF Vectorization  
        3. Random Forest Classification  
        4. Real-Time Prediction  
        """)

    with tab3:
        st.image(os.path.join(ASSETS, "tech_stack.png"), width=90)
        st.markdown("""
        ### ⚙ TECH STACK
        - Python  
        - Pandas, NumPy  
        - Scikit-Learn  
        - Streamlit  
        - Joblib  
        - Custom CSS  
        """)

    with tab4:
        st.image(os.path.join(ASSETS, "future_icon.png"), width=90)
        st.markdown("""
        ### 🚀 FUTURE SCOPE
        - Multi-lingual Fake News Detection  
        - BERT Hybrid Model  
        - Video/Audio Fake Detection  
        - Mobile App Deployment  
        """)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# HISTORY PAGE
# --------------------------------------------------
def page_history():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown("<h2 style='color:#67e8ff'>🕒 History</h2>", unsafe_allow_html=True)

    try:
        df = pd.read_csv(HISTORY_CSV)
    except:
        df = pd.DataFrame(columns=["timestamp", "text_snippet", "label", "confidence"])

    col1, col2 = st.columns([4,1])
    if col2.button("Clear History"):
        df = df.iloc[0:0]
        df.to_csv(HISTORY_CSV, index=False)
        # use st.rerun() (safe)
        st.rerun()

    st.dataframe(df.sort_values(by="timestamp", ascending=False), width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# ROUTER
# --------------------------------------------------
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

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown('<div class="footer-strip"></div>', unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center;color:#8aa6d6'>Developed by Nandini Raju Bhalekar — Bluff</div>",
    unsafe_allow_html=True
)
