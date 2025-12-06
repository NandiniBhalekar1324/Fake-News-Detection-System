import os
import re
import time
import joblib
import requests
import pandas as pd
import streamlit as st
from datetime import datetime

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
    # Ensure these files exist in your directory
    model = joblib.load("random_forest_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

try:
    model, vectorizer = load_resources()
except Exception as e:
    st.error(f"Error loading model files: {e}. Please ensure .pkl files are in the same directory.")
    st.stop()

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
    /* Global App Styling */
    .stApp {
        background: radial-gradient(circle at 10% 10%, rgba(86,26,139,0.08), transparent 8%),
                    linear-gradient(180deg, #05050a 0%, #020205 100%);
        color: #dbe9ff;
        font-family: Inter, sans-serif;
    }

    /* --- SIDEBAR CUSTOMIZATION (Dark Mode) --- */
    [data-testid="stSidebar"] {
        background-color: #020205;
        border-right: 1px solid rgba(77,166,255,0.1);
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }
    /* Sidebar Text Colors */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #67e8ff !important;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label, [data-testid="stSidebar"] div {
        color: #9fbde6;
    }

    /* Main card styling */
    .main-card {
        background: linear-gradient(135deg, rgba(18,10,30,0.6), rgba(8,8,18,0.62));
        border: 1px solid rgba(77,166,255,0.06);
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.5);
        backdrop-filter: blur(6px) saturate(120%);
    }

    /* Neon Titles */
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
        text-shadow: 0 0 5px #00eaff, 0 0 12px #00eaff, 0 0 22px #00eaff;
    }

    /* Radio Buttons */
    .stRadio .stRadio > div[role="radiogroup"] > label[aria-checked="true"],
    .stRadio label[aria-checked="true"] {
        background: linear-gradient(90deg, rgba(0,234,255,0.1), rgba(107,45,255,0.1));
        border-left: 3px solid #00eaff;
        color: #dffaff !important;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg,#1e1e2e,#252535);
        color: #67e8ff;
        border: 1px solid #67e8ff;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg,#00eaff,#0077ff);
        color: #000;
        border: 1px solid #00eaff;
    }

    /* Stats & Cards */
    .stat-card {
        background: linear-gradient(180deg, rgba(14,12,28,0.6), rgba(8,8,18,0.5));
        border-radius: 10px;
        padding: 12px;
        border: 1px solid rgba(77,166,255,0.06);
        text-align: center;
    }
    .stat-number { font-weight:800; color:#a7f3ff; font-size:20px; }
    .stat-label { color:#9fbde6; font-size:12px; }

    .placeholder-card {
        background: linear-gradient(90deg, rgba(30,10,40,0.6), rgba(8,8,18,0.5));
        border-radius: 10px;
        padding: 18px;
        border: 1px solid rgba(255,255,255,0.02);
        box-shadow: 0 8px 30px rgba(0,0,0,0.5);
        color:#dfefff;
    }

    .footer-strip {
        height: 4px;
        background: linear-gradient(90deg, rgba(0,234,255,0.25), rgba(107,45,255,0.20));
        border-radius: 4px;
        margin-top: 14px;
        margin-bottom: 6px;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: rgba(0,0,0,0.2);
        border-radius: 8px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 6px;
        color: #9fbde6;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(0, 234, 255, 0.1) !important;
        color: #00eaff !important;
        border: 1px solid rgba(0, 234, 255, 0.2);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Sidebar
# -------------------------
st.sidebar.markdown("""
<div class="logo-wrapper">
    <span class="neon-title blue">TRUTH<br>SCANNER 2.0</span>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("<h3 style='color:#67e8ff;margin:6px 0 2px 0;font-weight:700;'>Bluff</h3>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='color:#9fbde6;margin-top:-6px; font-size: 12px;'>Cyberpunk Fake News Detection</div>", unsafe_allow_html=True)
st.sidebar.markdown("---", unsafe_allow_html=True)

menu = st.sidebar.radio("Navigation", ["🏡 Home", "🔎 Detector", "📡 Live Feed", "📘 Project Info", "🕒 History"])

st.sidebar.markdown("---", unsafe_allow_html=True)
st.sidebar.markdown("<div style='color:#94a3b8;font-size:11px'>Developed by<br><b>Nandini Raju Bhalekar</b></div>", unsafe_allow_html=True)

# -------------------------
# Pages
# -------------------------
def page_home():
    st.markdown('<div class="header main-card">', unsafe_allow_html=True)
    st.markdown(f'<div style="display:flex; align-items:center; justify-content:space-between;"><div><h2 style="margin:0;color:#67e8ff">BLUFF</h2><div style="color:#9fbde6">Neon-assisted Fake News Detection</div></div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="main-card" style="margin-top:14px">', unsafe_allow_html=True)
    st.markdown('<div style="display:flex; gap:14px; align-items:stretch">', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns([1,1,1,1])
    with col1:
        st.markdown('<div class="stat-card"><div class="stat-number">Random Forest</div><div class="stat-label">Model</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stat-card"><div class="stat-number">96.5%</div><div class="stat-label">Accuracy (est)</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{history_count()}</div><div class="stat-label">Analyses</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="stat-card"><div class="stat-number">~0.6s</div><div class="stat-label">Latency</div></div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-card"><b>Quick Start</b><br>Paste an article in Detector → click <b>ANALYZE ARTICLE</b>. Use History to export results.</div>', unsafe_allow_html=True)

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
        st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
        st.markdown('<div style="color:#94a3b8; font-size:12px">Model: RandomForest<br>Vectorizer: TF-IDF<br>Lang: English</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    if analyze_clicked:
        if not user_text or len(user_text.strip()) < 20:
            st.warning("Text is too short. Please add more content.")
        else:
            with st.spinner("Scanning article..."):
                time.sleep(1.0) # Artificial delay for effect
            label, conf = predict_news(user_text)
            conf_pct = round(conf * 100, 2)
            
            # Result Display
            st.markdown("---")
            if label == "TRUE":
                st.markdown(f'''
                    <div style="background: rgba(0, 255, 128, 0.1); border: 1px solid #00ff80; padding: 20px; border-radius: 10px; text-align: center;">
                        <h2 style="color: #00ff80; margin:0;">✔ VERIFIED — REAL NEWS</h2>
                        <p style="color: #dbe9ff; margin:0;">Confidence: {conf_pct}%</p>
                    </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                    <div style="background: rgba(255, 0, 80, 0.1); border: 1px solid #ff0050; padding: 20px; border-radius: 10px; text-align: center;">
                        <h2 style="color: #ff0050; margin:0;">✖ ALERT — FAKE NEWS DETECTED</h2>
                        <p style="color: #dbe9ff; margin:0;">Confidence: {conf_pct}%</p>
                    </div>
                ''', unsafe_allow_html=True)

            snippet = (user_text[:300] + "...") if len(user_text) > 300 else user_text
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            try:
                df = pd.read_csv(HISTORY_CSV)
            except Exception:
                df = pd.DataFrame(columns=["timestamp", "text_snippet", "label", "confidence"])
            df.loc[len(df)] = [timestamp, snippet, label, conf]
            df.to_csv(HISTORY_CSV, index=False)

# ------------ NewsAPI fetcher (cached with ttl 5 minutes) ------------
@st.cache_data(ttl=300)
def fetch_newsapi(topic: str = "technology", page_size: int = 6):
    # Try Streamlit secrets first, then environment var
    api_key = st.secrets.get("NEWSAPI_KEY")
    if not api_key:
        api_key = os.getenv("NEWSAPI_KEY", "")

    if not api_key:
        return {"error": "NEWSAPI_KEY not found. Please set it in secrets or env vars."}

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
        if data.get("status") != "ok":
            return {"error": data.get("message", "Unknown NewsAPI error")}
        return {"articles": data.get("articles", [])}
    except Exception as e:
        return {"error": str(e)}

def page_live_feed():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown('<h2 style="color:#67e8ff">Live Feed</h2>', unsafe_allow_html=True)

    col_t, col_refresh = st.columns([4,1])
    with col_t:
        topic = st.text_input("Topic to search", "technology")
    with col_refresh:
        st.write("") # Spacer
        st.write("") # Spacer
        if st.button("Refresh Feed"):
            st.cache_data.clear()
            st.rerun()

    # fetch articles
    res = fetch_newsapi(topic=topic, page_size=6)

    if "error" in res:
        st.warning(f"⚠️ {res['error']}")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    articles = res["articles"]
    if not articles:
        st.info("No articles found for this topic.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    cols = st.columns(3)
    for i, art in enumerate(articles):
        col = cols[i % 3]
        title = art.get("title", "No title")
        desc = art.get("description") or art.get("content") or ""
        source = art.get("source", {}).get("name", "Unknown")
        published = art.get("publishedAt", "")[:16]
        url = art.get("url", "#")

        # Predict on live data
        label, conf = predict_news(desc or title)
        conf_pct = round(conf * 100, 1)
        color = "#6bffdf" if label == "TRUE" else "#ff6b7a"

        with col:
            st.markdown(
                f"""
                <div class="placeholder-card" style="margin-bottom: 20px; height: 100%;">
                    <div style="font-weight:800;color:#a8d6ff; height: 60px; overflow:hidden;">{title[:60]}...</div>
                    <div style="color:#9fbde6;font-size:12px; margin-top:4px">{source}</div>
                    <div style="height:12px"></div>
                    <div style="font-weight:700;color:{color}; border:1px solid {color}; border-radius:4px; text-align:center; padding:4px;">
                        {label} ({conf_pct}%)
                    </div>
                    <div style="height:8px"></div>
                    <a href="{url}" target="_blank" style="color:#67e8ff; text-decoration:none; font-size:13px;">🔗 Read Article</a>
                </div>
                """,
                unsafe_allow_html=True
            )
    st.markdown("</div>", unsafe_allow_html=True)

def page_project_info():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown("## 📘 Project Documentation", unsafe_allow_html=True)
    st.markdown("---")

    # TABS IMPLEMENTATION
    tab1, tab2, tab3, tab4 = st.tabs(["🔧 Overview", "🧠 Model Architecture", "💻 Tech Stack", "🔮 Future Scope"])

    with tab1:
        st.markdown("### About Bluff")
        st.info("Bluff is a real-time fake news detection system designed with a Cyberpunk aesthetic.")
        st.markdown("""
        **Core Features:**
        * **Real-time Detection:** Paste any text to get an instant credibility score.
        * **Live News Feed:** Fetches latest news via NewsAPI and runs detection on them automatically.
        * **History Tracking:** Saves all your analyses to a local CSV file for reporting.
        """)
    
    with tab2:
        st.markdown("### The Brain: Random Forest")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("""
            **Pipeline Steps:**
            1.  **Input:** Raw Text
            2.  **Cleaning:** Lowercase, Remove URLs, Remove Punctuation.
            3.  **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency).
            4.  **Classification:** Random Forest Classifier.
            """)
        with col_m2:
             st.success("Accuracy: ~96.5% (Validation Set)")
             st.markdown("Why Random Forest? It handles high-dimensional data (like text) exceptionally well and resists overfitting better than simple Decision Trees.")

    with tab3:
        st.markdown("### Stack & Libraries")
        st.code("""
        Language:   Python 3.9+
        Frontend:   Streamlit
        ML Libs:    Scikit-learn, Joblib, Pandas, Numpy
        API:        NewsAPI (for live feed)
        Styling:    Custom CSS / HTML Injection
        """, language="text")

    with tab4:
        st.markdown("### Planned Upgrades")
        st.warning("⚠️ Work in Progress")
        st.markdown("""
        1.  **Deep Learning Integration:** Switch to LSTM or BERT for context awareness.
        2.  **Chrome Extension:** Verify news directly in the browser.
        3.  **Multi-language Support:** Add Hindi and Marathi detection.
        """)

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
        st.markdown("#### Actions")
        if not df.empty:
            csv_bytes = df.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Download CSV", data=csv_bytes, file_name="history.csv", mime="text/csv")
            if st.button("🗑️ Clear History"):
                pd.DataFrame(columns=["timestamp", "text_snippet", "label", "confidence"]).to_csv(HISTORY_CSV, index=False)
                st.rerun()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    
    if df.empty:
        st.info("No history yet. Run analyses in the Detector tab to populate this table.")
    else:
        # Style the dataframe for better dark mode visibility
        st.dataframe(
            df.sort_values(by="timestamp", ascending=False).head(300),
            use_container_width=True,
            hide_index=True
        )
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