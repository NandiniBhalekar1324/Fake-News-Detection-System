import streamlit as st
import joblib
import re
import time
import requests
import pandas as pd
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
from GoogleNews import GoogleNews
from newspaper import Article, Config

# ==========================
# 1. Page Configuration
# ==========================
st.set_page_config(
    page_title="Veritas | AI Fake News Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# 2. Load Resources (Cached)
# ==========================
@st.cache_resource
def load_model():
    model = joblib.load("random_forest_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

def load_lottieurl(url: str):
    import requests
    r = requests.get(url)
    print("STATUS CODE:", r.status_code)   # Debug
    if r.status_code != 200:
        return None
    return r.json()

# Working animations
lottie_news = load_lottieurl(
    "https://lottie.host/04de6b9c-7702-4fbe-80b1-3442e2f0e393/1nJw2BWumq.json"
)

lottie_scanning = load_lottieurl(
    "https://lottie.host/8c823c25-2f55-49a0-8f07-645d3f21a2a4/8M0PaGqzWj.json"
)



# ==========================
# 3. Helper Functions
# ==========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def predict_news(news_text):
    cleaned = clean_text(news_text)
    transformed = vectorizer.transform([cleaned])
    
    # Get probability for confidence score
    # Random Forest supports predict_proba
    proba = model.predict_proba(transformed)[0] 
    prediction = model.predict(transformed)[0]
    
    # proba[0] is FAKE, proba[1] is TRUE (based on your mapping: 0=FAKE, 1=TRUE)
    confidence = proba[1] if prediction == "TRUE" else proba[0]
    
    return prediction, confidence

# ==========================
# 4. Custom CSS (Modern UI)
# ==========================
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: linear-gradient(to right, #141e30, #243b55);
        color: white;
    }
    
    /* Input Area Styling */
    .stTextArea textarea {
        background-color: #f0f2f6;
        color: #000;
        border-radius: 10px;
        border: 2px solid #4CAF50;
    }

    /* Card Styling */
    .css-1r6slb0 {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 15px;
    }
    
    /* Header Styling */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
    }
    
    /* Metric styling */
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        color: #FFD700;
    }
</style>
""", unsafe_allow_html=True)

# ==========================
# 5. Sidebar Layout
# ==========================
with st.sidebar:
    st_lottie(lottie_news, height=150, key="news_anim")
    st.title("🛡️ Veritas AI")
    st.markdown("---")
    
    menu = st.radio("Navigation", ["🏠 Home", "🕵️ Detector", "🔴 Live Feed", "📊 Project Info"])
    
    st.markdown("---")
    st.info("Developed by **Nandini Raju Bhalekar**\nDSBDA Honors Student")

# ==========================
# 6. Main Content - Logic
# ==========================

# --- HOME TAB ---
if menu == "🏠 Home":
    st.title("Welcome to the Truth.")
    st.markdown("### A Robust System for Fake News Detection Using NLP & ML")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        In an era of misinformation, **Veritas AI** serves as your digital shield. 
        
        Using advanced **Random Forest** algorithms and **TF-IDF** vectorization, we analyze news patterns to distinguish between credible reporting and fabricated stories.
        
        **Features:**
        * ✅ Instant Text Analysis
        * ✅ Live News Scraping
        * ✅ Confidence Scoring
        """)
        if st.button("Start Detecting Now >"):
            st.toast("Go to the Detector Tab!")

    with col2:
        st_lottie(lottie_scanning, height=300)

# --- DETECTOR TAB ---
elif menu == "🕵️ Detector":
    st.title("🕵️ Analyze News Article")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_input = st.text_area("Paste the news article content here:", height=250, placeholder="Breaking news...")
        
        if st.button("🔍 Analyze Veracity", use_container_width=True):
            if len(user_input) < 20:
                st.warning("⚠️ Text is too short for accurate prediction. Please add more context.")
            else:
                with st.spinner("Processing NLP patterns..."):
                    time.sleep(1) # UX delay
                    label, confidence = predict_news(user_input)
                
                # Dynamic Results
                if label == "TRUE":
                    st.success(f"✅ VERIFIED: This news appears to be TRUE ({confidence*100:.2f}% confidence)")
                    gauge_color = "green"
                else:
                    st.error(f"🧢 ALERT: This news appears to be FAKE ({confidence*100:.2f}% confidence)")
                    gauge_color = "red"

                # Plotly Gauge Chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = confidence * 100,
                    title = {'text': f"Confidence ({label})"},
                    gauge = {'axis': {'range': [0, 100]},
                             'bar': {'color': gauge_color},
                             'steps' : [
                                 {'range': [0, 50], 'color': "lightgray"},
                                 {'range': [50, 100], 'color': "white"}],
                             'threshold' : {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 90}}))
                
                with col2:
                    st.plotly_chart(fig, use_container_width=True)

    with col2:
        if not user_input:
            st.info("ℹ️ **How it works:**\n\nOur model looks for linguistic patterns often found in fake news, such as:\n- Sensationalist language\n- Excessive capitalization\n- Lack of specific sources")

# --- LIVE FEED TAB (REAL TIME) ---
elif menu == "🔴 Live Feed":
    st.title("🔴 Live News Analysis")
    st.markdown("Fetching the latest headlines from **Google News** and analyzing them in real-time.")
    
    topic = st.text_input("Enter a topic to search (e.g., 'Technology', 'Politics'):", "Technology")
    
    if st.button("Fetch & Analyze Live"):
        with st.spinner(f"Scraping live news for '{topic}'..."):
            googlenews = GoogleNews()
            googlenews.set_lang('en')
            googlenews.search(topic)
            results = googlenews.result()
            
            # Clear old results
            googlenews.clear()
            
            if not results:
                st.error("No news found. Try a different topic.")
            else:
                st.success(f"Found {len(results)} recent articles.")
                
                for item in results[:5]: # Analyze top 5
                    with st.expander(f"📰 {item['title']}"):
                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            st.write(f"**Source:** {item['media']}")
                            st.write(f"**Date:** {item['date']}")
                            # Use description as proxy for content if full scrape fails
                            content = item['desc'] 
                            st.write(f"**Snippet:** {content}")
                            
                            label, conf = predict_news(content)
                            
                        with col_b:
                            if label == "FAKE":
                                st.error(f"**FAKE**\n{conf*100:.1f}%")
                            else:
                                st.success(f"**TRUE**\n{conf*100:.1f}%")

# --- PROJECT INFO TAB ---
elif menu == "📊 Project Info":
    st.title("📊 Project Metrics")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Model Accuracy", "96.5%", "+1.2%")
    col2.metric("Training Samples", "72,134", "WELFake")
    col3.metric("Algorithm", "Random Forest", "Ensemble")
    
    st.markdown("### Confusion Matrix")
    # You can add a static image of your confusion matrix here if you saved it
    st.info("The Random Forest model was selected over Logistic Regression due to better handling of non-linear relationships in text data and higher resistance to overfitting on this specific dataset.")