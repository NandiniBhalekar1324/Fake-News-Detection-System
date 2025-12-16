<h1 align="center">📰 BLUFF — Fake News Detection System</h1>

<p align="center">
A machine‑learning powered web application that detects whether news content is real or fake.<br>
Built using <b>TF‑IDF</b>, <b>Random Forest Classifier</b>, and deployed using <b>Streamlit</b>.
</p>

<h3 align="center">
🔗 <a href="https://fake-news-detection-system-bluff.streamlit.app/" target="_blank">Live Demo</a>
</h3>

<hr>

<h2>🚀 Overview</h2>
<p>
BLUFF is a Fake News Detection System developed as part of our Mini Project (Semester 7).  
It allows users to:
</p>

<ul>
  <li>Paste any news article text</li>
  <li>Instantly check whether the article is <b>True</b> or <b>Fake</b></li>
  <li>View prediction confidence scores</li>
  <li>Analyze live news from the internet using the Live Feed feature</li>
  <li>Use the system on both desktop and mobile browsers</li>
  <li>Track prediction history with a built‑in <b>History</b> feature</li>
</ul>

<hr>

<h2>🧠 Tech Stack</h2>

<ul>
  <li><b>Python</b></li>
  <li><b>Scikit‑Learn</b></li>
  <li><b>TF‑IDF Vectorization</b></li>
  <li><b>Random Forest Classifier</b></li>
  <li><b>Pandas, NumPy</b></li>
  <li><b>Streamlit</b> (web deployment)</li>
  <li>Custom CSS (UI Enhancements)</li>
</ul>

<hr>

<h2>📉 Model Performance & Limitations</h2>
<p>
Our ML model is trained using a labeled dataset and performs very well on known or similar news patterns.
However, real‑time internet news is highly dynamic and complex, so predictions may not always generalize perfectly.
</p>

<b>We are actively improving:</b>
<ul>
  <li>Real‑time prediction accuracy</li>
  <li>Dataset expansion and cleaning</li>
  <li>Experiments with BERT & transformer‑based models</li>
</ul>

<hr>

<h2>📱 Mobile Support</h2>
<p>The application is fully responsive and works smoothly on smartphones via any browser.</p>

<hr>

<h2>👥 Team Members</h2>
<ul>
  <li><b>Nandini Raju Bhalekar</b></li>
  <li><b>Atharva Gurav</b></li>
  <li><b>Dhammadip Wagh</b></li>
  <li><b>Rushikesh Thokare</b></li>
</ul>

<hr>

<h2>📂 Project Structure</h2>

<pre>
├── app.py
├── random_forest_model.pkl
├── tfidf_vectorizer.pkl
├── assets/
│   ├── banner.gif
│   ├── icons...
├── history.csv
└── README.md
</pre>

<hr>

<h2>📝 How to Run Locally</h2>

<pre>
pip install -r requirements.txt
streamlit run app.py
</pre>

<hr>

<h3 align="center">⭐ If you like this project, consider giving it a star on GitHub!</h3>
