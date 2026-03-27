# 🤖 AI vs Human Text Detection System
## 📌 Project Overview
This project is a Machine Learning-based system that classifies whether a given text is:
* Human-written
* AI-generated
The system also provides a confidence score and a decision based on prediction probability.
## 📊 Dataset
* **Source**: Public dataset (GitHub / online sources)
* **Number of Samples**: XXXX (replace with your dataset size)
* **Columns**:
  * `text` → Input text
  * `label` → 0 (Human), 1 (AI)
## ⚙️ Preprocessing Steps
* Converted text to lowercase
* Removed punctuation and special characters
* Cleaned unnecessary spaces
## 🔢 Feature Extraction
* Used **TF-IDF Vectorizer**
* Included **n-grams** for better accuracy
## 🤖 Models Used
* Logistic Regression
* Naive Bayes
* Random Forest
## 📈 Model Evaluation
* Accuracy used as evaluation metric
* Compared performance of multiple models
## Decision Layer
Based on probability score:
* Confidence ≥ 0.85 → ✅ Highly Reliable
* 0.65 ≤ Confidence < 0.85 → ⚠️ Needs Review
* Confidence < 0.65 → ❌ Likely AI-generated
  Web Application
* Built using Flask (backend)
* HTML/CSS for frontend
* Allows user to input text and get real-time prediction
▶️ How to Run
1. Install dependencies:
pip install pandas scikit-learn flask
2. Run the application:
python app.py
3. Open browser:
http://127.0.0.1:5000
 Output
For each input text, the system displays:
* Prediction (Human / AI-generated)
* Confidence score
* Final decision
## 🚀 Conclusion
This project demonstrates how machine learning can be used to detect AI-generated content using text patterns and statistical features.


