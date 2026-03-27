from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
CORS(app)

# =======================
# LOAD DATASET
# =======================
try:
    data = pd.read_csv("ai_vs_human_3000 (2).csv")
    print("✅ Dataset loaded successfully!")
except Exception as e:
    print("❌ Dataset error:", e)

# =======================
# PREPROCESS
# =======================
data['text'] = data['text'].astype(str).str.lower()

# =======================
# TRAIN MODEL
# =======================
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['label']

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

print("✅ Model trained successfully!")

# =======================
# HOME PAGE
# =======================
@app.route('/')
def home():
    return render_template("index.html")

# =======================
# PREDICT API
# =======================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '')

        print("📩 Received:", text)

        if text.strip() == "":
            return jsonify({"error": "Empty input"})

        vec = vectorizer.transform([text.lower()])
        result = model.predict(vec)[0]

        # Confidence score
        prob = model.predict_proba(vec)[0]
        confidence = float(max(prob))

        # Label
        label = "Human" if result == 0 else "AI"

        # Decision logic (IMPORTANT)
        if label == "AI" and confidence > 0.8:
            decision = "Likely AI-generated"
        elif confidence > 0.6:
            decision = "Needs Review"
        else:
            decision = "Acceptable"

        print("🔮 Prediction:", label, confidence, decision)

        return jsonify({
            "prediction": label,
            "confidence": round(confidence, 2),
            "decision": decision
        })

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": str(e)})

# =======================
# RUN
# =======================
if __name__ == '__main__':
    app.run(debug=True)