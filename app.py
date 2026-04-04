from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import os
import uuid
import torch
from PIL import Image

# =========================
# ⚙️ PERFORMANCE FIX
# =========================
torch.set_num_threads(1)

app = Flask(__name__)
CORS(app)

# =========================
# MODEL PATH
# =========================
MODEL_PATH = "models/best.pt"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model = YOLO(MODEL_PATH)

# =========================
# UPLOAD FOLDER
# =========================
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================
# 🌿 DISEASE DATABASE (IMPROVED)
# =========================
DISEASE_INFO = {

    "tomato healthy": {
        "description": """
The tomato plant is in excellent health with strong stems, vibrant green leaves, and proper flowering and fruit development. No visible symptoms of disease or pest infestation are present.

தக்காளி செடி ஆரோக்கியமாக உள்ளது. இலைகள் பசுமையாகவும் செழுமையாகவும் உள்ளன.
""",
        "treatment": """
No treatment required. Continue regular monitoring and maintain proper care practices.

சிகிச்சை தேவையில்லை. தொடர்ந்து பராமரிக்கவும்.
""",
        "fertilizer": """
Apply organic compost along with balanced NPK fertilizers (10:10:10 or 20:20:20) every 2 weeks for sustained growth.

இயற்கை உரம் + NPK உரம் பயன்படுத்தவும்.
""",
        "routine": """
Water regularly but avoid waterlogging. Ensure good sunlight exposure (6–8 hours daily) and proper air circulation.

நீர் தேங்க விட வேண்டாம். போதுமான சூரிய ஒளி பெற வேண்டும்.
"""
    },

    "tomato early blight": {
        "description": """
Early blight is a fungal disease causing brown spots with concentric rings on leaves, stems, and fruits. It can reduce yield significantly if untreated.

இது ஒரு பூஞ்சை நோய் ஆகும். இலைகளில் வட்ட புள்ளிகள் உருவாகும்.
""",
        "treatment": """
Spray fungicides like Mancozeb or Copper Oxychloride every 7–10 days. Remove infected leaves.

பூஞ்சை மருந்து தெளிக்கவும் மற்றும் பாதிக்கப்பட்ட இலைகளை அகற்றவும்.
""",
        "fertilizer": """
Use potassium-rich fertilizers to improve resistance. Avoid excess nitrogen.

பொட்டாசியம் அதிகமுள்ள உரம் பயன்படுத்தவும்.
""",
        "routine": """
Avoid overhead watering. Maintain spacing and remove debris around plants.

இலை ஈரப்பதம் தவிர்க்கவும்.
"""
    },

    "tomato late blight": {
        "description": """
Late blight spreads rapidly and causes dark lesions on leaves, stems, and fruits. It can destroy crops quickly under humid conditions.

கடுமையான பூஞ்சை நோய். விரைவாக பரவும்.
""",
        "treatment": """
Use Metalaxyl or Copper-based fungicides immediately. Remove infected plants.

உடனடியாக மருந்து தெளிக்கவும்.
""",
        "fertilizer": """
Apply potassium and calcium fertilizers to strengthen plant tissues.

பொட்டாசியம் + கால்சியம்.
""",
        "routine": """
Avoid high humidity and ensure proper drainage.

ஈரப்பதம் தவிர்க்கவும்.
"""
    },

    "chilli leaf curl": {
        "description": """
Leaf curl disease is caused by viruses transmitted by whiteflies. Leaves curl upward and plant growth becomes stunted.

இலை சுருட்டல் நோய் வைரஸ் காரணமாக ஏற்படும்.
""",
        "treatment": """
Spray neem oil or Imidacloprid to control whiteflies. Remove infected plants.

வேப்பெண்ணெய் அல்லது பூச்சி மருந்து பயன்படுத்தவும்.
""",
        "fertilizer": """
Use micronutrients like Zinc and Magnesium.

மைக்ரோ ஊட்டச்சத்து உரம் பயன்படுத்தவும்.
""",
        "routine": """
Install yellow sticky traps and maintain field hygiene.

பூச்சி கண்காணிப்பு அவசியம்.
"""
    }
}

# =========================
# HOME
# =========================
@app.route("/")
def home():
    return "🌱 PlantIQ API Running Successfully..."

# =========================
# PREDICT
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]

        filename = f"{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Resize (performance)
        img = Image.open(filepath)
        img = img.resize((320, 320))
        img.save(filepath)

        results = model.predict(filepath, imgsz=160, conf=0.25, device='cpu')
        r = results[0]

        if r.probs is None:
            return jsonify({"error": "Model output empty"}), 500

        # 🔥 RAW prediction
        raw_prediction = r.names[int(r.probs.top1)]

        # ✅ Clean prediction for frontend
        prediction = raw_prediction.replace("_", " ").replace("-", " ").title()

        # ✅ Key for database
        key = raw_prediction.lower().replace("_", " ").replace("-", " ").strip()

        # Delete file
        os.remove(filepath)

        if key not in DISEASE_INFO:
            return jsonify({
                "prediction": prediction,
                "error": "Disease info not found"
            })

        return jsonify({
            "prediction": prediction,
            "description": DISEASE_INFO[key]["description"],
            "treatment": DISEASE_INFO[key]["treatment"],
            "fertilizer": DISEASE_INFO[key]["fertilizer"],
            "routine": DISEASE_INFO[key]["routine"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
