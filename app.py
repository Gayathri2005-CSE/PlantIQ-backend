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
# 🌿 DISEASE DATABASE (BIG CONTENT)
# =========================
DISEASE_INFO = {

    # 🌶️ CHILLI
    "chilli healthy": {
        "description": """
The chilli plant is in a completely healthy condition with vibrant green leaves and strong stem growth. There are no visible signs of disease, pest attack, or nutrient deficiency. Photosynthesis is active and plant growth is normal.

மிளகாய் செடி முழுமையாக ஆரோக்கியமாக உள்ளது. இலைகள் பசுமையாகவும் சுறுசுறுப்பாகவும் உள்ளன. எந்த நோயும் அல்லது பூச்சி தாக்கமும் இல்லை.
""",
        "treatment": """
No treatment is required. Continue regular monitoring and maintain good agricultural practices to keep the plant healthy.

சிகிச்சை தேவையில்லை. வழக்கமான பராமரிப்பை தொடரவும்.
""",
        "fertilizer": """
Apply balanced NPK fertilizer (10:10:10 or 20:20:20) every 15 days. Add organic compost or farmyard manure to enrich soil fertility.

ஒவ்வொரு 15 நாட்களுக்கும் NPK உரம் பயன்படுத்தவும். இயற்கை உரம் சேர்க்கவும்.
""",
        "routine": """
Water 2–3 times per week based on weather conditions. Ensure good sunlight exposure and avoid water stagnation near roots.

வாரத்திற்கு 2–3 முறை நீர் ஊற்றவும். நீர் தேங்க விட வேண்டாம்.
"""
    },

    "chilli leaf curl": {
        "description": """
Leaf Curl disease is caused by viral infection transmitted by whiteflies. Leaves curl upward, become thick, and plant growth becomes stunted. Yield is severely affected if not treated early.

இலை சுருட்டல் நோய் வைரஸ் காரணமாக ஏற்படும். இலைகள் சுருண்டு வளர்ச்சி குறையும்.
""",
        "treatment": """
Remove and destroy infected leaves immediately. Spray neem oil or imidacloprid to control whiteflies. Use yellow sticky traps.

பாதிக்கப்பட்ட இலைகளை அகற்றவும். வேப்பெண்ணெய் அல்லது பூச்சி மருந்து தெளிக்கவும்.
""",
        "fertilizer": """
Use potassium-rich fertilizers and micronutrients like Zinc (Zn) and Magnesium (Mg) to improve resistance.

பொட்டாசியம் மற்றும் மைக்ரோ ஊட்டச்சத்து உரம் பயன்படுத்தவும்.
""",
        "routine": """
Maintain field hygiene, avoid overcrowding, and regularly monitor for pests. Install sticky traps.

வயலை சுத்தமாக வைத்திருக்கவும் மற்றும் பூச்சி கண்காணிக்கவும்.
"""
    },

    "chilli leafspot": {
        "description": """
Leaf spot is a fungal disease that causes small brown or black circular spots on leaves. In severe cases, leaves dry and fall, reducing yield.

இலை புள்ளி நோய் பூஞ்சை காரணமாக ஏற்படும்.
""",
        "treatment": """
Spray copper-based fungicides or Mancozeb regularly. Remove infected leaves and burn them.

பூஞ்சை மருந்து தெளிக்கவும்.
""",
        "fertilizer": """
Apply nitrogen and potassium fertilizers to support new leaf growth.

நைட்ரஜன் + பொட்டாசியம் உரம்.
""",
        "routine": """
Avoid water on leaves, ensure good air circulation, and avoid overcrowding.

இலை ஈரம் தவிர்க்கவும்.
"""
    },

    # 🌰 GROUNDNUT
    "groundnut healthy": {
        "description": """
Groundnut plant is healthy with proper root development, green foliage, and good pod formation.

நிலக்கடலை செடி ஆரோக்கியமாக உள்ளது.
""",
        "treatment": """
No treatment required. Maintain soil moisture and monitor plant regularly.

சிகிச்சை தேவையில்லை.
""",
        "fertilizer": """
Apply phosphorus-rich fertilizer for better root and pod development.

பாஸ்பரஸ் உரம்.
""",
        "routine": """
Water moderately and avoid over-irrigation.

மிதமான நீர்.
"""
    },

    "groundnut leafspot": {
        "description": """
Fungal disease causing dark spots on leaves, leading to premature leaf fall and reduced yield.

பூஞ்சை நோய்.
""",
        "treatment": """
Spray Mancozeb or Chlorothalonil fungicide.

பூஞ்சை மருந்து தெளிக்கவும்.
""",
        "fertilizer": """
Use balanced NPK fertilizers to strengthen plant immunity.

சமமான NPK உரம்.
""",
        "routine": """
Maintain spacing and avoid humidity buildup.

இடைவெளி வைக்கவும்.
"""
    },

    # 🍅 TOMATO
    "tomato healthy": {
        "description": """
Tomato plant is in a healthy state with strong stems, green leaves, and proper flowering.

தக்காளி செடி ஆரோக்கியமாக உள்ளது.
""",
        "treatment": """
No treatment required. Continue regular care and monitoring.

சிகிச்சை தேவையில்லை.
""",
        "fertilizer": """
Apply compost and balanced NPK fertilizer regularly.

இயற்கை உரம் + NPK.
""",
        "routine": """
Water regularly but avoid water stagnation.

நீர் தேங்க விட வேண்டாம்.
"""
    },

    "tomato early blight": {
        "description": """
Early blight is a fungal disease causing brown concentric spots on leaves and stems, reducing plant productivity.

ஆரம்ப பிளைட் நோய்.
""",
        "treatment": """
Spray Copper oxychloride or Mancozeb at early stage.

பூஞ்சை மருந்து தெளிக்கவும்.
""",
        "fertilizer": """
Use calcium and potassium-rich fertilizers.

கால்சியம் + பொட்டாசியம்.
""",
        "routine": """
Maintain proper spacing and avoid wet leaves.

இடைவெளி வைக்கவும்.
"""
    },

    "tomato late blight": {
        "description": """
Late blight is a severe disease that spreads rapidly and destroys leaves, stems, and fruits.

கடுமையான பூஞ்சை நோய்.
""",
        "treatment": """
Apply Metalaxyl or Copper fungicides immediately.

உடனடி மருந்து தெளிக்கவும்.
""",
        "fertilizer": """
Use potassium-rich fertilizer to improve plant resistance.

பொட்டாசியம் உரம்.
""",
        "routine": """
Avoid high humidity and ensure good drainage.

ஈரப்பதம் தவிர்க்கவும்.
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

        # resize (memory optimization)
        img = Image.open(filepath)
        img = img.resize((320, 320))
        img.save(filepath)

        results = model.predict(filepath, imgsz=320, conf=0.25)
        r = results[0]

        if r.probs is None:
            return jsonify({"error": "Model output empty"}), 500

        prediction = r.names[int(r.probs.top1)]

        # 🔥 improved key matching
        key = prediction.lower().replace("_", " ").replace("-", " ").strip()

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
    app.run(host="0.0.0.0", port=port, debug=False)
