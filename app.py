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
# MODEL
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
# 🌿 BIG DISEASE DATABASE
# =========================
DISEASE_INFO = {

    # ================= CHILLI =================

    "chilli healthy": {
        "description": """
Chilli plant is in excellent healthy condition. Leaves are green, fresh, and photosynthesis activity is normal. No pest or fungal infection observed.

மிளகாய் செடி முழுமையாக ஆரோக்கியமாக உள்ளது. இலைகள் பசுமையாகவும் சுறுசுறுப்பாகவும் உள்ளன. எந்த நோயும் காணப்படவில்லை.
""",
        "treatment": """
No chemical treatment required. Maintain regular monitoring and keep field clean.

எந்த மருந்தும் தேவையில்லை. செடியை தொடர்ந்து கண்காணிக்கவும் மற்றும் வயலை சுத்தமாக வைத்திருக்கவும்.
""",
        "fertilizer": """
Apply balanced NPK (10:10:10 or 20:20:20) every 15 days. Add organic compost or cow dung manure to improve soil fertility.

ஒவ்வொரு 15 நாட்களுக்கும் சமமான NPK உரம் பயன்படுத்தவும். இயற்கை உரம் சேர்க்கவும்.
""",
        "routine": """
Water 2–3 times per week depending on climate. Ensure proper sunlight and avoid water stagnation.

வாரத்திற்கு 2–3 முறை நீர் ஊற்றவும். அதிக நீர் தேங்க விட வேண்டாம்.
"""
    },

    "chilli leaf curl": {
        "description": """
Leaf Curl is a viral disease transmitted by whiteflies. Leaves become curled, thickened, and plant growth is stunted.

இலை சுருட்டல் என்பது வெள்ளை ஈக்கள் மூலம் பரவும் வைரஸ் நோய். இலைகள் சுருண்டு வளர்ச்சி குறையும்.
""",
        "treatment": """
Remove infected leaves immediately. Use neem oil spray or insecticidal soap. Control whiteflies using sticky traps.

பாதிக்கப்பட்ட இலைகளை அகற்றவும். வேப்பெண்ணெய் தெளிக்கவும். வெள்ளை ஈக்களை கட்டுப்படுத்தவும்.
""",
        "fertilizer": """
Use potassium-rich fertilizer and micronutrients like Zinc (Zn) and Magnesium (Mg) to improve plant immunity.

பொட்டாசியம் மற்றும் மைக்ரோ ஊட்டச்சத்து உரம் பயன்படுத்தவும்.
""",
        "routine": """
Install yellow sticky traps. Avoid overwatering and maintain field hygiene.

மஞ்சள் ஒட்டும் தாள்கள் பயன்படுத்தவும். அதிக நீர் தவிர்க்கவும்.
"""
    },

    "chilli leafspot": {
        "description": """
Leaf spot is a fungal disease causing brown/black circular spots on leaves. It spreads quickly in humid weather.

இலை புள்ளி நோய் பூஞ்சை காரணமாக ஏற்படும். ஈரமான சூழலில் வேகமாக பரவும்.
""",
        "treatment": """
Spray copper-based fungicide or Mancozeb. Remove infected leaves and destroy them.

பூஞ்சை மருந்து தெளிக்கவும். பாதிக்கப்பட்ட இலைகளை அகற்றவும்.
""",
        "fertilizer": """
Apply nitrogen + potassium fertilizers to help recovery and new leaf growth.

நைட்ரஜன் + பொட்டாசியம் உரம் பயன்படுத்தவும்.
""",
        "routine": """
Avoid wet leaves and improve air circulation between plants.

செடிகளுக்கு இடைவெளி வைக்கவும்.
"""
    },

    # ================= GROUNDNUT =================

    "groundnut healthy": {
        "description": """
Groundnut plant is healthy with strong root system and proper pod development.

நிலக்கடலை செடி ஆரோக்கியமாக உள்ளது.
""",
        "treatment": """
No treatment required. Maintain soil moisture.

சிகிச்சை தேவையில்லை.
""",
        "fertilizer": """
Use phosphorus-rich fertilizer for better root and pod formation.

பாஸ்பரஸ் உரம் பயன்படுத்தவும்.
""",
        "routine": """
Water once or twice weekly depending on soil moisture.

மண்ணின் ஈரப்பதத்தைப் பொறுத்து நீர் ஊற்றவும்.
"""
    },

    "groundnut leafspot": {
        "description": """
Fungal leaf spot causes dark circular lesions on leaves leading to defoliation and yield loss.

பூஞ்சை காரணமாக இலை புள்ளிகள் ஏற்பட்டு விளைச்சல் குறையும்.
""",
        "treatment": """
Spray Mancozeb or Chlorothalonil fungicide regularly.

பூஞ்சை மருந்து தெளிக்கவும்.
""",
        "fertilizer": """
Use balanced NPK fertilizers to strengthen plant immunity.

சமமான NPK உரம்.
""",
        "routine": """
Avoid plant overcrowding and maintain field cleanliness.

வயலை சுத்தமாக வைத்திருக்கவும்.
"""
    },

    # ================= TOMATO =================

    "tomato healthy": {
        "description": """
Tomato plant is healthy with strong stems, green leaves, and proper flowering.

தக்காளி செடி ஆரோக்கியமாக உள்ளது.
""",
        "treatment": """
No treatment required. Continue monitoring.

சிகிச்சை தேவையில்லை.
""",
        "fertilizer": """
Apply organic compost + NPK fertilizer every 2 weeks.

இயற்கை உரம் + NPK.
""",
        "routine": """
Water regularly but avoid waterlogging.

அதிக நீர் தேங்க விட வேண்டாம்.
"""
    },

    "tomato early blight": {
        "description": """
Early blight is a fungal disease causing brown concentric spots on leaves and reduces yield.

தக்காளி ஆரம்ப பிளைட் நோய்.
""",
        "treatment": """
Spray Copper oxychloride or Mancozeb fungicide.

பூஞ்சை மருந்து தெளிக்கவும்.
""",
        "fertilizer": """
Use calcium + potassium rich fertilizers for resistance.

கால்சியம் + பொட்டாசியம் உரம்.
""",
        "routine": """
Maintain plant spacing and avoid wet leaves.

இடைவெளி வைக்கவும்.
"""
    },

    "tomato late blight": {
        "description": """
Late blight is a severe disease that destroys leaves, stems, and fruits rapidly.

கடுமையான பூஞ்சை நோய்.
""",
        "treatment": """
Immediate fungicide spray like Metalaxyl or Copper-based chemicals.

உடனடி மருந்து தெளிக்கவும்.
""",
        "fertilizer": """
Use potassium-rich fertilizer to improve disease resistance.

பொட்டாசியம் உரம்.
""",
        "routine": """
Avoid humidity and ensure proper drainage.

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

        # resize (memory fix)
        img = Image.open(filepath)
        img = img.resize((320, 320))
        img.save(filepath)

        # prediction
        # prediction
results = model.predict(filepath, imgsz=320, conf=0.25)
r = results[0]

if r.probs is None:
    return jsonify({"error": "Model output empty"}), 500

top1 = int(r.probs.top1)
prediction = r.names[top1]

key = prediction.lower().replace("_", " ").strip()

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
# RUN
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
