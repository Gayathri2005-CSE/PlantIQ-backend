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
# 🌿 DISEASE DATABASE (FULL CONTENT)
# =========================
DISEASE_INFO = {

# ================= CHILLI =================

"chilli healthy": {
"description": """Chilli plant is in excellent condition with lush green leaves, strong stem, and active growth. No visible disease, pest damage, or nutrient deficiency is observed. The plant is capable of producing high yield under proper care.

மிளகாய் செடி முழுமையாக ஆரோக்கியமாக உள்ளது. இலைகள் பசுமையாகவும் வளர்ச்சி சிறப்பாகவும் உள்ளது. எந்த நோயும் இல்லை.""",

"treatment": """No treatment required. Maintain current care practices and monitor regularly to prevent future infections.

சிகிச்சை தேவையில்லை. தொடர்ந்து கண்காணிக்கவும்.""",

"fertilizer": """Apply balanced NPK fertilizer (10:10:10 or 20:20:20) every 15 days. Add compost or organic manure to improve soil fertility.

NPK உரம் மற்றும் இயற்கை உரம் பயன்படுத்தவும்.""",

"routine": """Water 2–3 times per week depending on weather. Ensure good sunlight (6–8 hours daily) and proper drainage.

வாரத்திற்கு 2–3 முறை நீர் ஊற்றவும்."""
},

"chilli leaf curl": {
"description": """Leaf Curl disease is caused by viral infection spread by whiteflies. Leaves curl upward, shrink, and plant growth becomes stunted. Yield decreases drastically if not controlled early.

இலை சுருட்டல் நோய் வைரஸ் மூலம் ஏற்படும். இலைகள் சுருண்டு வளர்ச்சி குறையும்.""",

"treatment": """Remove infected plants immediately. Spray neem oil or Imidacloprid to control whiteflies. Use yellow sticky traps.

வேப்பெண்ணெய் அல்லது பூச்சி மருந்து பயன்படுத்தவும்.""",

"fertilizer": """Apply potassium-rich fertilizers and micronutrients like Zinc and Magnesium to strengthen plant resistance.

மைக்ரோ நியூட்ரியன்ட் உரம் பயன்படுத்தவும்.""",

"routine": """Maintain field hygiene, avoid overcrowding, and monitor pests regularly.

பூச்சி கண்காணிப்பு அவசியம்."""
},

"chilli leafspot": {
"description": """Leaf Spot is a fungal disease causing small brown or black spots on leaves. Severe infection leads to leaf drop and reduced yield.

இது ஒரு பூஞ்சை நோய். இலைகளில் புள்ளிகள் தோன்றும்.""",

"treatment": """Spray Mancozeb or Copper fungicide. Remove infected leaves and destroy them.

பூஞ்சை மருந்து தெளிக்கவும்.""",

"fertilizer": """Use nitrogen and potassium fertilizers to promote new healthy leaf growth.

நைட்ரஜன் + பொட்டாசியம் உரம்.""",

"routine": """Avoid water on leaves, ensure proper spacing and airflow.

இலை ஈரப்பதம் தவிர்க்கவும்."""
},

# ================= GROUNDNUT =================

"groundnut healthy": {
"description": """Groundnut plant is healthy with good root development, green leaves, and proper pod formation. No disease or stress symptoms are visible.

நிலக்கடலை செடி ஆரோக்கியமாக உள்ளது.""",

"treatment": """No treatment required. Continue regular monitoring.

சிகிச்சை தேவையில்லை.""",

"fertilizer": """Apply phosphorus-rich fertilizer for better root and pod development.

பாஸ்பரஸ் உரம் பயன்படுத்தவும்.""",

"routine": """Maintain moderate irrigation and avoid water stagnation.

மிதமான நீர்."""
},

"groundnut leafspot": {
"description": """Leaf Spot is a fungal disease causing dark circular spots leading to leaf fall and yield reduction.

இது பூஞ்சை நோய்.""",

"treatment": """Spray Chlorothalonil or Mancozeb regularly.

பூஞ்சை மருந்து தெளிக்கவும்.""",

"fertilizer": """Use balanced NPK fertilizers to strengthen plant immunity.

NPK உரம் பயன்படுத்தவும்.""",

"routine": """Maintain spacing and avoid humidity buildup.

இடைவெளி வைக்கவும்."""
},

# ================= TOMATO =================

"tomato healthy": {
"description": """Tomato plant is fully healthy with strong stems, vibrant leaves, proper flowering, and fruit development. No pest or disease symptoms observed.

தக்காளி செடி ஆரோக்கியமாக உள்ளது.""",

"treatment": """No treatment required. Maintain regular care.

சிகிச்சை தேவையில்லை.""",

"fertilizer": """Apply compost and balanced NPK fertilizer regularly for better yield.

இயற்கை உரம் + NPK.""",

"routine": """Water regularly but avoid waterlogging. Ensure sunlight and airflow.

நீர் தேங்க விட வேண்டாம்."""
},

"tomato early blight": {
"description": """Early Blight is a fungal disease causing brown spots with concentric rings on leaves and stems. It reduces plant productivity.

ஆரம்ப பிளைட் நோய்.""",

"treatment": """Spray Mancozeb or Copper fungicide every 7–10 days. Remove infected leaves.

பூஞ்சை மருந்து தெளிக்கவும்.""",

"fertilizer": """Use potassium-rich fertilizers and avoid excess nitrogen.

பொட்டாசியம் உரம் பயன்படுத்தவும்.""",

"routine": """Avoid overhead watering and maintain spacing.

இலை ஈரம் தவிர்க்கவும்."""
},

"tomato late blight": {
"description": """Late Blight is a severe fungal disease that spreads quickly and destroys leaves, stems, and fruits.

கடுமையான பூஞ்சை நோய்.""",

"treatment": """Use Metalaxyl or Copper fungicide immediately. Remove infected plants.

உடனடியாக மருந்து தெளிக்கவும்.""",

"fertilizer": """Apply potassium and calcium fertilizers to improve resistance.

பொட்டாசியம் + கால்சியம்.""",

"routine": """Avoid high humidity and ensure proper drainage.

ஈரப்பதம் தவிர்க்கவும்."""
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

        img = Image.open(filepath)
        img = img.resize((320, 320))
        img.save(filepath)

        results = model.predict(filepath, imgsz=160, conf=0.25, device='cpu')
        r = results[0]

        if r.probs is None:
            return jsonify({"error": "Model output empty"}), 500

        raw_prediction = r.names[int(r.probs.top1)]

        prediction = raw_prediction.replace("_", " ").replace("-", " ").title()

        key = raw_prediction.lower().replace("_", " ").replace("-", " ").strip()

        os.remove(filepath)

        if key not in DISEASE_INFO:
            return jsonify({
                "prediction": prediction,
                "error": "Disease info not found"
            })

        return jsonify({
            "prediction": prediction,
            "description": DISEASE_INFO[key]["description"].strip(),
            "treatment": DISEASE_INFO[key]["treatment"].strip(),
            "fertilizer": DISEASE_INFO[key]["fertilizer"].strip(),
            "routine": DISEASE_INFO[key]["routine"].strip()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
