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
# 🌿 DISEASE DATABASE (FULL EXTENDED)
# =========================
DISEASE_INFO = {

# 🌶 CHILLI
"chilli healthy": {
"description": """The chilli plant is in excellent health with lush green leaves, strong stem, and active growth. No disease or pest attack is visible. It has high yield potential.

மிளகாய் செடி முழுமையாக ஆரோக்கியமாக உள்ளது. எந்த நோயும் இல்லை.""",
"treatment": """No treatment required. Continue monitoring.

சிகிச்சை தேவையில்லை.""",
"fertilizer": """Apply balanced NPK fertilizer regularly and add compost.

NPK உரம் பயன்படுத்தவும்.""",
"routine": """Water 2–3 times weekly and ensure sunlight.

வாரத்திற்கு 2–3 முறை நீர் ஊற்றவும்."""
},

"chilli leaf curl": {
"description": """Leaf Curl disease is caused by virus spread by whiteflies. Leaves curl and plant growth reduces drastically.

இலை சுருட்டல் நோய் வைரஸ் மூலம் ஏற்படும்.""",
"treatment": """Remove infected plants. Spray neem oil or Imidacloprid.

வேப்பெண்ணெய் தெளிக்கவும்.""",
"fertilizer": """Use potassium and micronutrients.

பொட்டாசியம் உரம் பயன்படுத்தவும்.""",
"routine": """Maintain hygiene and monitor pests.

பூச்சி கண்காணிக்கவும்."""
},

"chilli leaf spot": {
"description": """Leaf Spot is a fungal disease causing brown spots and leaf drop.

இது பூஞ்சை நோய்.""",
"treatment": """Spray Mancozeb or Copper fungicide.

மருந்து தெளிக்கவும்.""",
"fertilizer": """Use nitrogen + potassium.

நைட்ரஜன் உரம்.""",
"routine": """Avoid wet leaves and maintain spacing.

இலை ஈரம் தவிர்க்கவும்."""
},

# 🥜 GROUNDNUT
"groundnut healthy": {
"description": """Groundnut plant is healthy with proper root and pod development.

நிலக்கடலை செடி ஆரோக்கியமாக உள்ளது.""",
"treatment": """No treatment required.

சிகிச்சை தேவையில்லை.""",
"fertilizer": """Apply phosphorus fertilizer.

பாஸ்பரஸ் உரம்.""",
"routine": """Avoid water stagnation.

நீர் தேங்க விட வேண்டாம்."""
},

"groundnut leaf spot": {
"description": """Fungal disease causing dark spots and yield loss.

பூஞ்சை நோய்.""",
"treatment": """Spray fungicides regularly.

மருந்து தெளிக்கவும்.""",
"fertilizer": """Use NPK fertilizers.

NPK உரம்.""",
"routine": """Maintain spacing and airflow.

இடைவெளி வைக்கவும்."""
},

# 🍅 TOMATO
"tomato healthy": {
"description": """Tomato plant is healthy with strong growth and fruiting.

தக்காளி செடி ஆரோக்கியமாக உள்ளது.""",
"treatment": """No treatment required.

சிகிச்சை தேவையில்லை.""",
"fertilizer": """Apply compost and NPK.

உரம் பயன்படுத்தவும்.""",
"routine": """Water regularly and avoid waterlogging.

நீர் தேங்க விட வேண்டாம்."""
},

"tomato early blight": {
"description": """Fungal disease with brown concentric spots. Reduces yield.

ஆரம்ப பிளைட் நோய்.""",
"treatment": """Spray Mancozeb every 7–10 days.

மருந்து தெளிக்கவும்.""",
"fertilizer": """Use potassium fertilizer.

பொட்டாசியம் உரம்.""",
"routine": """Remove infected leaves and avoid wetting.

இலை அகற்றவும்."""
},

"tomato late blight": {
"description": """Severe fungal disease spreading rapidly in humidity.

கடுமையான பூஞ்சை நோய்.""",
"treatment": """Use Metalaxyl or Copper fungicide immediately.

உடனடி மருந்து.""",
"fertilizer": """Apply calcium and potassium.

கால்சியம் உரம்.""",
"routine": """Avoid humidity and ensure drainage.

ஈரப்பதம் தவிர்க்கவும்."""
},

}

# =========================
# HOME
# =========================
@app.route("/")
def home():
    return "🌱 PlantIQ API Running..."

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

        # Resize image
        img = Image.open(filepath)
        img = img.resize((320, 320))
        img.save(filepath)

        # Model prediction
        results = model.predict(filepath, imgsz=160, conf=0.25, device='cpu')
        r = results[0]

        if r.probs is None:
            return jsonify({"error": "Model output empty"}), 500

        raw_prediction = r.names[int(r.probs.top1)]

        prediction = raw_prediction.replace("_", " ").title()
        key = raw_prediction.lower().replace("_", " ").strip()

        print("Prediction:", key)

        os.remove(filepath)

        if key not in DISEASE_INFO:
            return jsonify({
                "prediction": prediction,
                "error": "Disease info not found"
            })

        data = DISEASE_INFO[key]

        # 🔥 SPLIT ENGLISH & TAMIL
        return jsonify({
            "prediction": prediction,

            "description_en": data["description"].split("\n\n")[0],
            "description_ta": data["description"].split("\n\n")[1],

            "treatment_en": data["treatment"].split("\n\n")[0],
            "treatment_ta": data["treatment"].split("\n\n")[1],

            "fertilizer_en": data["fertilizer"].split("\n\n")[0],
            "fertilizer_ta": data["fertilizer"].split("\n\n")[1],

            "routine_en": data["routine"].split("\n\n")[0],
            "routine_ta": data["routine"].split("\n\n")[1],
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
