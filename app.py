from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import os
import uuid

app = Flask(__name__)
CORS(app)

# =========================
# 🔥 MODEL PATH
# =========================
MODEL_PATH = "models/best.pt"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model = YOLO(MODEL_PATH)

# =========================
# 📁 UPLOAD FOLDER
# =========================
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================
# 🌿 DISEASE DATABASE (BIG CONTENT)
# =========================
DISEASE_INFO = {
    "chilli healthy": {
        "description": "The chilli plant is healthy with no visible disease symptoms. Leaves are green, fresh, and actively photosynthesizing. / மிளகாய் செடி ஆரோக்கியமாக உள்ளது. இலைகள் பசுமையாகவும் உறுதியானதாகவும் உள்ளன.",
        "treatment": "No chemical treatment required. Maintain good farm hygiene and monitor regularly. / எந்த சிகிச்சையும் தேவையில்லை. செடியை கண்காணிக்கவும்.",
        "fertilizer": "Apply balanced NPK fertilizer (10:10:10) every 15 days. Add organic compost for better growth. / சமமான NPK உரம் பயன்படுத்தவும்.",
        "routine": "Water twice a week. Ensure proper sunlight and avoid water stagnation. / வாரத்தில் 2 முறை நீர் ஊற்றவும்."
    },

    "chilli leaf curl": {
        "description": "Leaf Curl disease is caused by virus transmitted by whiteflies. Leaves become curled, twisted and growth is reduced. / வைரஸ் காரணமாக இலை சுருட்டல் ஏற்படுகிறது.",
        "treatment": "Remove infected leaves immediately. Spray neem oil or insecticidal soap. Control whiteflies. / பாதிக்கப்பட்ட இலைகளை அகற்றவும்.",
        "fertilizer": "Use potassium-rich fertilizer and micronutrients like zinc and magnesium. / பொட்டாசியம் அதிகம் உள்ள உரம் பயன்படுத்தவும்.",
        "routine": "Avoid overwatering. Keep field clean and use yellow sticky traps. / அதிக நீர் ஊற்ற வேண்டாம்."
    },

    "chilli leafspot": {
        "description": "Leaf spot is a fungal disease causing brown and black spots on leaves. It spreads quickly in humid conditions. / பூஞ்சை நோய் காரணமாக இலை புள்ளிகள் உருவாகும்.",
        "treatment": "Spray copper-based fungicide. Remove infected leaves. / பூஞ்சை மருந்து தெளிக்கவும்.",
        "fertilizer": "Apply nitrogen + potassium fertilizer for recovery. / நைட்ரஜன் மற்றும் பொட்டாசியம் உரம்.",
        "routine": "Avoid wet leaves and improve air circulation. / இலைகளை ஈரமாக விட வேண்டாம்."
    },

    "groundnut healthy": {
        "description": "Groundnut plant is healthy with strong roots and green leaves. / நிலக்கடலை செடி ஆரோக்கியமாக உள்ளது.",
        "treatment": "No treatment required. Maintain soil moisture. / சிகிச்சை தேவையில்லை.",
        "fertilizer": "Use phosphorus-rich fertilizer for better pod development. / பாஸ்பரஸ் உரம்.",
        "routine": "Water once or twice a week depending on soil. / வாரத்திற்கு 1-2 முறை நீர்."
    },

    "groundnut leafspot": {
        "description": "Fungal infection causing circular dark spots on leaves leading to defoliation. / இலை புள்ளி நோய்.",
        "treatment": "Spray fungicide like Mancozeb. Remove infected parts. / பூஞ்சை மருந்து தெளிக்கவும்.",
        "fertilizer": "Use nitrogen and potassium fertilizers. / நைட்ரஜன் + பொட்டாசியம்.",
        "routine": "Avoid overcrowding plants. / செடிகளை அடர்த்தியாக வளர விட வேண்டாம்."
    },

    "tomato early blight": {
        "description": "Early blight causes brown concentric spots on leaves and reduces yield. / ஆரம்ப பிளைட் நோய்.",
        "treatment": "Use copper oxychloride spray and remove infected leaves. / பூஞ்சை மருந்து.",
        "fertilizer": "Apply balanced NPK fertilizer and calcium supplements. / சமமான உரம்.",
        "routine": "Avoid wet leaves and maintain spacing. / இடைவெளி வைக்கவும்."
    },

    "tomato healthy": {
        "description": "Tomato plant is healthy with strong stems and green leaves. / தக்காளி செடி ஆரோக்கியமாக உள்ளது.",
        "treatment": "No treatment required. Continue monitoring. / சிகிச்சை தேவையில்லை.",
        "fertilizer": "Use organic compost and NPK fertilizer every 2 weeks. / இயற்கை உரம்.",
        "routine": "Water regularly but avoid waterlogging. / நீர் அதிகமாக விட வேண்டாம்."
    },

    "tomato late blight": {
        "description": "Late blight is a severe fungal disease causing dark lesions and plant decay. / கடைசி நிலை பூஞ்சை நோய்.",
        "treatment": "Immediate fungicide spray (Metalaxyl or Copper based). / உடனடி மருந்து தெளிக்கவும்.",
        "fertilizer": "Use potassium-rich fertilizer to improve resistance. / பொட்டாசியம் உரம்.",
        "routine": "Avoid humidity and ensure good drainage. / ஈரப்பதம் தவிர்க்கவும்."
    }
}

# =========================
# 🌱 HOME ROUTE
# =========================
@app.route("/")
def home():
    return "🌱 PlantIQ API Running Successfully..."

# =========================
# 🔥 PREDICT ROUTE
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

        results = model(filepath)
        r = results[0]

        if r.probs is None:
            return jsonify({"error": "Model output is empty"}), 500

        prediction = r.names[int(r.probs.top1)]

        # 🔥 normalize key
        key = prediction.lower().replace("_", " ").strip()

        if key not in DISEASE_INFO:
            return jsonify({
                "prediction": prediction,
                "confidence": confidence,
                "error": "Disease info not found in database"
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
# 🚀 RUN SERVER
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
