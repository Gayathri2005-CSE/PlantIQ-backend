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
# 🌱 BIG DISEASE INFO DATABASE
# =========================
DISEASE_INFO = {

    # ================= CHILLI =================

    "chilli healthy": {
        "disease_name": "Chilli Healthy / ஆரோக்கியமான மிளகாய்",
        "description": "The chilli plant is in excellent healthy condition with strong green leaves, proper flowering, and normal growth. There are no visible symptoms of pests or diseases. This indicates balanced soil nutrients and proper environmental conditions.\n\nமிளகாய் செடி முழுமையாக ஆரோக்கியமாக உள்ளது. இலைகள் பசுமையாகவும், வளர்ச்சி சரியாகவும் உள்ளது.",
        "treatment": "No chemical treatment is required. Maintain natural plant care and continue regular monitoring for pests or disease prevention.\n\nஎந்த மருந்தும் தேவையில்லை. இயற்கை பராமரிப்பு மட்டும் போதுமானது.",
        "fertilizer": "Apply well-decomposed organic compost along with balanced NPK fertilizer (10:10:10) every 15–20 days to maintain soil fertility and strong plant growth.\n\n15–20 நாட்களுக்கு ஒருமுறை இயற்கை உரம் மற்றும் NPK உரம் இடவும்.",
        "routine": "Ensure 5–6 hours of sunlight daily, water the plant based on soil moisture, and inspect leaves weekly for early pest detection.\n\nதொடர்ந்து வெயில் மற்றும் நீர் வழங்கவும், வாரந்தோறும் பரிசோதிக்கவும்."
    },

    "chilli leafspot": {
        "disease_name": "Chilli Leaf Spot / மிளகாய் இலை புள்ளி நோய்",
        "description": "Leaf spot disease is a fungal infection that appears as small dark brown or black circular spots on leaves. As the disease progresses, leaves may turn yellow, dry, and fall off, reducing plant productivity and growth.\n\nபூஞ்சை காரணமாக இலைகளில் கருப்பு/பழுப்பு புள்ளிகள் உருவாகி செடி பாதிக்கப்படுகிறது.",
        "treatment": "Immediately remove infected leaves and spray copper oxychloride or organic neem oil solution every 7–10 days to prevent further fungal spread.\n\nபாதிக்கப்பட்ட இலைகளை அகற்றி, வேப்பெண்ணெய் அல்லது தாமிர பூஞ்சை மருந்து தெளிக்கவும்.",
        "fertilizer": "Use potassium-rich fertilizer and organic manure to improve plant immunity and resistance against fungal infections.\n\nபொட்டாசியம் அதிகமான உரம் மற்றும் இயற்கை உரம் பயன்படுத்தவும்.",
        "routine": "Avoid overhead watering, ensure proper air circulation between plants, and keep leaves dry to prevent fungal growth.\n\nஇலைகளில் நீர் விடாமல் காற்றோட்டம் வைத்திருக்கவும்."
    },

    "chilli leafcurl": {
        "disease_name": "Chilli Leaf Curl / மிளகாய் இலை சுருட்டல்",
        "description": "Leaf curl disease is a viral infection mainly transmitted by whiteflies. It causes severe curling, twisting, and stunted growth of leaves, leading to reduced flowering and fruit production.\n\nவெள்ளை ஈ மூலம் பரவும் வைரஸ் நோய் காரணமாக இலைகள் சுருண்டு வளர்ச்சி குறைகிறது.",
        "treatment": "Spray neem oil weekly and use yellow sticky traps to control whiteflies. Remove heavily infected plants to prevent spreading.\n\nவேப்பெண்ணெய் தெளிக்கவும், வெள்ளை ஈ கட்டுப்படுத்தவும்.",
        "fertilizer": "Apply balanced NPK fertilizer along with micronutrients like zinc and magnesium to improve plant recovery.\n\nNPK மற்றும் சிறு ஊட்டச்சத்துக்கள் பயன்படுத்தவும்.",
        "routine": "Inspect plants twice a week, avoid excessive watering, and maintain clean field conditions.\n\nவாரத்தில் 2 முறை பரிசோதனை செய்யவும்."
    },

    # ================= GROUNDNUT =================

    "groundnut healthy": {
        "disease_name": "Groundnut Healthy / ஆரோக்கியமான நிலக்கடலை",
        "description": "The groundnut plant is healthy with strong green leaves, proper root development, and good flowering. This indicates fertile soil and proper agricultural management.\n\nநிலக்கடலை செடி ஆரோக்கியமாக உள்ளது.",
        "treatment": "No treatment required. Continue regular monitoring.\n\nசிகிச்சை தேவையில்லை.",
        "fertilizer": "Apply gypsum, farmyard manure, and phosphorus-rich fertilizer for better pod development.\n\nஜிப்சம் மற்றும் இயற்கை உரம் பயன்படுத்தவும்.",
        "routine": "Maintain moderate irrigation and ensure proper sunlight exposure for maximum yield.\n\nசரியான நீர் மற்றும் வெயில் வழங்கவும்."
    },

    "groundnut leafspot": {
        "disease_name": "Groundnut Leaf Spot / நிலக்கடலை இலை புள்ளி நோய்",
        "description": "Leaf spot disease is caused by fungal infection leading to dark circular spots on leaves. Severe infection may cause early leaf drop and reduce crop yield significantly.\n\nபூஞ்சை காரணமாக இலைகளில் புள்ளிகள் உருவாகி விளைச்சல் குறைகிறது.",
        "treatment": "Spray mancozeb or copper-based fungicides every 10–14 days to control fungal spread.\n\nமாங்கோசெப் பூஞ்சை மருந்து பயன்படுத்தவும்.",
        "fertilizer": "Use potassium and phosphorus-rich fertilizers to strengthen plant immunity.\n\nபொட்டாசியம் மற்றும் பாஸ்பரஸ் உரம்.",
        "routine": "Remove infected leaves and avoid excess moisture in soil.\n\nபாதிக்கப்பட்ட இலைகளை அகற்றவும்."
    },

    # ================= TOMATO =================

    "tomato early blight": {
        "disease_name": "Tomato Early Blight / தக்காளி ஆரம்ப ப்ளைட்",
        "description": "Early blight is a fungal disease that starts as small brown spots on older leaves and gradually spreads, causing defoliation and reduced fruit production.\n\nபழைய இலைகளில் பழுப்பு புள்ளிகள் உருவாகி செடி பாதிக்கப்படுகிறது.",
        "treatment": "Apply copper oxychloride or approved fungicides immediately and remove infected leaves.\n\nதாமிர பூஞ்சை மருந்து பயன்படுத்தவும்.",
        "fertilizer": "Use potassium-rich fertilizer to improve plant resistance and fruit quality.\n\nபொட்டாசியம் உரம் பயன்படுத்தவும்.",
        "routine": "Avoid wet leaves, maintain proper spacing, and ensure good air circulation.\n\nஇலைகளை நனைக்காமல் வைத்திருக்கவும்."
    },

    "tomato healthy": {
        "disease_name": "Tomato Healthy / ஆரோக்கியமான தக்காளி",
        "description": "The tomato plant is healthy with strong stems, green leaves, and proper flowering. It shows good nutrient balance and growth conditions.\n\nதக்காளி செடி ஆரோக்கியமாக உள்ளது.",
        "treatment": "No treatment required. Maintain regular care.\n\nசிகிச்சை தேவையில்லை.",
        "fertilizer": "Apply balanced NPK fertilizer every 15 days for better yield and fruit quality.\n\n15 நாட்களுக்கு ஒருமுறை NPK உரம்.",
        "routine": "Provide consistent watering and full sunlight exposure for healthy growth.\n\nதொடர்ந்து நீர் மற்றும் வெயில்."
    },

    "tomato late blight": {
        "disease_name": "Tomato Late Blight / தக்காளி கடைசி ப்ளைட்",
        "description": "Late blight is a severe fungal disease that spreads rapidly under humid conditions, causing leaf rot, stem damage, and total plant collapse if not controlled early.\n\nகடுமையான பூஞ்சை நோய் காரணமாக செடி அழியும்.",
        "treatment": "Apply strong fungicides immediately and remove infected plants to prevent spread to other crops.\n\nவலுவான பூஞ்சை மருந்து பயன்படுத்தவும்.",
        "fertilizer": "Use calcium and potassium-rich fertilizer to strengthen plant structure and disease resistance.\n\nகால்சியம் மற்றும் பொட்டாசியம் உரம்.",
        "routine": "Avoid water stagnation, maintain dry field conditions, and improve drainage system.\n\nதண்ணீர் தேங்காமல் பார்த்துக் கொள்ளவும்."
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
            return jsonify({"error": "Invalid model output"}), 500

        prediction = r.names[int(r.probs.top1)]
        key = prediction.lower().replace("_", " ")

        info = DISEASE_INFO.get(key, {
            "disease_name": prediction,
            "description": "No information available",
            "treatment": "Not available",
            "fertilizer": "Not available",
            "routine": "Not available"
        })

        return jsonify({
            "prediction": info["disease_name"],
            "description": info["description"],
            "treatment": info["treatment"],
            "fertilizer": info["fertilizer"],
            "routine": info["routine"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# 🚀 RUN SERVER
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
