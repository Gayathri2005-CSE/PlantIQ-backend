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
# 🌿 DISEASE DATABASE (VERY DETAILED)
# =========================
DISEASE_INFO = {

# 🌶️ CHILLI
"chilli healthy": {
"description": """
The chilli plant is completely healthy with vibrant green leaves, strong stems, and proper growth. There are no signs of disease, pest attack, discoloration, or nutrient deficiency. Leaves are actively performing photosynthesis and plant development is optimal.

மிளகாய் செடி முழுமையாக ஆரோக்கியமாக உள்ளது. இலைகள் பசுமையாகவும் உறுதியானதாகவும் உள்ளன. எந்த நோய் அல்லது பூச்சி தாக்கமும் இல்லை.
""",
"treatment": """
No treatment is required. Continue good agricultural practices such as regular monitoring, clean environment, and proper watering.

சிகிச்சை தேவையில்லை. செடியை வழக்கமாக பராமரிக்கவும்.
""",
"fertilizer": """
Apply balanced NPK fertilizer (10:10:10 or 20:20:20) every 10–15 days. Add organic compost, vermicompost, or farmyard manure to maintain soil fertility.

ஒவ்வொரு 10–15 நாட்களுக்கும் NPK உரம் பயன்படுத்தவும். இயற்கை உரம் சேர்க்கவும்.
""",
"routine": """
Water 2–3 times per week depending on soil moisture. Ensure 6–8 hours of sunlight daily. Remove weeds regularly.

வாரத்திற்கு 2–3 முறை நீர் ஊற்றவும். நல்ல வெளிச்சம் கிடைக்க செய்யவும்.
"""
},

"chilli leaf curl": {
"description": """
Leaf Curl disease is caused by a viral infection transmitted by whiteflies. Leaves curl upward, become thick and deformed. Plant growth becomes stunted and yield reduces drastically.

இலை சுருட்டல் நோய் வைரஸ் காரணமாக ஏற்படும். இலைகள் சுருண்டு வளர்ச்சி குறையும்.
""",
"treatment": """
Remove infected leaves immediately. Spray neem oil or insecticides like Imidacloprid. Control whiteflies using yellow sticky traps.

பாதிக்கப்பட்ட இலைகளை அகற்றவும். வேப்பெண்ணெய் அல்லது பூச்சி மருந்து தெளிக்கவும்.
""",
"fertilizer": """
Use potassium-rich fertilizers and micronutrients like Zinc and Magnesium to strengthen plant immunity.

பொட்டாசியம் மற்றும் மைக்ரோ ஊட்டச்சத்து உரம் பயன்படுத்தவும்.
""",
"routine": """
Maintain field hygiene, avoid overcrowding, and monitor regularly for pests. Use pest traps.

வயலை சுத்தமாக வைத்திருக்கவும்.
"""
},

"chilli leafspot": {
"description": """
Leaf spot is a fungal disease that causes brown or black circular spots on leaves. Severe infection leads to leaf drop and reduced yield.

இலை புள்ளி நோய் பூஞ்சை காரணமாக ஏற்படும்.
""",
"treatment": """
Spray fungicides like Mancozeb or Copper oxychloride. Remove infected leaves and destroy them.

பூஞ்சை மருந்து தெளிக்கவும்.
""",
"fertilizer": """
Apply nitrogen and potassium fertilizers to promote new leaf growth.

நைட்ரஜன் + பொட்டாசியம் உரம்.
""",
"routine": """
Avoid water on leaves, improve air circulation, and maintain spacing between plants.

இலை ஈரம் தவிர்க்கவும்.
"""
},

# 🌰 GROUNDNUT
"groundnut healthy": {
"description": """
Groundnut plant is healthy with strong root system, green leaves, and good pod formation. Growth is uniform and free from disease.

நிலக்கடலை செடி ஆரோக்கியமாக உள்ளது.
""",
"treatment": """
No treatment required. Maintain proper irrigation and soil care.

சிகிச்சை தேவையில்லை.
""",
"fertilizer": """
Apply phosphorus-rich fertilizers and gypsum for better pod formation.

பாஸ்பரஸ் உரம் பயன்படுத்தவும்.
""",
"routine": """
Water moderately and avoid over-irrigation. Maintain weed-free field.

மிதமான நீர்.
"""
},

"groundnut leafspot": {
"description": """
Leaf spot is a fungal infection causing dark circular spots. Leads to premature leaf fall and reduced yield.

பூஞ்சை நோய்.
""",
"treatment": """
Spray Mancozeb or Chlorothalonil regularly.

பூஞ்சை மருந்து தெளிக்கவும்.
""",
"fertilizer": """
Use balanced NPK fertilizer to improve plant strength.

சமமான NPK உரம்.
""",
"routine": """
Maintain proper spacing and avoid high humidity.

இடைவெளி வைக்கவும்.
"""
},

# 🍅 TOMATO
"tomato healthy": {
"description": """
Tomato plant is healthy with strong stems, green leaves, and proper flowering and fruit development.

தக்காளி செடி ஆரோக்கியமாக உள்ளது.
""",
"treatment": """
No treatment required. Continue monitoring and care.

சிகிச்சை தேவையில்லை.
""",
"fertilizer": """
Apply compost and NPK fertilizers regularly for better yield.

இயற்கை உரம் + NPK.
""",
"routine": """
Water regularly but avoid waterlogging.

நீர் தேங்க விட வேண்டாம்.
"""
},

"tomato early blight": {
"description": """
Early blight is a fungal disease that causes concentric brown spots on leaves and stems. Reduces plant productivity.

ஆரம்ப பிளைட் நோய்.
""",
"treatment": """
Spray Mancozeb or Copper fungicides early.

பூஞ்சை மருந்து தெளிக்கவும்.
""",
"fertilizer": """
Use calcium and potassium fertilizers.

கால்சியம் + பொட்டாசியம்.
""",
"routine": """
Maintain spacing and avoid wet leaves.

இடைவெளி வைக்கவும்.
"""
},

"tomato late blight": {
"description": """
Late blight is a severe disease that spreads rapidly causing dark lesions and plant decay.

கடுமையான பூஞ்சை நோய்.
""",
"treatment": """
Apply Metalaxyl or Copper-based fungicides immediately.

உடனடி மருந்து தெளிக்கவும்.
""",
"fertilizer": """
Use potassium-rich fertilizers.

பொட்டாசியம் உரம்.
""",
"routine": """
Avoid humidity and ensure good drainage.

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
    filepath = None
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]

        filename = f"{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # resize (memory optimization)
        img = Image.open(filepath)
        img = img.resize((224, 224))
        img.save(filepath)

        results = model.predict(filepath, imgsz=160, conf=0.25, device='cpu')
        r = results[0]

        if r.probs is None:
            return jsonify({"error": "Model output empty"}), 500

        prediction = r.names[int(r.probs.top1)]

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

    finally:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)

# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
