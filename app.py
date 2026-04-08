from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import os
import uuid
import torch
from PIL import Image

# =========================
# ⚙️ PERFORMANCE
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
# UPLOAD
# =========================
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================
# 🔧 HELPER
# =========================
def split_text(text):
    parts = text.split("\n\n")
    if len(parts) >= 2:
        return parts[0], parts[1]
    return parts[0], parts[0]


def normalize_key(raw):
    key = raw.lower()
    key = key.replace("_", " ").replace("-", " ")

    # 🔥 Fix model naming issues
    key = key.replace("leafcurl", "leaf curl")
    key = key.replace("leafspot", "leaf spot")
    key = key.replace("earlyblight", "early blight")
    key = key.replace("lateblight", "late blight")

    return " ".join(key.split())


# =========================
# 🌿 DISEASE DATABASE (BIG CONTENT)
# =========================
DISEASE_INFO = {

# 🌶 CHILLI
"chilli healthy": {
"description": """The chilli plant is completely healthy with vibrant green leaves, strong stems, and proper branching. There are no visible symptoms of disease, pest attack, or nutrient deficiency. The plant shows good vigor and is capable of producing high yield if maintained properly. Healthy chilli plants usually have consistent growth and balanced foliage development.

மிளகாய் செடி முழுமையாக ஆரோக்கியமாக உள்ளது. இலைகள் பசுமையாகவும் தண்டு வலிமையாகவும் உள்ளது. எந்த நோயும் இல்லை.""",

"treatment": """No treatment is required. However, regular monitoring is essential to detect early signs of pests or diseases. Preventive spraying using organic solutions like neem oil can be done.

சிகிச்சை தேவையில்லை. தொடர்ந்து கண்காணிக்கவும்.""",

"fertilizer": """Apply balanced NPK fertilizer (10:10:10 or 20:20:20) every 10–15 days. Use organic compost or vermicompost to improve soil health. Micronutrients like zinc and magnesium can be added periodically.

NPK உரம் மற்றும் இயற்கை உரம் பயன்படுத்தவும்.""",

"routine": """Water the plant 2–3 times per week depending on weather. Ensure proper sunlight (6–8 hours), good drainage, and weed removal. Avoid overwatering.

வாரத்திற்கு 2–3 முறை நீர் ஊற்றவும்."""
},

"chilli leaf curl": {
"description": """Leaf Curl disease in chilli is caused by viruses transmitted by whiteflies. Symptoms include upward curling of leaves, thickened leaf structure, yellowing, and stunted plant growth. Severe infections reduce flowering and fruit formation drastically.

இலை சுருட்டல் நோய் வைரஸ் மூலம் ஏற்படும். இலைகள் சுருண்டு வளர்ச்சி குறையும்.""",

"treatment": """Remove and destroy infected plants immediately. Control whiteflies using neem oil, Imidacloprid, or Thiamethoxam. Install yellow sticky traps to monitor and reduce insect population.

வேப்பெண்ணெய் அல்லது பூச்சி மருந்து பயன்படுத்தவும்.""",

"fertilizer": """Apply potassium-rich fertilizers and micronutrients such as zinc and boron to enhance plant resistance and recovery.

மைக்ரோ நியூட்ரியன்ட் உரம் பயன்படுத்தவும்.""",

"routine": """Maintain field hygiene, avoid overcrowding, ensure good airflow, and regularly inspect plants for early symptoms.

பூச்சி கண்காணிப்பு அவசியம்."""
},

"chilli leaf spot": {
"description": """Leaf Spot is a fungal disease causing circular brown or black lesions on leaves. As the disease progresses, leaves dry and fall off, reducing photosynthesis and yield.

இது ஒரு பூஞ்சை நோய். இலைகளில் கரும்புள்ளிகள் தோன்றும்.""",

"treatment": """Spray fungicides such as Mancozeb or Copper oxychloride at regular intervals. Remove infected leaves and destroy them away from the field.

பூஞ்சை மருந்து தெளிக்கவும்.""",

"fertilizer": """Use nitrogen and potassium fertilizers to support new healthy growth. Avoid excessive nitrogen.

நைட்ரஜன் + பொட்டாசியம் உரம் பயன்படுத்தவும்.""",

"routine": """Avoid wetting leaves during irrigation. Maintain proper spacing and ensure good air circulation.

இலை ஈரப்பதம் தவிர்க்கவும்."""
},

# 🥜 GROUNDNUT
"groundnut healthy": {
"description": """Groundnut plants are healthy with strong root systems and proper pod formation. Leaves are green and free from any disease symptoms. The crop shows uniform growth and good productivity potential.

நிலக்கடலை செடி ஆரோக்கியமாக உள்ளது.""",

"treatment": """No treatment required. Regular monitoring is sufficient.

சிகிச்சை தேவையில்லை.""",

"fertilizer": """Apply phosphorus-rich fertilizers such as DAP or SSP to improve root and pod development.

பாஸ்பரஸ் உரம் பயன்படுத்தவும்.""",

"routine": """Maintain moderate irrigation and avoid waterlogging. Keep field weed-free.

மிதமான நீர் அளிக்கவும்."""
},

"groundnut leaf spot": {
"description": """Groundnut Leaf Spot is a fungal disease characterized by dark brown spots on leaves. Severe infection leads to defoliation and reduced yield.

பூஞ்சை நோய்.""",

"treatment": """Spray fungicides like Chlorothalonil or Mancozeb at 10-day intervals.

மருந்து தெளிக்கவும்.""",

"fertilizer": """Use balanced NPK fertilizers and apply gypsum to improve pod development.

NPK உரம் பயன்படுத்தவும்.""",

"routine": """Ensure proper plant spacing, avoid high humidity, and monitor crop regularly.

இடைவெளி வைக்கவும்."""
},

# 🍅 TOMATO
"tomato healthy": {
"description": """Tomato plant is healthy with strong stems, dark green leaves, and proper flowering. No disease or pest symptoms are present, and fruit development is normal.

தக்காளி செடி ஆரோக்கியமாக உள்ளது.""",

"treatment": """No treatment required.

சிகிச்சை தேவையில்லை.""",

"fertilizer": """Apply compost, vermicompost, and balanced NPK fertilizers.

உரம் பயன்படுத்தவும்.""",

"routine": """Water regularly, provide support (staking), and ensure proper drainage.

நீர் தேங்க விட வேண்டாம்."""
},

"tomato early blight": {
"description": """Early Blight is a fungal disease causing brown spots with concentric rings. It affects leaves, stems, and fruits, reducing yield significantly.

ஆரம்ப பிளைட் நோய்.""",

"treatment": """Spray Mancozeb or Copper fungicide every 7–10 days. Remove infected leaves.

மருந்து தெளிக்கவும்.""",

"fertilizer": """Use potassium fertilizers and avoid excessive nitrogen.

பொட்டாசியம் உரம் பயன்படுத்தவும்.""",

"routine": """Avoid overhead irrigation, maintain spacing, and ensure airflow.

இலை ஈரம் தவிர்க்கவும்."""
},

"tomato late blight": {
"description": """Late Blight is a severe fungal disease that spreads rapidly in humid conditions. It affects leaves, stems, and fruits, often destroying entire crops.

கடுமையான பூஞ்சை நோய்.""",

"treatment": """Apply Metalaxyl or Copper fungicides immediately. Remove infected plants.

உடனடி மருந்து.""",

"fertilizer": """Apply calcium and potassium fertilizers to improve resistance.

கால்சியம் + பொட்டாசியம்.""",

"routine": """Avoid high humidity, ensure drainage, and monitor weather conditions.

ஈரப்பதம் தவிர்க்கவும்."""
}

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

        # Resize
        img = Image.open(filepath)
        img = img.resize((320, 320))
        img.save(filepath)

        # Predict
        results = model.predict(filepath, imgsz=160, conf=0.25, device='cpu')
        r = results[0]

        if r.probs is None:
            return jsonify({"error": "Model output empty"}), 500

        raw_prediction = r.names[int(r.probs.top1)]

        key = normalize_key(raw_prediction)
        prediction = key.title()

        os.remove(filepath)

        if key not in DISEASE_INFO:
            return jsonify({
                "prediction": prediction,
                "description_en": "No description available.",
                "description_ta": "விளக்கம் இல்லை.",
                "treatment_en": "No treatment available.",
                "treatment_ta": "சிகிச்சை இல்லை.",
                "fertilizer_en": "No fertilizer info.",
                "fertilizer_ta": "உரம் தகவல் இல்லை.",
                "routine_en": "No routine available.",
                "routine_ta": "பராமரிப்பு இல்லை."
            })

        data = DISEASE_INFO[key]

        desc_en, desc_ta = split_text(data["description"])
        treat_en, treat_ta = split_text(data["treatment"])
        fert_en, fert_ta = split_text(data["fertilizer"])
        routine_en, routine_ta = split_text(data["routine"])

        return jsonify({
            "prediction": prediction,
            "description_en": desc_en,
            "description_ta": desc_ta,
            "treatment_en": treat_en,
            "treatment_ta": treat_ta,
            "fertilizer_en": fert_en,
            "fertilizer_ta": fert_ta,
            "routine_en": routine_en,
            "routine_ta": routine_ta,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# RUN
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
