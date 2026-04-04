from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import os
import uuid

app = Flask(__name__)
CORS(app)

# =========================
# 🔥 MODEL PATH (FIXED)
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
# 🌱 HOME ROUTE
# =========================
@app.route("/")
def home():
    return "🌱 PlantIQ API Running Successfully..."

# =========================
# 🔥 PREDICT ROUTE (SAFE VERSION)
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]

        # save image safely
        filename = f"{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # =========================
        # 🤖 YOLO INFERENCE
        # =========================
        results = model(filepath)
        r = results[0]

        # =========================
        # 🔥 SAFE CHECK (PREVENT 500 ERROR)
        # =========================
        if r.probs is None:
            return jsonify({
                "error": "Model output is empty. Check if model is classification."
            }), 500

        probs = r.probs
        names = r.names

        prediction = names[int(probs.top1)]
        confidence = float(probs.top1conf)

        return jsonify({
            "prediction": prediction,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# =========================
# 🚀 RUN SERVER (RENDER READY)
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
