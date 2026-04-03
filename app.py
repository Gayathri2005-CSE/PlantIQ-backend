from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import os
import uuid

app = Flask(__name__)
CORS(app)

# =========================
# 🔥 LOAD MODEL (SAFE PATH)
# =========================
MODEL_PATH = os.path.join(
    "runs",
    "classify",
    "plant_disease_model",
    "weights",
    "best.pt"
)

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
# 🔥 PREDICT ROUTE
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # check file exists
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]

        # safe filename (avoid overwrite issues)
        filename = f"{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        file.save(filepath)

        # =========================
        # 🤖 YOLO PREDICTION
        # =========================
        results = model(filepath)

        probs = results[0].probs
        names = results[0].names

        prediction = names[int(probs.top1)]
        confidence = float(probs.top1conf)

        # =========================
        # 📤 RESPONSE
        # =========================
        return jsonify({
            "prediction": prediction,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# =========================
# 🚀 RUN APP (LOCAL ONLY)
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)