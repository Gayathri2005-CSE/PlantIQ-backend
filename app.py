from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import os
import uuid
import torch

app = Flask(__name__)
CORS(app)

# =========================
# 🔥 SPEED + MEMORY FIX
# =========================
torch.set_num_threads(1)

# =========================
# 🔥 MODEL PATH
# =========================
MODEL_PATH = "models/best.pt"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model = YOLO(MODEL_PATH)
model.fuse()  # improves speed + reduces memory usage

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
# 🔥 PREDICT ROUTE (OPTIMIZED)
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

        # =========================
        # 🤖 OPTIMIZED INFERENCE
        # =========================
        results = model.predict(
            source=filepath,
            imgsz=320,
            conf=0.25,
            device="cpu",
            verbose=False
        )

        r = results[0]

        if r.probs is None:
            return jsonify({
                "error": "Model is not classification or output missing"
            }), 500

        prediction = r.names[int(r.probs.top1)]
        confidence = float(r.probs.top1conf)

        return jsonify({
            "prediction": prediction,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# 🚀 RUN (RENDER SAFE)
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
