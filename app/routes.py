from flask import Blueprint, render_template, request
import numpy as np
import cv2
from .services.detection import detect_all_models
from .services.utils import img_to_base64
main = Blueprint('main', __name__)
@main.route("/")
def index():
    return render_template("index.html")
@main.route("/predict", methods=["GET", "POST"])
def predict():
    results = []
    if request.method == "POST":
        files = request.files.getlist("images")
        for file in files:
            if file.filename == "":
                continue
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is not None:
                detected_img, count, confidence = detect_all_models(img)
                results.append({
                    "image": img_to_base64(detected_img),
                    "count": count,
                    "confidence": confidence
                })
    return render_template("predict.html", results=results)

@main.route("/model_info")
def model_info():
    return render_template("model_info.html")
@main.route("/performance")
def performance():
    return render_template("performance.html")
