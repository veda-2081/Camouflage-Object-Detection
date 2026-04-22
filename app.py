from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import torch
import os

torch.set_grad_enabled(False)

# Create Flask app FIRST
app = Flask(__name__)

# Configure file upload settings AFTER app is created
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = 'your_secret_key_here'

# Create uploads folder if it doesn't exist
os.makedirs('uploads', exist_ok=True)

# Configuration
IMG_SIZE = 640
MODEL_PATHS = {
    "military": "models/military_assets_dataset_best.pt",
    "uav": "models/hit-uav-best.pt",
    "mudcadx": "models/mudcadx_best.pt",
    "visd": "models/visd-best.pt",
    "snow": "models/best.pt"
}

# Load Models
models = {}
print("Loading models...")
for name, path in MODEL_PATHS.items():
    if os.path.exists(path):
        models[name] = YOLO(path)
        print(f"✅ Loaded: {name}")
    else:
        print(f"⚠️ Warning: Model file not found at {path}")

# Helper: Convert Image to Base64
def img_to_base64(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buff.getvalue()).decode()

# Helper: IoU Calculation
def compute_iou(boxA, boxB):
    inter_x1 = max(boxA["x1"], boxB["x1"])
    inter_y1 = max(boxA["y1"], boxB["y1"])
    inter_x2 = min(boxA["x2"], boxB["x2"])
    inter_y2 = min(boxA["y2"], boxB["y2"])

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    areaA = (boxA["x2"] - boxA["x1"]) * (boxA["y2"] - boxA["y1"])
    areaB = (boxB["x2"] - boxB["x1"]) * (boxB["y2"] - boxB["y1"])
    union = areaA + areaB - inter_area

    if union == 0: return 0
    return inter_area / union

# Core Detection Logic
# Core Detection Logic
def detect_all_models(img):
    if img is None: 
        print("❌ Image is None")
        return None, 0
    
    print(f"📷 Processing image: {img.shape}")
    
    # Resize for inference
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    output_img = img_resized.copy()
    img_h, img_w, _ = output_img.shape
    img_area = img_h * img_w

    final_boxes = []
    model_conf = {
        "military": 0.50, "uav": 0.45, "mudcadx": 0.55, "visd": 0.50, "snow": 0.55
    }

    # Run models
    for model_name, model in models.items():
        try:
            print(f"🔍 Running model: {model_name}")
            results = model(img_resized, conf=model_conf.get(model_name, 0.5), iou=0.6, imgsz=IMG_SIZE, augment=False, verbose=False)[0]
            
            if results.boxes is None:
                print(f"⚠️ No boxes detected by {model_name}")
                continue

            box_count = 0
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label_name = model.names.get(cls, f"class {cls}")
                
                # Filter small or huge boxes
                area = (x2 - x1) * (y2 - y1)
                if area < 2000 or area > 0.65 * img_area: 
                    continue

                new_box = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": conf, "label": label_name}
                
                # Non-Maximum Suppression (Manual)
                ignore = False
                for existing_box in final_boxes:
                    if compute_iou(new_box, existing_box) > 0.5:
                        ignore = True
                        break
                
                if not ignore:
                    final_boxes.append(new_box)
                    box_count += 1
            
            print(f"✅ {model_name}: {box_count} boxes")
        except Exception as e:
            print(f"❌ Error running {model_name}: {e}")

    # Draw Results
    for box in final_boxes:
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        score = box["score"]
        label = box["label"]

        # Color based on confidence
        if score > 0.85: color = (0, 255, 0)
        elif score > 0.70: color = (255, 165, 0)
        else: color = (255, 255, 0)

        cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)
        
        # Draw Label Background
        text = f"{label} | {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(output_img, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
        cv2.putText(output_img, text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    print(f"🎯 Total boxes detected: {len(final_boxes)}")
    return output_img, len(final_boxes)

# --- Routes ---

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    results = []
    
    if request.method == "POST":
        # Check if files exist
        if 'image' not in request.files:
            print("No file part in request")
            return "No file part", 400
        
        files = request.files.getlist("image")
        print(f"Received {len(files)} files")
        
        for file in files:
            if file.filename == "":
                continue
            
            # Validate file extension
            if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                print(f"Invalid file type: {file.filename}")
                continue
            
            try:
                file_bytes = np.frombuffer(file.read(), np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                if img is not None:
                    detected_img, count = detect_all_models(img)
                    if detected_img is not None:
                        results.append({
                            "image": img_to_base64(detected_img),
                            "count": count
                        })
                        print(f"Processed: {file.filename}, Count: {count}")
            except Exception as e:
                print(f"Error processing file {file.filename}: {e}")
                continue
    
    return render_template("predict.html", results=results)

@app.route("/model_info")
def model_info():
    return render_template("model_info.html")

@app.route("/performance")
def performance():
    return render_template("performance.html")
# Add to app.py
@app.route("/static/images/hero.jpg")
def serve_hero():
    # Create a simple placeholder
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    img[:] = [0, 102, 255]  # Blue background
    return img_to_base64(img)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)