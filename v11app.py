from flask import Flask, render_template, request
from ultralytics import YOLO
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# ===============================
# Load YOLOv11 model
# ===============================
model = YOLO("military_assets_dataset_best.pt")   # 🔁 change path if needed

# ===============================
# Draw bounding boxes
# ===============================
def draw_labels(results, img):
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        print("⚠️ No detections")
        return img

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])

        label = f"{results[0].names[cls]} {conf:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        (w, h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )

        cv2.rectangle(
            img,
            (x1, y1 - h - 10),
            (x1 + w + 6, y1),
            (255, 0, 0),
            -1
        )

        cv2.putText(
            img,
            label,
            (x1 + 3, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    return img

# ===============================
# Convert image to base64
# ===============================
def img_to_base64(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buff.getvalue()).decode()

# ===============================
# Flask Route
# ===============================
@app.route("/", methods=["GET", "POST"])
def index():
    uploaded_image = None
    predicted_image = None

    if request.method == "POST":
        file = request.files.get("image")

        if file:
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if img is None:
                return render_template("index.html")

            uploaded_image = img_to_base64(img)

            # YOLOv11 inference
            results = model(
                img,
                conf=0.25,
                iou=0.5
            )

            print("Detections:", len(results[0].boxes))

            output_img = draw_labels(results, img.copy())
            predicted_image = img_to_base64(output_img)

    return render_template(
        "index.html",
        uploaded_image=uploaded_image,
        predicted_image=predicted_image
    )

# ===============================
# Run App
# ===============================
if __name__ == "__main__":
    app.run(debug=True)
