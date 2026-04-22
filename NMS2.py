from flask import Flask, render_template, request
from ultralytics import YOLO
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import torch

torch.set_grad_enabled(False)

app = Flask(__name__)

IMG_SIZE = 640

# Model execution order
model_order = [
    "military",
    "uav",
    "mudcadx",
    "visd",
    "snow"
]

# Load models
models = {
    "military": YOLO("models/military_assets_dataset_best.pt"),
    "uav": YOLO("models/hit-uav-best.pt"),
    "mudcadx": YOLO("models/mudcadx_best.pt"),
    "visd": YOLO("models/visd-best.pt"),
    "snow": YOLO("models/best.pt")
}


# Convert image to base64
def img_to_base64(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    buff = BytesIO()
    pil_img.save(buff, format="PNG")

    return "data:image/png;base64," + base64.b64encode(buff.getvalue()).decode()


# IoU calculation
def compute_iou(boxA, boxB):

    inter_x1 = max(boxA["x1"], boxB["x1"])
    inter_y1 = max(boxA["y1"], boxB["y1"])
    inter_x2 = min(boxA["x2"], boxB["x2"])
    inter_y2 = min(boxA["y2"], boxB["y2"])

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    areaA = (boxA["x2"] - boxA["x1"]) * (boxA["y2"] - boxA["y1"])
    areaB = (boxB["x2"] - boxB["x1"]) * (boxB["y2"] - boxB["y1"])

    union = areaA + areaB - inter_area

    if union == 0:
        return 0

    return inter_area / union


# Detection using all models
def detect_all_models(img):

    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    output_img = img_resized.copy()

    img_h, img_w, _ = output_img.shape
    img_area = img_h * img_w

    final_boxes = []

    model_conf = {
        "military": 0.50,
        "uav": 0.45,
        "mudcadx": 0.55,
        "visd": 0.50,
        "snow": 0.55
    }

    # Run models sequentially
    for model_name in model_order:

        model = models[model_name]

        results = model(
            img_resized,
            conf=model_conf.get(model_name, 0.5),
            iou=0.6,
            imgsz=IMG_SIZE,
            augment=False,
            verbose=False
        )[0]

        if results.boxes is None:
            continue

        for box in results.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            area = (x2 - x1) * (y2 - y1)

            if area < 2000:
                continue

            if area > 0.65 * img_area:
                continue

            label_name = model.names.get(cls, f"class {cls}")

            new_box = {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "score": conf,
                "label": label_name
            }

            ignore = False

            for existing_box in final_boxes:
                if compute_iou(new_box, existing_box) > 0.5:
                    ignore = True
                    break

            if not ignore:
                final_boxes.append(new_box)

    object_count = len(final_boxes)

    # Draw boxes
    for box in final_boxes:

        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        score = box["score"]
        label = box["label"]

        if score > 0.85:
            color = (0, 255, 0)
        elif score > 0.70:
            color = (255, 165, 0)
        else:
            color = (255, 255, 0)

        cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)

        text = f"{label} | {score:.2f}"

        (tw, th), _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2
        )

        box_width = tw + 10
        box_height = th + 10

        if x1 + box_width > img_w:
            x1 = img_w - box_width - 5

        y_text_top = y1 - box_height if y1 - box_height > 0 else y2

        if y_text_top + box_height > img_h:
            y_text_top = img_h - box_height - 5

        cv2.rectangle(
            output_img,
            (x1, y_text_top),
            (x1 + box_width, y_text_top + box_height),
            color,
            -1
        )

        cv2.putText(
            output_img,
            text,
            (x1 + 5, y_text_top + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

    return output_img, object_count


# Flask route
@app.route("/", methods=["GET", "POST"])
def index():

    results = []

    if request.method == "POST":

        files = request.files.getlist("image")

        for file in files:

            if file.filename == "":
                continue

            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if img is None:
                continue

            detected_img, count = detect_all_models(img)

            results.append({
                "image": img_to_base64(detected_img),
                "count": count
            })

    return render_template("index1.html", results=results)


if __name__ == "__main__":
    app.run(debug=True)