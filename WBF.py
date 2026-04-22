from flask import Flask, render_template, request
from ultralytics import YOLO
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from ensemble_boxes import weighted_boxes_fusion
import torch

# Disable gradients (faster inference)
torch.set_grad_enabled(False)

app = Flask(__name__)

# Load all trained models
models = [
    YOLO("military_assets_dataset_best.pt"),
    YOLO("hit-uav-best.pt"),
    YOLO("mudcadx_best.pt"),
    YOLO("visd-best.pt"),
    YOLO("best.pt")   # snow camouflage model
]


# Convert image to base64 (for displaying in browser)
def img_to_base64(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buff.getvalue()).decode()


# ---------------- Weighted Box Fusion + Filtering ----------------
def run_wbf(models, img):

    h, w, _ = img.shape

    boxes_list = []
    scores_list = []
    labels_list = []

    # ---------------- Run All Models ----------------
    for model in models:

        results = model(
            img,
            conf=0.15,      # LOWERED (important)
            iou=0.5,
            imgsz=960,      # Increase if trained on large size
            verbose=False
        )[0]

        if results.boxes is None or len(results.boxes) == 0:
            continue

        boxes = []
        scores = []
        labels = []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            # Normalize coordinates for WBF
            boxes.append([x1 / w, y1 / h, x2 / w, y2 / h])
            scores.append(float(box.conf[0]))
            labels.append(int(box.cls[0]))

        boxes_list.append(boxes)
        scores_list.append(scores)
        labels_list.append(labels)

    # If nothing detected
    if not boxes_list:
        return img

    # ---------------- Weighted Box Fusion ----------------
    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        iou_thr=0.55,      # Slightly relaxed
        skip_box_thr=0.10  # Allow weak detections
    )

    final_boxes = []

    # ---------------- Light Filtering ----------------
    for box, score, label in zip(boxes, scores, labels):

        if score < 0.20:   # relaxed final threshold
            continue

        x1 = int(box[0] * w)
        y1 = int(box[1] * h)
        x2 = int(box[2] * w)
        y2 = int(box[3] * h)

        # Prevent invalid boxes
        if x2 <= x1 or y2 <= y1:
            continue

        final_boxes.append((x1, y1, x2, y2, score, label))

    object_count = len(final_boxes)

    # ---------------- Draw Boxes ----------------
    for i, (x1, y1, x2, y2, score, label) in enumerate(final_boxes):

        label_name = models[0].names.get(label, f"class {label}")

        if object_count == 1:
            text_lines = [
                f"{label_name} {score:.2f}",
                f"Count: {object_count}"
            ]
        else:
            if i == 0:
                text_lines = [
                    f"{label_name} {score:.2f}",
                    f"Count: {object_count}"
                ]
            else:
                text_lines = [
                    f"{label_name} {score:.2f}"
                ]

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # ---- Text Size Calculation ----
        line_heights = []
        max_width = 0

        for line in text_lines:
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            max_width = max(max_width, tw)
            line_heights.append(th)

        box_width = max_width + 10
        box_height = sum(line_heights) + 8 * len(text_lines) + 5

        # Prevent right overflow
        if x1 + box_width > w:
            x1 = w - box_width - 5

        # Decide label position
        if y1 - box_height > 0:
            y_text = y1 - box_height
        else:
            y_text = y2

        if y_text + box_height > h:
            y_text = h - box_height - 5

        # Background box
        cv2.rectangle(
            img,
            (x1, y_text),
            (x1 + box_width, y_text + box_height),
            (0, 255, 0),
            -1
        )

        # Draw text
        current_y = y_text + 5
        for idx, line in enumerate(text_lines):
            cv2.putText(
                img,
                line,
                (x1 + 5, current_y + line_heights[idx]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
            current_y += line_heights[idx] + 8

    return img


# ---------------- Flask Route ----------------
@app.route("/", methods=["GET", "POST"])
def index():

    results_images = []

    if request.method == "POST":

        files = request.files.getlist("image")

        for file in files:
            if file.filename == "":
                continue

            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if img is None:
                continue

            detected_img = run_wbf(models, img.copy())
            results_images.append(img_to_base64(detected_img))

    return render_template("index1.html", images=results_images)


if __name__ == "__main__":
    app.run(debug=True)