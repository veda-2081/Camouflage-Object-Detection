from ultralytics import YOLO
import cv2
import numpy as np
IMG_SIZE = 640
models = {
    "military": YOLO("models/military_assets_dataset_best.pt"),
    "uav": YOLO("models/hit-uav-best.pt"),
    "mudcadx": YOLO("models/mudcadx_best.pt"),
    "visd": YOLO("models/visd-best.pt"),
    "snow": YOLO("models/best.pt")
}
model_order = ["military", "uav", "mudcadx", "visd", "snow"]
model_conf = {
    "military": 0.50, "uav": 0.45, "mudcadx": 0.55, "visd": 0.50, "snow": 0.55
}
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
def detect_all_models(img):
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    output_img = img_resized.copy()
    img_h, img_w, _ = output_img.shape
    img_area = img_h * img_w
    final_boxes = []
    all_confidences = []
    for model_name in model_order:
        model = models[model_name]
        results = model(img_resized, conf=model_conf.get(model_name, 0.5), iou=0.6, imgsz=IMG_SIZE, verbose=False)[0]  
        if results.boxes is None: continue
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            area = (x2 - x1) * (y2 - y1)
            if area < 2000 or area > 0.65 * img_area: continue
            label_name = model.names.get(cls, f"class {cls}")
            new_box = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": conf, "label": label_name}
            ignore = False
            for existing_box in final_boxes:
                if compute_iou(new_box, existing_box) > 0.5:
                    ignore = True
                    break
            if not ignore:
                final_boxes.append(new_box)
                all_confidences.append(conf)
    for box in final_boxes:
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        score = box["score"]
        label = box["label"]
        color = (0, 255, 0) if score > 0.85 else (255, 165, 0) if score > 0.70 else (255, 255, 0)
        cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(output_img, f"{label} {score:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    avg_confidence = round(sum(all_confidences) / len(all_confidences) * 100, 2) if all_confidences else 0.0
    return output_img, len(final_boxes), avg_confidence