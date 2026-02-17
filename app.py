from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
from ultralytics import YOLO
import io
import random
import base64

app = Flask(__name__)

# =============================
# LOAD MODEL (sekali saat startup)
# =============================

# ML1: Model Klasifikasi Tray Omprengan
TRAY_MODEL_PATH = "model/yolo-11s-seg-omprengan-epoch50.pt"
model_tray = YOLO(TRAY_MODEL_PATH)

# ML2: Model Segmentasi Menu Makanan
FOOD_MODEL_PATH = "model/yolo-11s-seg-menu10-epoch50.pt"
model_food = YOLO(FOOD_MODEL_PATH)

# Backward compatibility (untuk endpoint lama)
model = model_food

# Class colors untuk menu makanan (konsisten)
random.seed(42)
CLASS_COLORS = {}
for i in range(len(model_food.names)):
    CLASS_COLORS[i] = tuple(random.randint(80, 255) for _ in range(3))

CLASS_NAMES = model_food.names

# Class colors untuk tray
random.seed(99)
TRAY_COLORS = {}
for i in range(len(model_tray.names)):
    TRAY_COLORS[i] = tuple(random.randint(80, 255) for _ in range(3))

TRAY_NAMES = model_tray.names


# =============================
# HELPER FUNCTIONS
# =============================
def crop_tray_with_padding(image, bbox, padding_ratio=0.15):
    """
    Crop area tray dari gambar asli dengan padding tambahan
    agar makanan yang sedikit keluar tray tetap tertangkap.
    """
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]

    tray_w = x2 - x1
    tray_h = y2 - y1

    pad_x = int(tray_w * padding_ratio)
    pad_y = int(tray_h * padding_ratio)

    x1_pad = max(0, x1 - pad_x)
    y1_pad = max(0, y1 - pad_y)
    x2_pad = min(w, x2 + pad_x)
    y2_pad = min(h, y2 + pad_y)

    return image[y1_pad:y2_pad, x1_pad:x2_pad], (x1_pad, y1_pad)


def is_food_center_in_tray(food_bbox_global, tray_bbox):
    """
    Cek apakah titik tengah makanan berada di dalam bbox tray asli.
    Untuk filter makanan offside agar masuk ke tray yang benar.
    """
    fx1, fy1, fx2, fy2 = food_bbox_global
    tx1, ty1, tx2, ty2 = tray_bbox

    center_x = (fx1 + fx2) / 2
    center_y = (fy1 + fy2) / 2

    return (tx1 <= center_x <= tx2) and (ty1 <= center_y <= ty2)


def encode_image_to_base64(image_bgr):
    """
    Encode gambar (BGR numpy array) ke base64 string PNG.
    Untuk dikirim dalam JSON response.
    """
    _, buffer = cv2.imencode('.png', image_bgr)
    return base64.b64encode(buffer).decode('utf-8')


def crop_food_from_image(image, bbox, padding=10):
    """
    Crop area makanan dari gambar dengan sedikit padding.
    """
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    return image[y1:y2, x1:x2]


def compute_iou(box1, box2):
    """
    Hitung IoU (Intersection over Union) antara 2 bounding box.
    box format: (x1, y1, x2, y2)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union == 0:
        return 0
    return intersection / union


def is_box_inside(inner_box, outer_box, threshold=0.7):
    """
    Cek apakah inner_box berada di dalam outer_box.
    Jika area overlap >= threshold * area inner_box, dianggap nested.
    """
    x1 = max(inner_box[0], outer_box[0])
    y1 = max(inner_box[1], outer_box[1])
    x2 = min(inner_box[2], outer_box[2])
    y2 = min(inner_box[3], outer_box[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    inner_area = (inner_box[2] - inner_box[0]) * (inner_box[3] - inner_box[1])

    if inner_area == 0:
        return False
    return (intersection / inner_area) >= threshold


def filter_nested_trays(tray_data_list, mode="largest"):
    """
    Filter tray yang overlap/nested.

    mode:
        'largest' → jika ada tray di dalam tray lain, ambil yang TERBESAR (tray utama)
        'smallest' → jika ada tray di dalam tray lain, ambil yang TERKECIL (sekat)
    """
    if len(tray_data_list) <= 1:
        return tray_data_list

    # Hitung area tiap tray
    for t in tray_data_list:
        b = t["bbox_tuple"]
        t["area"] = (b[2] - b[0]) * (b[3] - b[1])

    # Sort berdasarkan area (terbesar dulu)
    sorted_trays = sorted(tray_data_list, key=lambda x: x["area"], reverse=True)

    if mode == "largest":
        # Hapus tray kecil yang berada di dalam tray besar
        kept = []
        for tray in sorted_trays:
            is_nested = False
            for kept_tray in kept:
                if is_box_inside(tray["bbox_tuple"], kept_tray["bbox_tuple"], threshold=0.7):
                    is_nested = True
                    break
            if not is_nested:
                kept.append(tray)
        return kept

    elif mode == "smallest":
        # Hapus tray besar yang mengandung tray kecil
        kept = []
        reversed_trays = list(reversed(sorted_trays))  # kecil dulu
        for tray in reversed_trays:
            is_parent = False
            for kept_tray in kept:
                if is_box_inside(kept_tray["bbox_tuple"], tray["bbox_tuple"], threshold=0.7):
                    is_parent = True
                    break
            if not is_parent:
                kept.append(tray)
        return kept

    return tray_data_list


@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "message": "YOLOv11 Menu Segmentation API (Flask)",
        "endpoints": {
            "/predict": "POST - Upload gambar, deteksi menu (JSON + polygon)",
            "/predict/simple": "POST - Upload gambar, daftar makanan terdeteksi (tanpa polygon)",
            "/predict/image": "POST - Upload gambar, gambar hasil annotasi (PNG)",
            "/predict/pipeline": "POST - Upload gambar, deteksi tray → segmentasi makanan per tray (JSON)",
            "/predict/pipeline/image": "POST - Upload gambar, gambar annotasi pipeline tray + makanan (PNG)",
            "/classes": "GET - Daftar class menu makanan",
            "/classes/tray": "GET - Daftar class tray omprengan",
        }
    })


@app.route("/classes", methods=["GET"])
def get_classes():
    """Daftar semua class menu makanan yang bisa dikenali model"""
    return jsonify({"num_classes": len(CLASS_NAMES), "classes": CLASS_NAMES})


@app.route("/classes/tray", methods=["GET"])
def get_tray_classes():
    """Daftar semua class tray omprengan yang bisa dikenali model"""
    return jsonify({"num_classes": len(TRAY_NAMES), "classes": TRAY_NAMES})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Upload gambar -> dapatkan hasil deteksi dalam bentuk JSON.
    Termasuk bounding box, class name, confidence, dan segmentation polygon.

    Query params:
        conf_threshold (float): default 0.6
        iou_threshold (float): default 0.45
    """
    conf_threshold = request.args.get("conf_threshold", 0.6, type=float)
    iou_threshold = request.args.get("iou_threshold", 0.45, type=float)

    # Cek file upload
    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file yang diupload. Gunakan key 'file'"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nama file kosong"}), 400

    # Baca image dari upload
    file_bytes = file.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Gambar tidak valid"}), 400

    # Predict
    results = model.predict(
        source=image,
        conf=0.25,
        iou=iou_threshold,
        save=False,
        verbose=False
    )

    result = results[0]
    boxes = result.boxes
    masks = result.masks

    # Cek confidence
    detections = []
    max_conf = 0.0
    recognized = False

    if boxes is not None and len(boxes) > 0 and boxes.conf is not None:
        confidences = boxes.conf.cpu().numpy()
        if len(confidences) > 0:
            max_conf = float(confidences.max())
            if max_conf >= conf_threshold:
                recognized = True

        for i, (box, conf, cls) in enumerate(zip(
            boxes.xyxy.cpu().numpy(),
            boxes.conf.cpu().numpy(),
            boxes.cls.cpu().numpy()
        )):
            x1, y1, x2, y2 = map(int, box)
            cls_id = int(cls)
            detection = {
                "id": i,
                "class_id": cls_id,
                "class_name": CLASS_NAMES[cls_id],
                "confidence": round(float(conf), 4),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            }

            # Tambahkan mask polygon jika ada
            if masks is not None and i < len(masks.data):
                mask_data = masks.data[i].cpu().numpy()
                mask_resized = cv2.resize(mask_data, (image.shape[1], image.shape[0]))
                contours, _ = cv2.findContours(
                    (mask_resized > 0.5).astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                polygons = []
                for contour in contours:
                    polygon = contour.squeeze().tolist()
                    if isinstance(polygon, list) and len(polygon) > 0:
                        if isinstance(polygon[0], list):
                            polygons.append(polygon)
                detection["segmentation_polygon"] = polygons

            detections.append(detection)

    return jsonify({
        "recognized": recognized,
        "max_confidence": round(max_conf, 4),
        "conf_threshold": conf_threshold,
        "iou_threshold": iou_threshold,
        "num_detections": len(detections),
        "detections": detections,
        "image_size": {"width": image.shape[1], "height": image.shape[0]},
    })


@app.route("/predict/simple", methods=["POST"])
def predict_simple():
    """
    Upload gambar -> dapatkan daftar makanan yang terdeteksi (tanpa polygon).
    Response ringkas: hanya nama makanan, confidence, dan bounding box.

    Query params:
        conf_threshold (float): default 0.6
        iou_threshold (float): default 0.45
    """
    conf_threshold = request.args.get("conf_threshold", 0.6, type=float)
    iou_threshold = request.args.get("iou_threshold", 0.45, type=float)

    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file yang diupload. Gunakan key 'file'"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nama file kosong"}), 400

    file_bytes = file.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Gambar tidak valid"}), 400

    results = model.predict(
        source=image,
        conf=0.25,
        iou=iou_threshold,
        save=False,
        verbose=False
    )

    result = results[0]
    boxes = result.boxes

    detections = []
    max_conf = 0.0
    recognized = False

    if boxes is not None and len(boxes) > 0 and boxes.conf is not None:
        confidences = boxes.conf.cpu().numpy()
        if len(confidences) > 0:
            max_conf = float(confidences.max())
            if max_conf >= conf_threshold:
                recognized = True

        for i, (box, conf, cls) in enumerate(zip(
            boxes.xyxy.cpu().numpy(),
            boxes.conf.cpu().numpy(),
            boxes.cls.cpu().numpy()
        )):
            x1, y1, x2, y2 = map(int, box)
            cls_id = int(cls)
            detections.append({
                "id": i,
                "class_name": CLASS_NAMES[cls_id],
                "confidence": round(float(conf), 4),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            })

    # Daftar makanan unik yang terdeteksi
    food_list = list(set(d["class_name"] for d in detections))

    return jsonify({
        "recognized": recognized,
        "max_confidence": round(max_conf, 4),
        "conf_threshold": conf_threshold,
        "num_detections": len(detections),
        "food_detected": food_list,
        "detections": detections,
        "image_size": {"width": image.shape[1], "height": image.shape[0]},
    })


@app.route("/predict/image", methods=["POST"])
def predict_image():
    """
    Upload gambar -> dapatkan gambar hasil annotasi lengkap (PNG).
    Termasuk bounding box, label, dan segmentation mask overlay.

    Query params:
        conf_threshold (float): default 0.6
        iou_threshold (float): default 0.45
        mask_opacity (float): default 0.4
    """
    conf_threshold = request.args.get("conf_threshold", 0.6, type=float)
    iou_threshold = request.args.get("iou_threshold", 0.45, type=float)
    mask_opacity = request.args.get("mask_opacity", 0.4, type=float)

    # Cek file upload
    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file yang diupload. Gunakan key 'file'"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nama file kosong"}), 400

    # Baca image
    file_bytes = file.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Gambar tidak valid"}), 400

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # =============================
    # IMAGE ENHANCEMENT
    # =============================
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, kernel)

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # =============================
    # PREDICT
    # =============================
    results = model.predict(
        source=cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        conf=0.25,
        iou=iou_threshold,
        save=False,
        verbose=False
    )

    result = results[0]
    boxes = result.boxes
    masks = result.masks

    # =============================
    # CONFIDENCE CHECK
    # =============================
    unknown_flag = True
    max_conf = 0

    if boxes is not None and len(boxes) > 0 and boxes.conf is not None:
        confidences = boxes.conf.cpu().numpy()
        if len(confidences) > 0:
            max_conf = float(confidences.max())
            if max_conf >= conf_threshold:
                unknown_flag = False

    # =============================
    # UNKNOWN IMAGE
    # =============================
    if unknown_flag:
        cv2.putText(
            image, "GAMBAR TIDAK DIKENALI", (30, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 5
        )

    # =============================
    # DRAW DETECTIONS
    # =============================
    else:
        # Draw segmentation masks
        if masks is not None:
            for mask_data, cls_m in zip(masks.data.cpu().numpy(), boxes.cls.cpu().numpy()):
                cls_m = int(cls_m)
                color = CLASS_COLORS[cls_m]
                mask_resized = cv2.resize(mask_data, (image.shape[1], image.shape[0]))
                mask_bool = mask_resized > 0.5
                overlay = image.copy()
                overlay[mask_bool] = color
                image = cv2.addWeighted(overlay, mask_opacity, image, 1 - mask_opacity, 0)

        # Draw bounding boxes + labels
        for box, conf, cls in zip(
            boxes.xyxy.cpu().numpy(),
            boxes.conf.cpu().numpy(),
            boxes.cls.cpu().numpy()
        ):
            x1, y1, x2, y2 = map(int, box)
            cls = int(cls)
            color = CLASS_COLORS[cls]

            # Bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 10)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 6)

            label = f"{CLASS_NAMES[cls]} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)

            # Label background
            cv2.rectangle(image, (x1, y1 - th - 25), (x1 + tw + 20, y1), color, -1)

            # Label text
            cv2.putText(image, label, (x1 + 8, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5)
            cv2.putText(image, label, (x1 + 8, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)

    # =============================
    # RETURN IMAGE
    # =============================
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', image_bgr)
    io_buf = io.BytesIO(buffer.tobytes())
    io_buf.seek(0)

    return send_file(io_buf, mimetype="image/png")


# =============================
# PIPELINE: TRAY → MENU (JSON)
# =============================
@app.route("/predict/pipeline", methods=["POST"])
def predict_pipeline():
    """
    Pipeline lengkap: Upload gambar → ML1 deteksi tray → crop per tray → ML2 segmentasi makanan.
    Setiap makanan terasosiasi ke tray yang benar.
    Response termasuk gambar crop per tray dan per makanan (base64 PNG).

    Query params:
        conf_threshold (float): confidence threshold untuk menu (default 0.6)
        iou_threshold (float): IOU threshold (default 0.45)
        tray_conf_threshold (float): confidence threshold untuk tray (default 0.5)
        padding_ratio (float): padding saat crop tray (default 0.15)
        include_images (bool): sertakan gambar base64 (default true, set 0 untuk tanpa gambar)
        tray_mode (str): 'largest' ambil tray utama, 'smallest' (default) ambil sekat, 'all' tanpa filter
    """
    conf_threshold = request.args.get("conf_threshold", 0.6, type=float)
    iou_threshold = request.args.get("iou_threshold", 0.45, type=float)
    tray_conf_threshold = request.args.get("tray_conf_threshold", 0.5, type=float)
    padding_ratio = request.args.get("padding_ratio", 0.15, type=float)
    include_images = request.args.get("include_images", 1, type=int)
    tray_mode = request.args.get("tray_mode", "smallest", type=str)

    # Cek file upload
    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file yang diupload. Gunakan key 'file'"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nama file kosong"}), 400

    file_bytes = file.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Gambar tidak valid"}), 400

    # =============================
    # TAHAP 1: ML1 - Deteksi Tray
    # =============================
    tray_results = model_tray.predict(
        source=image,
        conf=tray_conf_threshold,
        iou=iou_threshold,
        save=False,
        verbose=False
    )

    tray_result = tray_results[0]
    tray_boxes = tray_result.boxes

    if tray_boxes is None or len(tray_boxes) == 0:
        return jsonify({
            "recognized": False,
            "message": "Tidak ada tray omprengan yang terdeteksi",
            "num_trays": 0,
            "trays": [],
            "all_foods": [],
            "image_size": {"width": image.shape[1], "height": image.shape[0]},
        })

    # =============================
    # TAHAP 2: Filter tray yang overlap/nested
    # =============================
    raw_trays = []
    for t_idx, (tbox, tconf, tcls) in enumerate(zip(
        tray_boxes.xyxy.cpu().numpy(),
        tray_boxes.conf.cpu().numpy(),
        tray_boxes.cls.cpu().numpy()
    )):
        tx1, ty1, tx2, ty2 = map(int, tbox)
        raw_trays.append({
            "bbox_tuple": (tx1, ty1, tx2, ty2),
            "conf": float(tconf),
            "cls_id": int(tcls),
        })

    # =============================
    # TAHAP 2 & 3: Detect food per tray
    # =============================
    trays_output = []
    all_foods_flat = []

    if tray_mode == "smallest":
        # =============================
        # HYBRID MODE: Predict pada tray terbesar, assign ke sekat terkecil
        # Alasan: crop sekat terlalu kecil → ML2 gagal deteksi makanan
        # Solusi: crop parent (besar) → predict akurat → assign ke sekat via center point
        # =============================

        # Hitung area tiap tray
        for t in raw_trays:
            b = t["bbox_tuple"]
            t["area"] = (b[2] - b[0]) * (b[3] - b[1])

        sorted_trays = sorted(raw_trays, key=lambda x: x["area"], reverse=True)

        # Group nested trays: parent (terbesar) → children (sekat di dalamnya)
        assigned_indices = set()
        groups = []
        for i, tray in enumerate(sorted_trays):
            if i in assigned_indices:
                continue
            group = {"parent": tray, "children": []}
            assigned_indices.add(i)
            for j, other in enumerate(sorted_trays):
                if j in assigned_indices:
                    continue
                if is_box_inside(other["bbox_tuple"], tray["bbox_tuple"], threshold=0.7):
                    group["children"].append(other)
                    assigned_indices.add(j)
            groups.append(group)

        global_tray_number = 0

        for group in groups:
            parent = group["parent"]
            children = group["children"]
            ptx1, pty1, ptx2, pty2 = parent["bbox_tuple"]

            # Crop parent tray (besar) dengan padding → ML2 lebih akurat
            crop, (offset_x, offset_y) = crop_tray_with_padding(
                image, parent["bbox_tuple"], padding_ratio
            )

            # ML2: Segmentasi makanan di dalam crop parent
            food_results = model_food.predict(
                source=crop,
                conf=0.25,
                iou=iou_threshold,
                save=False,
                verbose=False
            )

            food_result = food_results[0]
            food_boxes = food_result.boxes

            # Kumpulkan semua makanan terdeteksi (koordinat global)
            detected_foods = []
            if food_boxes is not None and len(food_boxes) > 0:
                for f_idx, (fbox, fconf, fcls) in enumerate(zip(
                    food_boxes.xyxy.cpu().numpy(),
                    food_boxes.conf.cpu().numpy(),
                    food_boxes.cls.cpu().numpy()
                )):
                    fx1, fy1, fx2, fy2 = map(int, fbox)

                    # Konversi ke koordinat global
                    fx1_g = fx1 + offset_x
                    fy1_g = fy1 + offset_y
                    fx2_g = fx2 + offset_x
                    fy2_g = fy2 + offset_y

                    # Filter: center point harus dalam parent tray
                    food_global = (fx1_g, fy1_g, fx2_g, fy2_g)
                    if not is_food_center_in_tray(food_global, parent["bbox_tuple"]):
                        continue

                    fcls_id = int(fcls)
                    if float(fconf) < conf_threshold:
                        continue

                    # Crop gambar makanan
                    food_image_b64 = None
                    if include_images:
                        food_crop = crop_food_from_image(
                            image, (fx1_g, fy1_g, fx2_g, fy2_g)
                        )
                        food_image_b64 = encode_image_to_base64(food_crop)

                    detected_foods.append({
                        "id": f_idx,
                        "class_name": CLASS_NAMES[fcls_id],
                        "confidence": round(float(fconf), 4),
                        "bbox": {"x1": fx1_g, "y1": fy1_g, "x2": fx2_g, "y2": fy2_g},
                        "image_base64": food_image_b64,
                    })

            if children:
                # Assign setiap makanan ke sekat terkecil yang mengandung center point-nya
                for food in detected_foods:
                    cx = (food["bbox"]["x1"] + food["bbox"]["x2"]) / 2
                    cy = (food["bbox"]["y1"] + food["bbox"]["y2"]) / 2

                    best_child = None
                    best_area = float("inf")
                    for child in children:
                        sx1, sy1, sx2, sy2 = child["bbox_tuple"]
                        if sx1 <= cx <= sx2 and sy1 <= cy <= sy2:
                            if child["area"] < best_area:
                                best_area = child["area"]
                                best_child = child

                    food["_assigned_id"] = id(best_child) if best_child else None

                # Output setiap sekat sebagai tray entry
                for child in children:
                    global_tray_number += 1
                    child_obj_id = id(child)
                    tcls_id = child["cls_id"]
                    tray_class_name = TRAY_NAMES[tcls_id]
                    tx1, ty1, tx2, ty2 = child["bbox_tuple"]

                    # Filter foods yang di-assign ke sekat ini
                    child_foods = [f for f in detected_foods if f.get("_assigned_id") == child_obj_id]

                    foods = []
                    for food in child_foods:
                        food_item = {
                            "id": food["id"],
                            "class_name": food["class_name"],
                            "confidence": food["confidence"],
                            "tray_id": global_tray_number,
                            "tray_class": tray_class_name,
                            "bbox": food["bbox"],
                        }
                        if include_images:
                            food_item["image_base64"] = food["image_base64"]
                        foods.append(food_item)

                        all_foods_flat.append({
                            "class_name": food["class_name"],
                            "confidence": food["confidence"],
                            "tray_id": global_tray_number,
                            "tray_class": tray_class_name,
                        })

                    # Encode gambar sekat
                    tray_image_b64 = None
                    if include_images:
                        tray_crop_clean = image[ty1:ty2, tx1:tx2]
                        tray_image_b64 = encode_image_to_base64(tray_crop_clean)

                    tray_data = {
                        "tray_id": global_tray_number,
                        "tray_class": tray_class_name,
                        "tray_confidence": round(float(child["conf"]), 4),
                        "tray_bbox": {"x1": tx1, "y1": ty1, "x2": tx2, "y2": ty2},
                        "num_foods": len(foods),
                        "food_detected": list(set(f["class_name"] for f in foods)),
                        "foods": foods,
                    }
                    if include_images:
                        tray_data["tray_image_base64"] = tray_image_b64

                    trays_output.append(tray_data)

                # Makanan yang tidak masuk sekat manapun → assign ke parent tray
                unassigned = [f for f in detected_foods if f.get("_assigned_id") is None]
                if unassigned:
                    global_tray_number += 1
                    ptcls_id = parent["cls_id"]
                    parent_class_name = TRAY_NAMES[ptcls_id]

                    foods = []
                    for food in unassigned:
                        food_item = {
                            "id": food["id"],
                            "class_name": food["class_name"],
                            "confidence": food["confidence"],
                            "tray_id": global_tray_number,
                            "tray_class": parent_class_name,
                            "bbox": food["bbox"],
                        }
                        if include_images:
                            food_item["image_base64"] = food["image_base64"]
                        foods.append(food_item)

                        all_foods_flat.append({
                            "class_name": food["class_name"],
                            "confidence": food["confidence"],
                            "tray_id": global_tray_number,
                            "tray_class": parent_class_name,
                        })

                    tray_image_b64 = None
                    if include_images:
                        tray_crop_clean = image[pty1:pty2, ptx1:ptx2]
                        tray_image_b64 = encode_image_to_base64(tray_crop_clean)

                    tray_data = {
                        "tray_id": global_tray_number,
                        "tray_class": parent_class_name,
                        "tray_confidence": round(float(parent["conf"]), 4),
                        "tray_bbox": {"x1": ptx1, "y1": pty1, "x2": ptx2, "y2": pty2},
                        "num_foods": len(foods),
                        "food_detected": list(set(f["class_name"] for f in foods)),
                        "foods": foods,
                    }
                    if include_images:
                        tray_data["tray_image_base64"] = tray_image_b64

                    trays_output.append(tray_data)

            else:
                # Standalone tray (tidak punya sekat) → output langsung
                global_tray_number += 1
                tcls_id = parent["cls_id"]
                tray_class_name = TRAY_NAMES[tcls_id]
                tx1, ty1, tx2, ty2 = parent["bbox_tuple"]

                foods = []
                for food in detected_foods:
                    food_item = {
                        "id": food["id"],
                        "class_name": food["class_name"],
                        "confidence": food["confidence"],
                        "tray_id": global_tray_number,
                        "tray_class": tray_class_name,
                        "bbox": food["bbox"],
                    }
                    if include_images:
                        food_item["image_base64"] = food["image_base64"]
                    foods.append(food_item)

                    all_foods_flat.append({
                        "class_name": food["class_name"],
                        "confidence": food["confidence"],
                        "tray_id": global_tray_number,
                        "tray_class": tray_class_name,
                    })

                tray_image_b64 = None
                if include_images:
                    tray_crop_clean = image[ty1:ty2, tx1:tx2]
                    tray_image_b64 = encode_image_to_base64(tray_crop_clean)

                tray_data = {
                    "tray_id": global_tray_number,
                    "tray_class": tray_class_name,
                    "tray_confidence": round(float(parent["conf"]), 4),
                    "tray_bbox": {"x1": tx1, "y1": ty1, "x2": tx2, "y2": ty2},
                    "num_foods": len(foods),
                    "food_detected": list(set(f["class_name"] for f in foods)),
                    "foods": foods,
                }
                if include_images:
                    tray_data["tray_image_base64"] = tray_image_b64

                trays_output.append(tray_data)

    else:
        # =============================
        # MODE LARGEST / ALL: Predict per tray (perilaku asli)
        # =============================
        if tray_mode == "largest":
            filtered_trays = filter_nested_trays(raw_trays, mode="largest")
        else:
            filtered_trays = raw_trays

        for t_idx, tray_info in enumerate(filtered_trays):
            tx1, ty1, tx2, ty2 = tray_info["bbox_tuple"]
            tconf = tray_info["conf"]
            tcls_id = tray_info["cls_id"]
            tray_bbox_original = (tx1, ty1, tx2, ty2)
            tray_number = t_idx + 1
            tray_class_name = TRAY_NAMES[tcls_id]

            # Crop tray dengan padding
            crop, (offset_x, offset_y) = crop_tray_with_padding(
                image, tray_bbox_original, padding_ratio
            )

            # Encode gambar crop tray ke base64
            tray_image_b64 = None
            if include_images:
                tray_crop_clean = image[ty1:ty2, tx1:tx2]
                tray_image_b64 = encode_image_to_base64(tray_crop_clean)

            # ML2: Segmentasi makanan di dalam crop tray
            food_results = model_food.predict(
                source=crop,
                conf=0.25,
                iou=iou_threshold,
                save=False,
                verbose=False
            )

            food_result = food_results[0]
            food_boxes = food_result.boxes

            foods = []
            if food_boxes is not None and len(food_boxes) > 0:
                for f_idx, (fbox, fconf, fcls) in enumerate(zip(
                    food_boxes.xyxy.cpu().numpy(),
                    food_boxes.conf.cpu().numpy(),
                    food_boxes.cls.cpu().numpy()
                )):
                    fx1, fy1, fx2, fy2 = map(int, fbox)

                    fx1_global = fx1 + offset_x
                    fy1_global = fy1 + offset_y
                    fx2_global = fx2 + offset_x
                    fy2_global = fy2 + offset_y

                    food_global = (fx1_global, fy1_global, fx2_global, fy2_global)
                    if not is_food_center_in_tray(food_global, tray_bbox_original):
                        continue

                    fcls_id = int(fcls)
                    if float(fconf) < conf_threshold:
                        continue

                    food_image_b64 = None
                    if include_images:
                        food_crop = crop_food_from_image(
                            image, (fx1_global, fy1_global, fx2_global, fy2_global)
                        )
                        food_image_b64 = encode_image_to_base64(food_crop)

                    food_item = {
                        "id": f_idx,
                        "class_name": CLASS_NAMES[fcls_id],
                        "confidence": round(float(fconf), 4),
                        "tray_id": tray_number,
                        "tray_class": tray_class_name,
                        "bbox": {
                            "x1": fx1_global, "y1": fy1_global,
                            "x2": fx2_global, "y2": fy2_global
                        },
                    }

                    if include_images:
                        food_item["image_base64"] = food_image_b64

                    foods.append(food_item)

                    all_foods_flat.append({
                        "class_name": CLASS_NAMES[fcls_id],
                        "confidence": round(float(fconf), 4),
                        "tray_id": tray_number,
                        "tray_class": tray_class_name,
                    })

            food_list = list(set(f["class_name"] for f in foods))

            tray_data = {
                "tray_id": tray_number,
                "tray_class": tray_class_name,
                "tray_confidence": round(float(tconf), 4),
                "tray_bbox": {"x1": tx1, "y1": ty1, "x2": tx2, "y2": ty2},
                "num_foods": len(foods),
                "food_detected": food_list,
                "foods": foods,
            }

            if include_images:
                tray_data["tray_image_base64"] = tray_image_b64

            trays_output.append(tray_data)

    # Ringkasan semua makanan per tray (flat view)
    summary_by_tray = {}
    for f in all_foods_flat:
        key = f"Tray {f['tray_id']} ({f['tray_class']})"
        if key not in summary_by_tray:
            summary_by_tray[key] = []
        if f["class_name"] not in summary_by_tray[key]:
            summary_by_tray[key].append(f["class_name"])

    return jsonify({
        "recognized": len(trays_output) > 0,
        "num_trays": len(trays_output),
        "total_foods_detected": len(all_foods_flat),
        "conf_threshold": conf_threshold,
        "tray_conf_threshold": tray_conf_threshold,
        "tray_mode": tray_mode,
        "summary": summary_by_tray,
        "all_foods": all_foods_flat,
        "trays": trays_output,
        "image_size": {"width": image.shape[1], "height": image.shape[0]},
    })


# =============================
# PIPELINE: TRAY → MENU (IMAGE)
# =============================
@app.route("/predict/pipeline/image", methods=["POST"])
def predict_pipeline_image():
    """
    Pipeline lengkap dengan output gambar annotasi.
    Menampilkan bbox tray, bbox makanan, label, dan segmentation mask.

    Query params:
        conf_threshold (float): confidence threshold untuk menu (default 0.6)
        iou_threshold (float): IOU threshold (default 0.45)
        tray_conf_threshold (float): confidence threshold untuk tray (default 0.5)
        padding_ratio (float): padding saat crop tray (default 0.15)
        mask_opacity (float): opacity segmentation mask (default 0.4)
        tray_mode (str): 'largest' ambil tray utama, 'smallest' (default) ambil sekat, 'all' tanpa filter
    """
    conf_threshold = request.args.get("conf_threshold", 0.6, type=float)
    iou_threshold = request.args.get("iou_threshold", 0.45, type=float)
    tray_conf_threshold = request.args.get("tray_conf_threshold", 0.5, type=float)
    padding_ratio = request.args.get("padding_ratio", 0.15, type=float)
    mask_opacity = request.args.get("mask_opacity", 0.4, type=float)
    tray_mode = request.args.get("tray_mode", "smallest", type=str)

    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file yang diupload. Gunakan key 'file'"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nama file kosong"}), 400

    file_bytes = file.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Gambar tidak valid"}), 400

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # =============================
    # TAHAP 1: ML1 - Deteksi Tray
    # =============================
    tray_results = model_tray.predict(
        source=cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        conf=tray_conf_threshold,
        iou=iou_threshold,
        save=False,
        verbose=False
    )

    tray_result = tray_results[0]
    tray_boxes = tray_result.boxes

    if tray_boxes is None or len(tray_boxes) == 0:
        cv2.putText(
            image, "TRAY TIDAK TERDETEKSI", (30, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 5
        )
    else:
        # =============================
        # TAHAP 2: Filter tray overlap/nested
        # =============================
        raw_trays = []
        for t_idx, (tbox, tconf, tcls) in enumerate(zip(
            tray_boxes.xyxy.cpu().numpy(),
            tray_boxes.conf.cpu().numpy(),
            tray_boxes.cls.cpu().numpy()
        )):
            tx1, ty1, tx2, ty2 = map(int, tbox)
            raw_trays.append({
                "bbox_tuple": (tx1, ty1, tx2, ty2),
                "conf": float(tconf),
                "cls_id": int(tcls),
            })

        # Hitung area tiap tray
        for t in raw_trays:
            b = t["bbox_tuple"]
            t["area"] = (b[2] - b[0]) * (b[3] - b[1])

        if tray_mode == "smallest":
            # =============================
            # HYBRID MODE: Draw semua tray, predict pada parent crop
            # =============================
            sorted_trays = sorted(raw_trays, key=lambda x: x["area"], reverse=True)

            # Group nested trays
            assigned_indices = set()
            groups = []
            for i, tray in enumerate(sorted_trays):
                if i in assigned_indices:
                    continue
                group = {"parent": tray, "children": []}
                assigned_indices.add(i)
                for j, other in enumerate(sorted_trays):
                    if j in assigned_indices:
                        continue
                    if is_box_inside(other["bbox_tuple"], tray["bbox_tuple"], threshold=0.7):
                        group["children"].append(other)
                        assigned_indices.add(j)
                groups.append(group)

            tray_counter = 0

            for group in groups:
                parent = group["parent"]
                children = group["children"]

                # Draw SEMUA tray boxes (parent + children)
                all_trays_in_group = [parent] + children
                for tray_info in all_trays_in_group:
                    tray_counter += 1
                    tx1, ty1, tx2, ty2 = tray_info["bbox_tuple"]
                    tconf = tray_info["conf"]
                    tcls_id = tray_info["cls_id"]
                    tray_color = TRAY_COLORS[tcls_id]

                    cv2.rectangle(image, (tx1, ty1), (tx2, ty2), (0, 0, 0), 12)
                    cv2.rectangle(image, (tx1, ty1), (tx2, ty2), tray_color, 6)

                    tray_label = f"TRAY {tray_counter}: {TRAY_NAMES[tcls_id]} {tconf:.2f}"
                    (tw_l, th_l), _ = cv2.getTextSize(tray_label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                    cv2.rectangle(image, (tx1, ty1 - th_l - 30), (tx1 + tw_l + 20, ty1), tray_color, -1)
                    cv2.putText(image, tray_label, (tx1 + 8, ty1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5)
                    cv2.putText(image, tray_label, (tx1 + 8, ty1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

                # Crop parent tray (besar) → ML2 lebih akurat
                image_bgr_for_crop = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                crop, (offset_x, offset_y) = crop_tray_with_padding(
                    image_bgr_for_crop, parent["bbox_tuple"], padding_ratio
                )

                food_results = model_food.predict(
                    source=crop,
                    conf=0.25,
                    iou=iou_threshold,
                    save=False,
                    verbose=False
                )

                food_result = food_results[0]
                food_boxes = food_result.boxes
                food_masks = food_result.masks

                if food_boxes is not None and len(food_boxes) > 0:
                    # Draw segmentation masks
                    if food_masks is not None:
                        for f_idx, (mask_data, fcls_m) in enumerate(zip(
                            food_masks.data.cpu().numpy(),
                            food_boxes.cls.cpu().numpy()
                        )):
                            fbox = food_boxes.xyxy.cpu().numpy()[f_idx]
                            fx1, fy1, fx2, fy2 = map(int, fbox)
                            fx1_g = fx1 + offset_x
                            fy1_g = fy1 + offset_y
                            fx2_g = fx2 + offset_x
                            fy2_g = fy2 + offset_y

                            if not is_food_center_in_tray(
                                (fx1_g, fy1_g, fx2_g, fy2_g), parent["bbox_tuple"]
                            ):
                                continue

                            fcls_id = int(fcls_m)
                            color = CLASS_COLORS[fcls_id]

                            mask_resized = cv2.resize(mask_data, (crop.shape[1], crop.shape[0]))
                            mask_bool = mask_resized > 0.5

                            full_mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
                            crop_h, crop_w = crop.shape[:2]
                            end_y = min(offset_y + crop_h, image.shape[0])
                            end_x = min(offset_x + crop_w, image.shape[1])
                            mask_h = end_y - offset_y
                            mask_w = end_x - offset_x
                            full_mask[offset_y:end_y, offset_x:end_x] = mask_bool[:mask_h, :mask_w]

                            overlay = image.copy()
                            overlay[full_mask] = color
                            image = cv2.addWeighted(overlay, mask_opacity, image, 1 - mask_opacity, 0)

                    # Draw food bounding boxes + labels
                    for fbox, fconf, fcls in zip(
                        food_boxes.xyxy.cpu().numpy(),
                        food_boxes.conf.cpu().numpy(),
                        food_boxes.cls.cpu().numpy()
                    ):
                        fx1, fy1, fx2, fy2 = map(int, fbox)
                        fx1_g = fx1 + offset_x
                        fy1_g = fy1 + offset_y
                        fx2_g = fx2 + offset_x
                        fy2_g = fy2 + offset_y

                        if not is_food_center_in_tray(
                            (fx1_g, fy1_g, fx2_g, fy2_g), parent["bbox_tuple"]
                        ):
                            continue

                        fcls_id = int(fcls)
                        if float(fconf) < conf_threshold:
                            continue

                        color = CLASS_COLORS[fcls_id]

                        cv2.rectangle(image, (fx1_g, fy1_g), (fx2_g, fy2_g), (0, 0, 0), 6)
                        cv2.rectangle(image, (fx1_g, fy1_g), (fx2_g, fy2_g), color, 3)

                        label = f"{CLASS_NAMES[fcls_id]} {fconf:.2f}"
                        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                        cv2.rectangle(image, (fx1_g, fy1_g - lh - 15),
                                      (fx1_g + lw + 10, fy1_g), color, -1)
                        cv2.putText(image, label, (fx1_g + 5, fy1_g - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
                        cv2.putText(image, label, (fx1_g + 5, fy1_g - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        elif tray_mode == "largest":
            # =============================
            # MODE LARGEST: Draw semua tray (parent + children),
            # detect food pada parent, lalu CROP image ke area parent (omprengan)
            # =============================
            sorted_trays = sorted(raw_trays, key=lambda x: x["area"], reverse=True)

            # Group nested trays (same logic as smallest mode)
            assigned_indices = set()
            groups = []
            for i, tray in enumerate(sorted_trays):
                if i in assigned_indices:
                    continue
                group = {"parent": tray, "children": []}
                assigned_indices.add(i)
                for j, other in enumerate(sorted_trays):
                    if j in assigned_indices:
                        continue
                    if is_box_inside(other["bbox_tuple"], tray["bbox_tuple"], threshold=0.7):
                        group["children"].append(other)
                        assigned_indices.add(j)
                groups.append(group)

            # Track the parent food-tray bbox for final cropping
            crop_parent_bbox = None
            tray_counter = 0

            for group in groups:
                parent = group["parent"]
                children = group["children"]

                # Remember the first (largest) parent for cropping
                if crop_parent_bbox is None:
                    crop_parent_bbox = parent["bbox_tuple"]

                # Draw ALL tray boxes (parent + children) so all 1-5 trays are visible
                all_trays_in_group = [parent] + children
                for tray_info in all_trays_in_group:
                    tray_counter += 1
                    tx1, ty1, tx2, ty2 = tray_info["bbox_tuple"]
                    tconf = tray_info["conf"]
                    tcls_id = tray_info["cls_id"]
                    tray_color = TRAY_COLORS[tcls_id]

                    cv2.rectangle(image, (tx1, ty1), (tx2, ty2), (0, 0, 0), 12)
                    cv2.rectangle(image, (tx1, ty1), (tx2, ty2), tray_color, 6)

                    tray_label = f"TRAY {tray_counter}: {TRAY_NAMES[tcls_id]} {tconf:.2f}"
                    (tw_l, th_l), _ = cv2.getTextSize(tray_label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                    cv2.rectangle(image, (tx1, ty1 - th_l - 30), (tx1 + tw_l + 20, ty1), tray_color, -1)
                    cv2.putText(image, tray_label, (tx1 + 8, ty1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5)
                    cv2.putText(image, tray_label, (tx1 + 8, ty1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

                # Crop parent tray (besar) → ML2 food detection
                image_bgr_for_crop = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                crop, (offset_x, offset_y) = crop_tray_with_padding(
                    image_bgr_for_crop, parent["bbox_tuple"], padding_ratio
                )

                food_results = model_food.predict(
                    source=crop,
                    conf=0.25,
                    iou=iou_threshold,
                    save=False,
                    verbose=False
                )

                food_result = food_results[0]
                food_boxes = food_result.boxes
                food_masks = food_result.masks

                if food_boxes is not None and len(food_boxes) > 0:
                    # Draw segmentation masks
                    if food_masks is not None:
                        for f_idx, (mask_data, fcls_m) in enumerate(zip(
                            food_masks.data.cpu().numpy(),
                            food_boxes.cls.cpu().numpy()
                        )):
                            fbox = food_boxes.xyxy.cpu().numpy()[f_idx]
                            fx1, fy1, fx2, fy2 = map(int, fbox)
                            fx1_g = fx1 + offset_x
                            fy1_g = fy1 + offset_y
                            fx2_g = fx2 + offset_x
                            fy2_g = fy2 + offset_y

                            if not is_food_center_in_tray(
                                (fx1_g, fy1_g, fx2_g, fy2_g), parent["bbox_tuple"]
                            ):
                                continue

                            fcls_id = int(fcls_m)
                            color = CLASS_COLORS[fcls_id]

                            mask_resized = cv2.resize(mask_data, (crop.shape[1], crop.shape[0]))
                            mask_bool = mask_resized > 0.5

                            full_mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
                            crop_h, crop_w = crop.shape[:2]
                            end_y = min(offset_y + crop_h, image.shape[0])
                            end_x = min(offset_x + crop_w, image.shape[1])
                            mask_h = end_y - offset_y
                            mask_w = end_x - offset_x
                            full_mask[offset_y:end_y, offset_x:end_x] = mask_bool[:mask_h, :mask_w]

                            overlay = image.copy()
                            overlay[full_mask] = color
                            image = cv2.addWeighted(overlay, mask_opacity, image, 1 - mask_opacity, 0)

                    # Draw food bounding boxes + labels
                    for fbox, fconf, fcls in zip(
                        food_boxes.xyxy.cpu().numpy(),
                        food_boxes.conf.cpu().numpy(),
                        food_boxes.cls.cpu().numpy()
                    ):
                        fx1, fy1, fx2, fy2 = map(int, fbox)
                        fx1_g = fx1 + offset_x
                        fy1_g = fy1 + offset_y
                        fx2_g = fx2 + offset_x
                        fy2_g = fy2 + offset_y

                        if not is_food_center_in_tray(
                            (fx1_g, fy1_g, fx2_g, fy2_g), parent["bbox_tuple"]
                        ):
                            continue

                        fcls_id = int(fcls)
                        if float(fconf) < conf_threshold:
                            continue

                        color = CLASS_COLORS[fcls_id]

                        cv2.rectangle(image, (fx1_g, fy1_g), (fx2_g, fy2_g), (0, 0, 0), 6)
                        cv2.rectangle(image, (fx1_g, fy1_g), (fx2_g, fy2_g), color, 3)

                        label = f"{CLASS_NAMES[fcls_id]} {fconf:.2f}"
                        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                        cv2.rectangle(image, (fx1_g, fy1_g - lh - 15),
                                      (fx1_g + lw + 10, fy1_g), color, -1)
                        cv2.putText(image, label, (fx1_g + 5, fy1_g - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
                        cv2.putText(image, label, (fx1_g + 5, fy1_g - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # =============================
            # CROP image ke area parent food-tray (omprengan) dengan sedikit padding
            # =============================
            if crop_parent_bbox is not None:
                px1, py1, px2, py2 = crop_parent_bbox
                tray_w = px2 - px1
                tray_h = py2 - py1
                # Padding 3% agar tray label di atas tetap terlihat
                pad_x = int(tray_w * 0.03)
                pad_y = int(tray_h * 0.03)
                # Extra top padding untuk tray label text
                pad_top_extra = 50
                cx1 = max(0, px1 - pad_x)
                cy1 = max(0, py1 - pad_y - pad_top_extra)
                cx2 = min(image.shape[1], px2 + pad_x)
                cy2 = min(image.shape[0], py2 + pad_y)
                image = image[cy1:cy2, cx1:cx2]

        else:
            # =============================
            # MODE ALL
            # =============================
            filtered_trays = raw_trays

            for t_idx, tray_info in enumerate(filtered_trays):
                tx1, ty1, tx2, ty2 = tray_info["bbox_tuple"]
                tconf = tray_info["conf"]
                tcls_id = tray_info["cls_id"]
                tray_bbox_original = (tx1, ty1, tx2, ty2)
                tray_color = TRAY_COLORS[tcls_id]

                cv2.rectangle(image, (tx1, ty1), (tx2, ty2), (0, 0, 0), 12)
                cv2.rectangle(image, (tx1, ty1), (tx2, ty2), tray_color, 6)

                tray_label = f"TRAY {t_idx + 1}: {TRAY_NAMES[tcls_id]} {tconf:.2f}"
                (tw_l, th_l), _ = cv2.getTextSize(tray_label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                cv2.rectangle(image, (tx1, ty1 - th_l - 30), (tx1 + tw_l + 20, ty1), tray_color, -1)
                cv2.putText(image, tray_label, (tx1 + 8, ty1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5)
                cv2.putText(image, tray_label, (tx1 + 8, ty1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

                image_bgr_for_crop = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                crop, (offset_x, offset_y) = crop_tray_with_padding(
                    image_bgr_for_crop, tray_bbox_original, padding_ratio
                )

                food_results = model_food.predict(
                    source=crop,
                    conf=0.25,
                    iou=iou_threshold,
                    save=False,
                    verbose=False
                )

                food_result = food_results[0]
                food_boxes = food_result.boxes
                food_masks = food_result.masks

                if food_boxes is not None and len(food_boxes) > 0:
                    if food_masks is not None:
                        for f_idx, (mask_data, fcls_m) in enumerate(zip(
                            food_masks.data.cpu().numpy(),
                            food_boxes.cls.cpu().numpy()
                        )):
                            fbox = food_boxes.xyxy.cpu().numpy()[f_idx]
                            fx1, fy1, fx2, fy2 = map(int, fbox)
                            fx1_g = fx1 + offset_x
                            fy1_g = fy1 + offset_y
                            fx2_g = fx2 + offset_x
                            fy2_g = fy2 + offset_y

                            if not is_food_center_in_tray(
                                (fx1_g, fy1_g, fx2_g, fy2_g), tray_bbox_original
                            ):
                                continue

                            fcls_id = int(fcls_m)
                            color = CLASS_COLORS[fcls_id]

                            mask_resized = cv2.resize(mask_data, (crop.shape[1], crop.shape[0]))
                            mask_bool = mask_resized > 0.5

                            full_mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
                            crop_h, crop_w = crop.shape[:2]
                            end_y = min(offset_y + crop_h, image.shape[0])
                            end_x = min(offset_x + crop_w, image.shape[1])
                            mask_h = end_y - offset_y
                            mask_w = end_x - offset_x
                            full_mask[offset_y:end_y, offset_x:end_x] = mask_bool[:mask_h, :mask_w]

                            overlay = image.copy()
                            overlay[full_mask] = color
                            image = cv2.addWeighted(overlay, mask_opacity, image, 1 - mask_opacity, 0)

                    for fbox, fconf, fcls in zip(
                        food_boxes.xyxy.cpu().numpy(),
                        food_boxes.conf.cpu().numpy(),
                        food_boxes.cls.cpu().numpy()
                    ):
                        fx1, fy1, fx2, fy2 = map(int, fbox)
                        fx1_g = fx1 + offset_x
                        fy1_g = fy1 + offset_y
                        fx2_g = fx2 + offset_x
                        fy2_g = fy2 + offset_y

                        if not is_food_center_in_tray(
                            (fx1_g, fy1_g, fx2_g, fy2_g), tray_bbox_original
                        ):
                            continue

                        fcls_id = int(fcls)
                        if float(fconf) < conf_threshold:
                            continue

                        color = CLASS_COLORS[fcls_id]

                        cv2.rectangle(image, (fx1_g, fy1_g), (fx2_g, fy2_g), (0, 0, 0), 6)
                        cv2.rectangle(image, (fx1_g, fy1_g), (fx2_g, fy2_g), color, 3)

                        label = f"{CLASS_NAMES[fcls_id]} {fconf:.2f}"
                        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                        cv2.rectangle(image, (fx1_g, fy1_g - lh - 15),
                                      (fx1_g + lw + 10, fy1_g), color, -1)
                        cv2.putText(image, label, (fx1_g + 5, fy1_g - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
                        cv2.putText(image, label, (fx1_g + 5, fy1_g - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # =============================
    # RETURN IMAGE
    # =============================
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', image_bgr)
    io_buf = io.BytesIO(buffer.tobytes())
    io_buf.seek(0)

    return send_file(io_buf, mimetype="image/png")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
