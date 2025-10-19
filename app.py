import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from flask import Flask, render_template, request, jsonify
import base64, logging
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
import cv2

try:
    from mtcnn import MTCNN

    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    print("MTCNN not available. Install with: pip install mtcnn")

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
log = logging.getLogger("predict")

try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except Exception as e:
    log.warning(f"GPU config: {e}")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


log.info("Loading models...")
try:
    model1 = tf.keras.models.load_model('rasnet50_model.h5', compile=False)
    model2 = tf.keras.models.load_model('vgg16_model.h5', compile=False)
    log.info(f"Model 1 input: {model1.input_shape}, VGG16 input: {model2.input_shape}")
except Exception as e:
    log.error(f"Model load error: {e}")
    model1, model2 = None, None

# OPTIMIZED CONFIG - Better confidence distribution
MODEL_CONFIG = {
    "model1": {
        "preprocess": "resnet",
        "positive_class": "Non-Autistic",
        "threshold_autistic": 0.40,  # Wider separation
        "threshold_non_autistic": 0.60,
        "tta": True,
        "tta_count": 8,
        "calibration_temp": None
    },
    "vgg16": {
        "preprocess": "vgg16",
        "positive_class": "Non-Autistic",
        "threshold_autistic": 0.39,
        "threshold_non_autistic": 0.61,
        "tta": True,
        "tta_count": 8,
        "calibration_temp": None
    }
}

# OPTIMIZED TIERS - More decisive predictions
CONFIDENCE_TIERS = {
    "high": 0.68,  # Lowered from 0.78
    "medium": 0.56,  # Lowered from 0.62
    "low": 0.46  # Lowered from 0.52
}

SAFETY_CONFIG = {
    "face_required": True,
    "auto_crop_face": True,
    "min_face_size": 15,
    "max_faces": 5,
    "ood_min_conf": 0.50,
    "ood_max_entropy": 0.92,
    "ood_energy_threshold": -2.5,
    "ood_combined_threshold": True,
    "ensemble_mode": "weighted_vote",
    "max_dimension": 3000,
    "adversarial_smooth": True,
    "human_face_check": True,
    "min_skin_ratio": 0.12,
    "min_sharpness": 40,
    "max_edge_density": 0.80,
    "min_color_variance": 180,
    "min_confidence_score": 0.25,
    "detection_order": ["mtcnn", "dnn", "haar"]
}


# ---- FACE DETECTORS ----

def _load_haar_cascade():
    haar_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    haar = cv2.CascadeClassifier(haar_path) if os.path.exists(haar_path) else None
    if haar is not None and not haar.empty():
        log.info("✓ Haar cascade loaded")
        return haar
    log.warning("✗ Haar cascade unavailable")
    return None


def _load_dnn_detector():
    try:
        model_file = "opencv_face_detector_uint8.pb"
        config_file = "opencv_face_detector.pbtxt"

        if not os.path.exists(model_file) or not os.path.exists(config_file):
            import urllib.request
            base_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/"
            urllib.request.urlretrieve(base_url + model_file, model_file)
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/" + config_file,
                config_file
            )

        net = cv2.dnn.readNetFromTensorflow(model_file, config_file)
        log.info("✓ DNN face detector loaded")
        return net
    except Exception as e:
        log.warning(f"✗ DNN detector unavailable: {e}")
        return None


def _load_mtcnn_detector():
    if MTCNN_AVAILABLE:
        try:
            detector = MTCNN()
            log.info("✓ MTCNN detector loaded")
            return detector
        except Exception as e:
            log.warning(f"✗ MTCNN detector error: {e}")
            return None
    return None


HAAR_DET = _load_haar_cascade()
DNN_DET = _load_dnn_detector()
MTCNN_DET = _load_mtcnn_detector()


def adversarial_defense(pil_img):
    return pil_img.filter(ImageFilter.GaussianBlur(radius=0.5))


def smart_resize(pil_img):
    w, h = pil_img.size
    max_dim = SAFETY_CONFIG["max_dimension"]
    resized = False

    if w > max_dim or h > max_dim:
        ratio = min(max_dim / w, max_dim / h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        resized = True
        log.info(f"Resized from {w}x{h} to {new_w}x{new_h}")

    return pil_img, resized


# ---- FACE DETECTION ----

def detect_faces_mtcnn(pil_img):
    if MTCNN_DET is None:
        return []
    try:
        img_array = np.array(pil_img)
        detections = MTCNN_DET.detect_faces(img_array)
        boxes = []
        for detection in detections:
            if detection['confidence'] > 0.90:
                x, y, w, h = detection['box']
                boxes.append((x, y, x + w, y + h))
        log.info(f"MTCNN detected {len(boxes)} faces")
        return boxes
    except Exception as e:
        log.warning(f"MTCNN detection failed: {e}")
        return []


def detect_faces_dnn(pil_img):
    if DNN_DET is None:
        return []
    try:
        img_array = np.array(pil_img)
        h, w = img_array.shape[:2]
        blob = cv2.dnn.blobFromImage(img_array, 1.0, (300, 300), [104, 117, 123], False, False)
        DNN_DET.setInput(blob)
        detections = DNN_DET.forward()
        boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                boxes.append((x1, y1, x2, y2))
        log.info(f"DNN detected {len(boxes)} faces")
        return boxes
    except Exception as e:
        log.warning(f"DNN detection failed: {e}")
        return []


def detect_faces_haar(pil_img):
    if HAAR_DET is None or HAAR_DET.empty():
        return []

    boxes = []
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)
    except Exception:
        gray_eq = gray

    rects = HAAR_DET.detectMultiScale(gray_eq, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in rects:
        boxes.append((x, y, x + w, y + h))

    if not boxes:
        rects2 = HAAR_DET.detectMultiScale(gray_eq, scaleFactor=1.03, minNeighbors=2, minSize=(24, 24),
                                           flags=cv2.CASCADE_SCALE_IMAGE)
        for (x, y, w, h) in rects2:
            boxes.append((x, y, x + w, y + h))

    if not boxes:
        rects3 = HAAR_DET.detectMultiScale(gray, scaleFactor=1.02, minNeighbors=1, minSize=(20, 20),
                                           flags=cv2.CASCADE_SCALE_IMAGE)
        for (x, y, w, h) in rects3:
            boxes.append((x, y, x + w, y + h))

    if not boxes:
        for scale in [1.5, 1.0, 0.7]:
            if scale != 1.0:
                w_img, h_img = gray.shape[1], gray.shape[0]
                scaled_gray = cv2.resize(gray, (int(w_img * scale), int(h_img * scale)), interpolation=cv2.INTER_LINEAR)
            else:
                scaled_gray = gray

            rects4 = HAAR_DET.detectMultiScale(scaled_gray, scaleFactor=1.05, minNeighbors=2, minSize=(15, 15),
                                               flags=cv2.CASCADE_SCALE_IMAGE)
            for (x, y, w, h) in rects4:
                if scale != 1.0:
                    x, y, w, h = int(x / scale), int(y / scale), int(w / scale), int(h / scale)
                boxes.append((x, y, x + w, y + h))

            if boxes:
                break

    log.info(f"Haar detected {len(boxes)} faces")
    return boxes


def detect_faces_pil(pil_img):
    detection_order = SAFETY_CONFIG.get("detection_order", ["mtcnn", "dnn", "haar"])

    for method in detection_order:
        boxes = []

        if method == "mtcnn":
            boxes = detect_faces_mtcnn(pil_img)
        elif method == "dnn":
            boxes = detect_faces_dnn(pil_img)
        elif method == "haar":
            boxes = detect_faces_haar(pil_img)

        min_sz = SAFETY_CONFIG["min_face_size"]
        filtered = [(x1, y1, x2, y2) for (x1, y1, x2, y2) in boxes
                    if (x2 - x1) >= min_sz and (y2 - y1) >= min_sz]

        if filtered:
            log.info(f"✓ Face detection successful using {method.upper()} ({len(filtered)} faces)")
            return filtered

    log.warning("No faces detected by any algorithm")
    return []


def crop_to_largest_face(pil_img, boxes):
    if not boxes:
        return pil_img
    (x1, y1, x2, y2) = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    return pil_img.crop((x1, y1, x2, y2))


# ---- HUMAN FACE VERIFICATION ----

def check_sharpness(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(lap_var)


def check_skin_tone(img_array):
    ycrcb = cv2.cvtColor(img_array, cv2.COLOR_RGB2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    skin_ratio = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
    return float(skin_ratio)


def check_edge_density(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    return float(edge_ratio)


def check_color_variance(img_array):
    var_r = np.var(img_array[:, :, 0])
    var_g = np.var(img_array[:, :, 1])
    var_b = np.var(img_array[:, :, 2])
    total_var = var_r + var_g + var_b
    return float(total_var)


def verify_human_face(pil_img, face_box):
    if not SAFETY_CONFIG["human_face_check"]:
        return True, 1.0, []

    x1, y1, x2, y2 = face_box
    face_crop = pil_img.crop((x1, y1, x2, y2))
    face_array = np.array(face_crop)

    if face_array.shape[0] < 30 or face_array.shape[1] < 30:
        return False, 0.0, ["detected region too small for verification"]

    issues = []
    confidence_score = 1.0
    flags = []

    sharpness = check_sharpness(face_array)
    if sharpness < SAFETY_CONFIG["min_sharpness"]:
        issues.append(f"extremely blurry image (sharpness={sharpness:.1f})")
        confidence_score *= 0.4
        flags.append("blur")
    elif sharpness < 80:
        confidence_score *= 0.8

    skin_ratio = check_skin_tone(face_array)
    if skin_ratio < SAFETY_CONFIG["min_skin_ratio"]:
        if skin_ratio < 0.05:
            issues.append(f"no human skin tone detected - may be animal/object/document ({skin_ratio:.1%})")
        else:
            issues.append(f"insufficient human skin tone ({skin_ratio:.1%})")
        confidence_score *= 0.5
        flags.append("skin")
    elif skin_ratio < 0.20:
        confidence_score *= 0.85

    edge_density = check_edge_density(face_array)
    if edge_density > SAFETY_CONFIG["max_edge_density"]:
        issues.append(f"cartoon/drawing/sketch-like appearance ({edge_density:.1%} edges)")
        confidence_score *= 0.3
        flags.append("cartoon")
    elif edge_density > 0.65:
        confidence_score *= 0.85

    color_var = check_color_variance(face_array)
    if color_var < SAFETY_CONFIG["min_color_variance"]:
        issues.append(f"unnatural/flat color distribution (variance={color_var:.0f})")
        confidence_score *= 0.6
        flags.append("color")
    elif color_var < 300:
        confidence_score *= 0.9

    rgb_std = [np.std(face_array[:, :, i]) for i in range(3)]
    color_diversity = max(rgb_std) - min(rgb_std)
    if color_diversity < 5:
        issues.append("grayscale or monochrome image detected")
        confidence_score *= 0.6
        flags.append("grayscale")

    gray = cv2.cvtColor(face_array, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_ratio = np.sum(binary == 255) / binary.size
    black_ratio = np.sum(binary == 0) / binary.size
    if white_ratio > 0.7 or black_ratio > 0.7:
        issues.append("predominantly uniform regions detected (may be document/text)")
        confidence_score *= 0.5
        flags.append("uniform")

    min_conf = SAFETY_CONFIG["min_confidence_score"]
    is_human = confidence_score >= min_conf

    if not is_human:
        if "skin" in flags and "cartoon" not in flags:
            summary = "This appears to be an animal, object, or document rather than a human face."
        elif "cartoon" in flags:
            summary = "This appears to be a cartoon, drawing, or illustration rather than a real photo."
        elif "blur" in flags:
            summary = "Image is too blurry for accurate analysis."
        elif "uniform" in flags or "grayscale" in flags:
            summary = "This appears to be a document, text, or non-photographic image."
        else:
            summary = "This does not appear to be a valid human face photograph."

        issues.insert(0, summary)

    log.info(
        f"Human face verification: conf={confidence_score:.2f}, sharp={sharpness:.1f}, "
        f"skin={skin_ratio:.2%}, edge={edge_density:.2%}, color_var={color_var:.0f}, "
        f"flags={flags}, is_human={is_human}"
    )

    return is_human, confidence_score, issues


# ---- PREDICTION FUNCTIONS ----

def predictive_entropy(vec):
    v = np.asarray(vec).reshape(-1)
    if v.shape[0] == 1:
        p = float(np.clip(v[0], 1e-6, 1 - 1e-6))
        return - (p * np.log(p) + (1 - p) * np.log(1 - p))
    v = v / (np.sum(v) + 1e-9)
    v = np.clip(v, 1e-9, 1.0)
    return - float(np.sum(v * np.log(v)))


def energy_score(vec):
    logits = np.asarray(vec).reshape(-1)
    if logits.shape[0] == 1:
        p = float(np.clip(logits[0], 1e-6, 1 - 1e-6))
        logit = np.log(p / (1.0 - p))
        return float(logit)
    return float(np.log(np.sum(np.exp(logits))))


def prep_array(arr, mode):
    if mode == "rescale":
        return arr.astype("float32") / 255.0
    if mode == "vgg16":
        return vgg16_preprocess(arr.astype("float32"))
    if mode == "resnet":
        return resnet_preprocess(arr.astype("float32"))
    return arr


def get_hw(model):
    return model.input_shape[1], model.input_shape[2]


def _maybe_temperature_scale(p, T):
    if not T or T == 1.0:
        return float(p)
    eps = 1e-6
    p = float(min(max(p, eps), 1.0 - eps))
    logit = np.log(p / (1.0 - p))
    return float(1.0 / (1.0 + np.exp(-logit / float(T))))


def _apply_tiered_classification(p_pos, cfg, model_key):
    thr_a = float(cfg.get("threshold_autistic", 0.5))
    thr_n = float(cfg.get("threshold_non_autistic", 0.5))
    pos_label = cfg.get("positive_class", "Non-Autistic")

    if pos_label == "Non-Autistic":
        p_aut = 1.0 - p_pos
        if p_pos >= thr_n:
            raw_label = "Non-Autistic"
            conf = p_pos
        elif p_aut >= (1.0 - thr_a):
            raw_label = "Autistic"
            conf = p_aut
        else:
            raw_label = "Non-Autistic" if p_pos > 0.5 else "Autistic"
            conf = max(p_pos, p_aut)
        a_pct = round(p_aut * 100, 2)
        n_pct = round(p_pos * 100, 2)
        true_labels = ["Autistic", "Non-Autistic"]
    else:
        p_non = 1.0 - p_pos
        if p_pos >= thr_a:
            raw_label = "Autistic"
            conf = p_pos
        elif p_non >= thr_n:
            raw_label = "Non-Autistic"
            conf = p_non
        else:
            raw_label = "Autistic" if p_pos > 0.5 else "Non-Autistic"
            conf = max(p_pos, p_non)
        a_pct = round(p_pos * 100, 2)
        n_pct = round(p_non * 100, 2)
        true_labels = ["Non-Autistic", "Autistic"]

    if conf >= CONFIDENCE_TIERS["high"]:
        tier = "high"
        final_label = raw_label
    elif conf >= CONFIDENCE_TIERS["medium"]:
        tier = "medium"
        final_label = f"Likely {raw_label}"
    elif conf >= CONFIDENCE_TIERS["low"]:
        tier = "low"
        final_label = f"Possibly {raw_label}"
    else:
        tier = "very_low"
        final_label = f"Uncertain - leaning {raw_label}"

    log.info(f"[{model_key}] p_pos={p_pos:.4f} -> {final_label} (conf={conf:.4f}, tier={tier})")
    return final_label, conf, a_pct, n_pct, true_labels, tier


def _predict_core(arr_batched, model):
    pred = model.predict(arr_batched, verbose=0)[0]
    return np.asarray(pred).reshape(-1)


def _tta_views_enhanced(img, w, h, count=8):
    base = np.array(img.resize((w, h)))
    views = [base, np.fliplr(base)]

    for scale in [0.92, 0.88, 0.82]:
        margin = int((1.0 - scale) * w / 2)
        crop = Image.fromarray(base).crop((margin, margin, w - margin, h - margin)).resize((w, h))
        views.append(np.array(crop))

    pil_base = Image.fromarray(base)
    for angle in [-5, -2, 2, 5]:
        rot = pil_base.rotate(angle, fillcolor=(128, 128, 128), resample=Image.Resampling.BICUBIC)
        views.append(np.array(rot.resize((w, h))))

    enhancer = ImageEnhance.Brightness(pil_base)
    for factor in [0.92, 1.08]:
        bright = enhancer.enhance(factor)
        views.append(np.array(bright))

    return views[:count]


def predict_with_model(img, model, cfg, model_key):
    h, w = get_hw(model)
    tta_enabled = bool(cfg.get("tta", False))
    tta_count = int(cfg.get("tta_count", 8))
    arrays = _tta_views_enhanced(img, w, h, count=tta_count) if tta_enabled else [np.array(img.resize((w, h)))]
    mode = cfg["preprocess"]

    probs = []
    raw_vectors = []
    for arr in arrays:
        arr_p = prep_array(arr, mode)
        arr_p = np.expand_dims(arr_p, 0)
        vec = _predict_core(arr_p, model)
        raw_vectors.append(vec.tolist())

        if vec.shape[0] == 1:
            p = float(vec[0])
        elif vec.shape[0] == 2:
            p = float(vec[1]) if cfg.get("positive_class") == "Non-Autistic" else float(vec[0])
        else:
            raise ValueError(f"{model_key} unexpected output shape {vec.shape[0]}")
        probs.append(p)

    probs_array = np.array(probs)
    mean_p = np.mean(probs_array)
    std_p = np.std(probs_array)
    mask = np.abs(probs_array - mean_p) <= 2 * std_p
    filtered_probs = probs_array[mask] if mask.sum() > 0 else probs_array

    weights = np.exp(-0.5 * ((filtered_probs - 0.5) ** 2) / 0.12)
    weights /= weights.sum()
    p_mean = float(np.average(filtered_probs, weights=weights))

    p_pos = _maybe_temperature_scale(p_mean, cfg.get("calibration_temp", None))
    final_label, conf, a_pct, n_pct, true_labels, tier = _apply_tiered_classification(p_pos, cfg, model_key)

    ent = predictive_entropy(raw_vectors[0])
    e_score = energy_score(raw_vectors[0])
    max_prob = max(p_pos, 1.0 - p_pos)
    tta_std = float(np.std(probs_array))

    result = {
        "predicted_class": final_label,
        "confidence": conf,
        "confidence_tier": tier,
        "autistic_percentage": a_pct,
        "non_autistic_percentage": n_pct,
        "true_labels": true_labels,
        "output_vector": raw_vectors[0],
        "tta_prob_mean": p_mean,
        "tta_std": tta_std,
        "entropy": ent,
        "energy_score": e_score,
        "input_size": f"{w}x{h}",
        "tta_views": len(arrays)
    }

    ood_flags = []
    ood_score = 0

    if max_prob < SAFETY_CONFIG["ood_min_conf"]:
        ood_flags.append(f"low confidence ({max_prob:.2f})")
        ood_score += 1
    if ent > SAFETY_CONFIG["ood_max_entropy"]:
        ood_flags.append(f"high uncertainty (entropy={ent:.2f})")
        ood_score += 1
    if e_score < SAFETY_CONFIG["ood_energy_threshold"]:
        ood_flags.append(f"low energy ({e_score:.2f})")
        ood_score += 1
    if tta_std > 0.15:
        ood_flags.append(f"unstable prediction (std={tta_std:.2f})")
        ood_score += 1

    if SAFETY_CONFIG.get("ood_combined_threshold", True):
        if ood_score >= 2:
            result["ood_warning"] = f"Prediction may be unreliable ({ood_score} indicators): " + ", ".join(ood_flags)
    else:
        if ood_flags:
            result["ood_warning"] = "Prediction may be less reliable: " + ", ".join(ood_flags)

    return result


# OPTIMIZED ENSEMBLE - Better confidence boost
def ensemble_predictions(res1, res2):
    c1, c2 = res1.get("confidence", 0.5), res2.get("confidence", 0.5)
    l1, l2 = res1.get("predicted_class"), res2.get("predicted_class")

    def extract_base(lbl):
        for prefix in ["Likely ", "Possibly ", "Uncertain - leaning "]:
            if lbl.startswith(prefix):
                return lbl.replace(prefix, "")
        return lbl

    base1, base2 = extract_base(l1), extract_base(l2)

    if base1 == base2:
        # MODELS AGREE - Give 20% confidence boost!
        avg_conf = (c1 + c2) / 2.0
        avg_conf = min(avg_conf * 1.20, 1.0)

        if avg_conf >= CONFIDENCE_TIERS["high"]:
            final_label = base1
            final_tier = "high"
        elif avg_conf >= CONFIDENCE_TIERS["medium"]:
            final_label = f"Likely {base1}"
            final_tier = "medium"
        elif avg_conf >= CONFIDENCE_TIERS["low"]:
            final_label = f"Possibly {base1}"
            final_tier = "low"
        else:
            final_label = f"Uncertain - leaning {base1}"
            final_tier = "very_low"

        result = {
            "predicted_class": final_label,
            "confidence": avg_conf,
            "confidence_tier": final_tier,
            "autistic_percentage": (res1.get("autistic_percentage", 0) + res2.get("autistic_percentage", 0)) / 2,
            "non_autistic_percentage": (res1.get("non_autistic_percentage", 0) + res2.get("non_autistic_percentage",
                                                                                          0)) / 2,
            "ensemble_method": "weighted_vote_agree"
        }

        ood_warns = []
        if res1.get("ood_warning"):
            ood_warns.append(f"Model1: {res1['ood_warning']}")
        if res2.get("ood_warning"):
            ood_warns.append(f"VGG16: {res2['ood_warning']}")
        if ood_warns:
            result["ood_warning"] = " | ".join(ood_warns)

        return result
    else:
        # MODELS DISAGREE - Use stronger weighting
        w1 = c1 ** 2.2
        w2 = c2 ** 2.2
        total_w = w1 + w2

        a_avg = (res1.get("autistic_percentage", 0) * w1 + res2.get("autistic_percentage", 0) * w2) / total_w
        n_avg = (res1.get("non_autistic_percentage", 0) * w1 + res2.get("non_autistic_percentage", 0) * w2) / total_w

        final_label = "Autistic" if a_avg > n_avg else "Non-Autistic"
        final_conf = max(a_avg, n_avg) / 100.0

        if final_conf >= CONFIDENCE_TIERS["high"]:
            tier = "high"
        elif final_conf >= CONFIDENCE_TIERS["medium"]:
            final_label = f"Likely {final_label}"
            tier = "medium"
        elif final_conf >= CONFIDENCE_TIERS["low"]:
            final_label = f"Possibly {final_label}"
            tier = "low"
        else:
            final_label = f"Uncertain - leaning {final_label}"
            tier = "very_low"

        result = {
            "predicted_class": final_label,
            "confidence": final_conf,
            "confidence_tier": tier,
            "autistic_percentage": a_avg,
            "non_autistic_percentage": n_avg,
            "ensemble_method": "weighted_avg_disagree"
        }

        ood_warns = []
        if res1.get("ood_warning"):
            ood_warns.append(f"Model1: {res1['ood_warning']}")
        if res2.get("ood_warning"):
            ood_warns.append(f"VGG16: {res2['ood_warning']}")
        if ood_warns:
            result["ood_warning"] = " | ".join(ood_warns)

        return result


def predict_image(image_path, model_choice="both"):
    try:
        with Image.open(image_path) as im:
            img = im.convert('RGB')

        original_size = img.size
        img, was_resized = smart_resize(img)

        if SAFETY_CONFIG["adversarial_smooth"]:
            img = adversarial_defense(img)

        warnings = []

        if SAFETY_CONFIG["face_required"]:
            faces = detect_faces_pil(img)
            if not faces:
                return {
                    "status": "no_face",
                    "message": "No face detected. Please upload a clear face image.",
                    "filename": os.path.basename(image_path)
                }

            if len(faces) > SAFETY_CONFIG["max_faces"]:
                return {
                    "status": "quality_error",
                    "message": f"Too many faces detected ({len(faces)}). Please upload image with fewer faces.",
                    "filename": os.path.basename(image_path)
                }
            elif len(faces) > 1:
                warnings.append(f"Multiple faces detected ({len(faces)}). Using largest face.")

            largest_face = max(faces, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            is_human, human_conf, human_issues = verify_human_face(img, largest_face)

            if not is_human:
                return {
                    "status": "not_human_face",
                    "message": "Image might not be a real human face or it might be too blurry.",
                    "details": human_issues,
                    "confidence": round(human_conf, 2),
                    "filename": os.path.basename(image_path)
                }

            if human_conf < 0.6:
                warnings.append(f"Face verification confidence low ({human_conf:.1%}): " + ", ".join(human_issues))

            if SAFETY_CONFIG["auto_crop_face"]:
                img = crop_to_largest_face(img, faces)

        if was_resized:
            warnings.append(f"Image resized from {original_size[0]}x{original_size[1]} to {img.size[0]}x{img.size[1]}.")

        results = {}
        if model_choice in ["model1", "both"] and model1 is not None:
            results["model1"] = predict_with_model(img, model1, MODEL_CONFIG["model1"], "model1")
        if model_choice in ["vgg16", "both"] and model2 is not None:
            results["vgg16"] = predict_with_model(img, model2, MODEL_CONFIG["vgg16"], "vgg16")

        if model_choice == "both" and "model1" in results and "vgg16" in results:
            results["ensemble"] = ensemble_predictions(results["model1"], results["vgg16"])

        if warnings:
            for key in results:
                if isinstance(results[key], dict):
                    if "quality_warnings" not in results[key]:
                        results[key]["quality_warnings"] = warnings
                    else:
                        results[key]["quality_warnings"].extend(warnings)

        return results

    except Exception as e:
        log.error(f"Prediction error: {e}")
        return {"error": f"Prediction error: {str(e)}"}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/resize')
def resize_page():
    return render_template('resize.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        f = request.files['file']
        if f.filename == '':
            return jsonify({'error': 'No file selected'})
        if not allowed_file(f.filename):
            return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, GIF, BMP'})

        filename = secure_filename(f.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(path)

        choice = request.form.get('model', 'both')
        results = predict_image(path, choice)

        try:
            with open(path, 'rb') as fh:
                image_b64 = base64.b64encode(fh.read()).decode()
        except Exception:
            image_b64 = None

        if isinstance(results, dict) and ('status' in results or 'error' in results):
            if image_b64:
                results['image_data'] = image_b64
            results['filename'] = filename
            try:
                os.remove(path)
            except Exception:
                pass
            return jsonify(results)

        if 'error' not in results and image_b64:
            results['image_data'] = image_b64
            results['filename'] = filename

        try:
            os.remove(path)
        except Exception:
            pass

        return jsonify(results)

    except Exception as e:
        log.error(f"Server error: {e}")
        return jsonify({'error': f'Server error: {str(e)}'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
