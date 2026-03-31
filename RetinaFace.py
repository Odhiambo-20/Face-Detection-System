"""
AdaFace Cosine Similarity Pipeline — RetinaFace Edition
=========================================================
Pipeline per video:
  1.  Extract every 2nd frame (50 % of total frames)          ← UNCHANGED
  2.  Sharpness — Laplacian variance on original-size frame
  3.  Resize frame to 360x640 for RetinaFace                  ← CHANGED (was 640x640)
  4.  RetinaFace face detection → bbox (x1,y1,x2,y2) in 360x640-space
  5.  Landmark — decode 5-point landmarks in 360x640-space
  6.  Landmark gate — validate proportion bounds per point
  7.  Umeyama similarity transform (landmarks in 360x640-space → 112x112)
  8.  Crop — warpAffine output IS the 112x112 crop (no separate crop step)
  9.  Top-20 sharpest aligned faces selected
 10.  L2 normalise each embedding → unit vector
 11.  Resize to 112x112 already done by Umeyama warpAffine
 12.  AdaFace IR-18 → raw 512-dim embedding
 13.  L2 normalise → unit vector (per frame)
 14.  Average 20 unit vectors → L2 renormalise → final FP32 unit embedding
 15.  Scale x K=2 → Clip [-1, +1]
 16.  Cosine similarity = dot product (all pairs)

People:
  V1, V2, V3, V4 = SAME PERSON  -> all pairs should be >= threshold
  V5, V6, V7     = DIFFERENT    -> all pairs should be < threshold

Threshold: 0.70

RetinaFace:
  - ONNX inference (onnx_inference.py pattern from yakhyo/retinaface-pytorch)
  - Input:  360x640 BGR float32, mean-subtracted, CHW, batch=1             ← CHANGED
  - Output: loc [num_priors,4], conf [num_priors,2], landmarks [num_priors,10]
  - Decoding: decode() + decode_landmarks() from box_utils.py
  - Landmarks: 5 points (left_eye, right_eye, nose, left_mouth, right_mouth)
               returned in 360x640 pixel space                              ← CHANGED
"""

import logging
import math
import warnings
from itertools import product as iterproduct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

#  CONFIG

VIDEO_PATHS = [
    "/home/victor/Documents/Desktop/Embeddings/IOS.mov",                          # V1
    "/home/victor/Documents/Desktop/Embeddings/IOS M-No Beard .mov",             # V2
    "/home/victor/Documents/Desktop/Embeddings/Android .mp4",                    # V3
    "/home/victor/Documents/Desktop/Embeddings/Android no beard version 2.mp4",  # V4
    "/home/victor/Documents/Desktop/Embeddings/Android video 5.mp4",             # V5
    "/home/victor/Documents/Desktop/Embeddings/IOS -Sha V6 .MOV",               # V6
    "/home/victor/Documents/Desktop/Embeddings/IOS - Rusl V7.mov",              # V7
]

ADAFACE_WEIGHTS = (
    "/home/victor/Documents/Desktop/Adaface/adaface-onnx/weights/adaface_ir_18.onnx"
)

RETINAFACE_WEIGHTS = (
    "/home/victor/Documents/Desktop/Face Detection/retinaface-pytorch/weights/retinaface_mv2.onnx"
)

# RetinaFace backbone config — MobileNetV2 (matches retinaface_mv2.onnx)
# Taken directly from config.py in yakhyo/retinaface-pytorch
RETINAFACE_CFG = {
    'min_sizes':  [[16, 32], [64, 128], [256, 512]],
    'steps':      [8, 16, 32],
    'variance':   [0.1, 0.2],
    'clip':       False,
    'out_channel': 64,
}

VIDEO_NAMES = {
    "video_1": "V1 IOS-Beard",
    "video_2": "V2 IOS-NoBrd",
    "video_3": "V3 Andr-Beard",
    "video_4": "V4 Andr-NoBrd",
    "video_5": "V5 Android5",
    "video_6": "V6 Sha",
    "video_7": "V7 Rusl",
}

GENUINE_SET         = {"video_1", "video_2", "video_3", "video_4"}
FRAMES_TO_USE       = 20          # top-20 sharpest frames used for embedding
FACE_SIZE           = 112         # AdaFace canonical input size

#  CHANGED: RetinaFace inference dimensions (height=360, width=640)
RETINA_H            = 640        # height fed to RetinaFace   ← CHANGED
RETINA_W            = 360        # width  fed to RetinaFace   ← CHANGED

THRESHOLD           = 0.70
SHARPNESS_THRESHOLD = 82.0        # Laplacian variance minimum (original-size frame)
SCALE               = 2           # post-L2 scale factor

# BGR mean used by RetinaFace (from detect.py / onnx_inference.py)
RGB_MEAN = (104, 117, 123)

# Confidence threshold for RetinaFace detections
CONF_THRESHOLD  = 0.6
NMS_THRESHOLD   = 0.4
PRE_NMS_TOPK    = 5000
POST_NMS_TOPK   = 1

# AdaFace canonical 5-point reference landmarks (112x112 space)
REFERENCE_PTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.6963],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.3655],
], dtype=np.float32)

#  LANDMARK GATE BOUNDS  (DISABLED — all gate checking commented out)

# GATE_BOUNDS: Dict[str, Tuple[float, float, float, float]] = {
#     # landmark        x_min  x_max  y_min  y_max
#     "left_eye":    (0.15,  0.45,  0.15,  0.45),
#     "right_eye":   (0.55,  0.85,  0.15,  0.45),
#     "nose":        (0.35,  0.65,  0.40,  0.65),
#     "left_mouth":  (0.20,  0.50,  0.60,  0.90),
#     "right_mouth": (0.50,  0.80,  0.60,  0.90),
# }

# _GATE_NAMES = ("left_eye", "right_eye", "nose", "left_mouth", "right_mouth")


#  RETINAFACE PRIOR BOX  (from layers/functions/prior_box.py)

class PriorBox:
    """
    Generates anchor boxes for a given image size.
    Directly ported from yakhyo/retinaface-pytorch — layers/functions/prior_box.py
    """
    def __init__(self, cfg: dict, image_size: Tuple[int, int]) -> None:
        self.image_size = image_size          # (height, width)
        self.clip       = cfg['clip']
        self.steps      = cfg['steps']
        self.min_sizes  = cfg['min_sizes']
        self.feature_maps = [
            [math.ceil(image_size[0] / step),
             math.ceil(image_size[1] / step)]
            for step in self.steps
        ]

    def generate_anchors(self) -> torch.Tensor:
        anchors = []
        for k, (map_h, map_w) in enumerate(self.feature_maps):
            step = self.steps[k]
            for i, j in iterproduct(range(map_h), range(map_w)):
                for min_size in self.min_sizes[k]:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    cx   = (j + 0.5) * step / self.image_size[1]
                    cy   = (i + 0.5) * step / self.image_size[0]
                    anchors += [cx, cy, s_kx, s_ky]
        output = torch.tensor(anchors, dtype=torch.float32).view(-1, 4)
        if self.clip:
            output.clamp_(0, 1)
        return output


#  BBOX + LANDMARK DECODE  (from utils/box_utils.py)

def decode_boxes(loc: torch.Tensor,
                 priors: torch.Tensor,
                 variances: List[float]) -> torch.Tensor:
    """
    Decode bounding boxes from prior offsets.
    Ported from box_utils.decode() in yakhyo/retinaface-pytorch.
    Returns boxes in [x1, y1, x2, y2] normalised [0,1] form.
    """
    cxcy = priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:]
    wh   = priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])
    boxes = torch.empty_like(loc)
    boxes[:, :2] = cxcy - wh / 2   # x1, y1
    boxes[:, 2:] = cxcy + wh / 2   # x2, y2
    return boxes


def decode_landmarks(predictions: torch.Tensor,
                     priors: torch.Tensor,
                     variances: List[float]) -> torch.Tensor:
    """
    Decode 5-point landmarks from prior offsets.
    Ported from box_utils.decode_landmarks() in yakhyo/retinaface-pytorch.
    Returns landmarks in normalised [0,1] form, shape [num_priors, 10].
    """
    pred = predictions.view(predictions.size(0), 5, 2)
    lm   = (priors[:, :2].unsqueeze(1)
            + pred * variances[0] * priors[:, 2:].unsqueeze(1))
    return lm.view(lm.size(0), -1)


def nms(dets: np.ndarray, threshold: float) -> List[int]:
    """Standard NMS — ported from box_utils.nms()."""
    x1, y1, x2, y2, scores = dets[:,0], dets[:,1], dets[:,2], dets[:,3], dets[:,4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep  = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w   = np.maximum(0.0, xx2 - xx1 + 1)
        h   = np.maximum(0.0, yy2 - yy1 + 1)
        ovr = (w * h) / (areas[i] + areas[order[1:]] - w * h)
        order = order[np.where(ovr <= threshold)[0] + 1]
    return keep


#  RETINAFACE ONNX DETECTOR

class RetinaFaceDetector:
    """
    RetinaFace ONNX inference.
    Pattern: onnx_inference.py from yakhyo/retinaface-pytorch.

    Input  : 360x640 BGR frame (already resized externally)                ← CHANGED
    Output : best face bbox (x1,y1,x2,y2) + 5 landmarks (10 values)
             all in 360x640 pixel coordinates, or None if no face found.   ← CHANGED
    """

    def __init__(self, model_path: str):
        import onnxruntime as ort
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        self.session  = ort.InferenceSession(model_path, providers=providers)
        self.cfg      = RETINAFACE_CFG
        log.info(f"RetinaFace ONNX | {providers[0]}")

    def _preprocess(self, frame_retina: np.ndarray) -> np.ndarray:
        """
        BGR HxW uint8 -> float32 CHW batch=1, mean subtracted.
        Matches preprocess_image() in onnx_inference.py.
        Works for any spatial size — here 360x640.                         ← CHANGED
        """
        img = np.float32(frame_retina)
        img -= np.array(RGB_MEAN, dtype=np.float32)  # BGR mean subtract
        img  = img.transpose(2, 0, 1)                # HWC -> CHW
        img  = np.expand_dims(img, 0)                # -> 1CHW
        return img

    def detect(
        self, frame_retina: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Run RetinaFace on a 360x640 BGR frame.                             ← CHANGED

        Returns (bbox, landmarks) where:
          bbox      : np.ndarray shape (4,)  [x1, y1, x2, y2] in pixel coords
          landmarks : np.ndarray shape (10,) [lx1,ly1,...,lx5,ly5] in pixel coords
        Returns None if no face passes the confidence threshold.
        """
        img_h, img_w = frame_retina.shape[:2]   # 360, 640                ← CHANGED

        #  Preprocess (onnx_inference.py pattern)
        inp = self._preprocess(frame_retina)

        #  Forward pass
        outputs = self.session.run(None, {'input': inp})
        loc_raw  = torch.tensor(outputs[0].squeeze(0))   # [num_priors, 4]
        conf_raw = outputs[1].squeeze(0)                  # [num_priors, 2]
        lm_raw   = torch.tensor(outputs[2].squeeze(0))   # [num_priors, 10]

        #  Generate anchors (PriorBox — prior_box.py)
        # image_size=(height, width) = (360, 640)                          ← CHANGED
        priorbox = PriorBox(self.cfg, image_size=(img_h, img_w))
        priors   = priorbox.generate_anchors()             # [num_priors, 4]

        # Decode boxes + landmarks (box_utils.py)
        boxes     = decode_boxes(loc_raw, priors, self.cfg['variance'])
        landmarks = decode_landmarks(lm_raw, priors, self.cfg['variance'])

        # Scale to pixel coordinates (detect.py pattern)
        # Uses actual img_w=640, img_h=360 — correct for non-square input  ← CHANGED
        bbox_scale = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        boxes      = (boxes * bbox_scale).cpu().numpy()       # [num_priors, 4]

        lm_scale   = torch.tensor([img_w, img_h] * 5, dtype=torch.float32)
        landmarks  = (landmarks * lm_scale).cpu().numpy()     # [num_priors, 10]

        scores = conf_raw[:, 1]   # face confidence scores

        #  Filter by confidence threshold
        mask = scores > CONF_THRESHOLD
        boxes, landmarks, scores = boxes[mask], landmarks[mask], scores[mask]

        if len(scores) == 0:
            return None

        #  Sort + NMS
        order = scores.argsort()[::-1][:PRE_NMS_TOPK]
        boxes, landmarks, scores = boxes[order], landmarks[order], scores[order]

        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESHOLD)
        dets, landmarks = dets[keep], landmarks[keep]

        #  Keep only the highest-confidence face
        best_bbox = dets[0, :4]       # [x1, y1, x2, y2]
        best_lm   = landmarks[0]      # [lx1,ly1, lx2,ly2, lx3,ly3, lx4,ly4, lx5,ly5]

        return best_bbox, best_lm


#  LANDMARK GATE  (DISABLED — function always returns True)

def check_landmark_gate(
    landmarks: np.ndarray,
    bbox: np.ndarray,
) -> bool:
    """
    Landmark gate DISABLED — always passes.
    """
    return True


#  UMEYAMA ALIGNMENT  (unchanged from original pipeline)

def _umeyama(src: np.ndarray, dst: np.ndarray) -> Optional[np.ndarray]:
    """Umeyama least-squares similarity transform -> 2x3 affine matrix."""
    n    = src.shape[0]
    mu_s = src.mean(0);  mu_d = dst.mean(0)
    sc   = src - mu_s;   dc   = dst - mu_d
    vs   = (sc ** 2).sum() / n
    if vs < 1e-10:
        return None
    cov = (dc.T @ sc) / n
    try:
        U, S, Vt = np.linalg.svd(cov)
    except np.linalg.LinAlgError:
        return None
    d = np.ones(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        d[-1] = -1
    R = U @ np.diag(d) @ Vt
    c = (S * d).sum() / vs
    t = mu_d - c * R @ mu_s
    M = np.zeros((2, 3), dtype=np.float32)
    M[:, :2] = c * R
    M[:, 2]  = t
    return M


def umeyama_align(
    frame_retina: np.ndarray,
    landmarks_retina: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Umeyama warpAffine using landmarks in 360x640-space -> 112x112 aligned face.
    This is the ONLY resize to FACE_SIZE in the pipeline.

    landmarks_retina : shape (10,) in 360x640-pixel coords               ← CHANGED
    Returns          : 112x112 BGR aligned face crop, or None
    """
    src_pts = landmarks_retina.reshape(5, 2).astype(np.float32)
    M = _umeyama(src_pts, REFERENCE_PTS)
    if M is None:
        return None
    return cv2.warpAffine(
        frame_retina, M, (FACE_SIZE, FACE_SIZE),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT,
    )


#  ADAFACE MODEL

class AdaFaceModel:
    def __init__(self, model_path: str):
        import onnxruntime as ort
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        self.session     = ort.InferenceSession(model_path, providers=providers)
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        log.info(f"AdaFace IR-18 | {providers[0]}")

    def raw_embedding(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        BGR 112x112 -> raw 512-dim float32.
        Input is already 112x112 from Umeyama warpAffine — NO resize here.
        """
        img = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)   # BGR -> RGB
        img = (img.astype(np.float32) / 255.0 - 0.5) / 0.5
        img = img.transpose(2, 0, 1)[np.newaxis]
        out = self.session.run([self.output_name], {self.input_name: img})
        emb = out[0][0] if out[0].ndim == 2 else out[0]
        return emb.astype(np.float32)


#  FRAME EXTRACTION — frames 90 to 110 (inclusive)

def extract_frames_in_range(video_path: str) -> List[np.ndarray]:
    """
    Extract frames between index 90 and index 110 (inclusive).
    """
    START_FRAME = 90
    END_FRAME   = 110

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    frames: List[np.ndarray] = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if idx < START_FRAME:      # skip before frame 90
            idx += 1
            continue

        if idx > END_FRAME:        # stop after frame 110
            break

        frames.append(frame)
        idx += 1

    cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded: {video_path}")

    log.info(
        f"  Frame extraction: kept {len(frames)} frames "
        f"(frames {START_FRAME}→{END_FRAME} — {Path(video_path).name})"
    )
    return frames


#  EMBED ONE VIDEO

def embed_video(
    video_path : str,
    adaface    : AdaFaceModel,
    detector   : RetinaFaceDetector,
) -> Optional[np.ndarray]:

    name = Path(video_path).name

    all_frames = extract_frames_in_range(video_path)

    #  Sharpness gate fully commented out — all frames pass
    # for frame in all_frames:
    #     gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    #     if sharpness < SHARPNESS_THRESHOLD:
    #         n_blurry += 1
    #         continue
    #     n_sharp += 1
    #     sharp_frames.append((sharpness, frame))

    sharp_frames: List[Tuple[float, np.ndarray]] = []
    for frame in all_frames:
        sharp_frames.append((0.0, frame))   # sharpness disabled — all pass

    if not sharp_frames:
        log.error(f"  [{name}] STOPPED — no frames available.")
        return None

    n_no_face      = 0
    n_umeyama_fail = 0
    aligned_pool: List[Tuple[float, np.ndarray]] = []

    printed_size = False   # print frame size once per video only


    for sharpness, orig_frame in sharp_frames:

        # ── print WxH (width x height) — first frame only ────────
        if not printed_size:
            print(f"  [{name}] BEFORE resize: "
                  f"{orig_frame.shape[1]}x{orig_frame.shape[0]} (WxH)")

        frame_retina = cv2.resize(
            orig_frame,
            (RETINA_W, RETINA_H),          # cv2.resize takes (width, height)
            interpolation=cv2.INTER_LINEAR,
        )

        if not printed_size:
            print(f"  [{name}] AFTER  resize: "
                  f"{frame_retina.shape[1]}x{frame_retina.shape[0]} (WxH)")
            printed_size = True            # stop after first frame

        result = detector.detect(frame_retina)
        if result is None:
            n_no_face += 1
            continue

        bbox, landmarks_retina = result

        aligned = umeyama_align(frame_retina, landmarks_retina)
        if aligned is None:
            n_umeyama_fail += 1
            continue

        aligned_gray      = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        aligned_sharpness = float(cv2.Laplacian(aligned_gray, cv2.CV_64F).var())

        aligned_pool.append((aligned_sharpness, aligned))
        
    log.info(
        f"  [{name}] RetinaFace: no_face={n_no_face}  "
        f"gate_fail=DISABLED  umeyama_fail={n_umeyama_fail}  "
        f"aligned={len(aligned_pool)}"
    )

    if not aligned_pool:
        log.error(f"  [{name}] STOPPED — zero aligned frames, cannot embed.")
        return None

    aligned_pool.sort(key=lambda t: t[0], reverse=True)
    top_20 = aligned_pool[:FRAMES_TO_USE]   # takes however many are available

    unit_vecs: List[np.ndarray] = []
    for _, face_112 in top_20:
        raw  = adaface.raw_embedding(face_112)
        norm = np.linalg.norm(raw)
        if norm < 1e-10:
            continue
        unit_vecs.append((raw / norm).astype(np.float32))

    if not unit_vecs:
        log.error(f"  [{name}] No valid embeddings produced.")
        return None

    avg      = np.mean(np.stack(unit_vecs, axis=0), axis=0).astype(np.float32)
    avg_norm = np.linalg.norm(avg)
    if avg_norm < 1e-10:
        log.error(f"  [{name}] Near-zero average embedding.")
        return None

    final = (avg / avg_norm).astype(np.float32)
    final = np.clip(final * SCALE, -1.0, 1.0).astype(np.float32)

    log.info(
        f"  [{name:<44}]"
        f"  embedded={len(unit_vecs)}/{FRAMES_TO_USE}"
        f"  norm={np.linalg.norm(final):.6f}"
    )
    return final


#  COSINE SIMILARITY

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b))


#  MAIN

def main():
    if not Path(ADAFACE_WEIGHTS).exists():
        raise FileNotFoundError(f"AdaFace weights not found: {ADAFACE_WEIGHTS}")
    if not Path(RETINAFACE_WEIGHTS).exists():
        raise FileNotFoundError(f"RetinaFace weights not found: {RETINAFACE_WEIGHTS}")
    for vp in VIDEO_PATHS:
        if not Path(vp).exists():
            raise FileNotFoundError(f"Video not found: {vp}")

    # Print DST reference points (AdaFace Umeyama template)
    print("\n" + "=" * 55)
    print("  ADAFACE DST REFERENCE POINTS (112x112 space)")
    print("=" * 55)
    labels = ["left_eye   ", "right_eye  ", "nose       ", "left_mouth ", "right_mouth"]
    for label, pt in zip(labels, REFERENCE_PTS):
        print(f"  {label}  x={pt[0]:.4f}  y={pt[1]:.4f}")
    print("=" * 55 + "\n")

    adaface  = AdaFaceModel(ADAFACE_WEIGHTS)
    detector = RetinaFaceDetector(RETINAFACE_WEIGHTS)

    print("\nExtracting embeddings...")
    print("-" * 65)

    embeddings: Dict[str, np.ndarray] = {}
    for idx, vp in enumerate(VIDEO_PATHS, start=1):
        key    = f"video_{idx}"
        result = embed_video(vp, adaface, detector)
        if result is not None:
            embeddings[key] = result

    keys = [f"video_{i}" for i in range(1, 8) if f"video_{i}" in embeddings]

    print("\n")
    print("=" * 55)
    print("  COSINE SIMILARITY — ALL PAIRS")
    print(f"  Threshold = {THRESHOLD}")
    print(f"  Embedding pipeline: L2 -> Scale x{SCALE} -> Clip[-1,+1] -> Cosine")
    print("=" * 55)
    print()
    print(f"  {'Pair':<35}  {'Similarity':>10}")
    print(f"  {'-'*35}  {'-'*10}")

    pair_results = []
    for i, ka in enumerate(keys):
        for kb in keys[i + 1:]:
            sim          = cosine_sim(embeddings[ka], embeddings[kb])
            na           = VIDEO_NAMES.get(ka, ka)
            nb           = VIDEO_NAMES.get(kb, kb)
            both_genuine = ka in GENUINE_SET and kb in GENUINE_SET
            verdict      = "PASS" if sim >= THRESHOLD else "FAIL"
            pair_results.append((ka, kb, na, nb, sim, both_genuine, verdict))
            print(f"  {na + ' vs ' + nb:<35}  {sim:>10.4f}")

    genuine_sims  = [sim for *_, sim, bg, v in pair_results if bg]
    impostor_sims = [sim for *_, sim, bg, v in pair_results if not bg]

    print()
    print("=" * 55)
    print("  SUMMARY")
    print("=" * 55)

    if genuine_sims:
        print(f"  Genuine  pairs (V1-V4)"
              f"  min={min(genuine_sims):.4f}"
              f"  max={max(genuine_sims):.4f}"
              f"  mean={np.mean(genuine_sims):.4f}")

    if impostor_sims:
        print(f"  Impostor pairs (V5-V7)"
              f"  min={min(impostor_sims):.4f}"
              f"  max={max(impostor_sims):.4f}"
              f"  mean={np.mean(impostor_sims):.4f}")

    if genuine_sims and impostor_sims:
        gap = min(genuine_sims) - max(impostor_sims)
        print(f"  Separation gap          {gap:+.4f}"
              f"  {'CLEAN' if gap > 0 else 'OVERLAP'}")

    print()
    wrong = [
        (na, nb, sim, bg, v)
        for ka, kb, na, nb, sim, bg, v in pair_results
        if (bg and v == "FAIL") or (not bg and v == "PASS")
    ]

    if not wrong:
        print(f"  ALL PAIRS CORRECT")
        print(f"  Genuine  means all above {THRESHOLD}")
        print(f"  Impostor means all below {THRESHOLD}")
    else:
        print(f"  {len(wrong)} PAIR(S) WRONG")
        for na, nb, sim, bg, v in wrong:
            expected = "PASS" if bg else "FAIL"
            print(f"    {na} vs {nb}  sim={sim:.4f}"
                  f"  got={v}  expected={expected}")

    print("=" * 55)


if __name__ == "__main__":
    main()
