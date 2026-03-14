"""
Dual-engine OCR pipeline:
  - EasyOCR  → English / Latin script
  - TrOCR    → Arabic script (community model from HuggingFace)

Script is auto-detected per ROI and routed to the appropriate engine.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import unicodedata
import logging
from functools import lru_cache

import cv2
import numpy as np
import easyocr
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model loading (lazy singletons)
# ---------------------------------------------------------------------------

# Community Arabic handwriting TrOCR model.
# Replace with your own fine-tuned model ID if you train one later.
ARABIC_TROCR_MODEL_ID = "microsoft/trocr-base-handwritten"
# Better alternative when available on the Hub:
#   "wikidepia/arabic-trocr-handwritten"
#   "anzhc/arabic-trocr-handwritten"
# Swap the ID above once you verify which community model fits your documents.


@lru_cache(maxsize=1)
def _load_easyocr() -> easyocr.Reader:
    """Load EasyOCR reader for English (Latin script only)."""
    logger.info("Loading EasyOCR (English)...")
    return easyocr.Reader(["en"], gpu=True, verbose=False)


@lru_cache(maxsize=1)
def _load_trocr() -> tuple[TrOCRProcessor, VisionEncoderDecoderModel]:
    """Load TrOCR processor + model for Arabic handwriting."""
    logger.info("Loading TrOCR model: %s", ARABIC_TROCR_MODEL_ID)
    processor = TrOCRProcessor.from_pretrained(ARABIC_TROCR_MODEL_ID)
    model = VisionEncoderDecoderModel.from_pretrained(ARABIC_TROCR_MODEL_ID)
    model.eval()
    return processor, model


# ---------------------------------------------------------------------------
# Script detection
# ---------------------------------------------------------------------------

_ARABIC_BLOCK_NAMES = frozenset({
    "ARABIC",
    "ARABIC SUPPLEMENT",
    "ARABIC EXTENDED-A",
    "ARABIC EXTENDED-B",
    "ARABIC PRESENTATION FORMS-A",
    "ARABIC PRESENTATION FORMS-B",
})


def _is_arabic_char(ch: str) -> bool:
    try:
        block = unicodedata.name(ch, "").split(" ")[0:2]
    except ValueError:
        return False
    # Check first word or first two words of the Unicode name
    return block[0] == "ARABIC" or " ".join(block) in _ARABIC_BLOCK_NAMES


def detect_script(img: np.ndarray) -> str:
    """
    Run a quick EasyOCR pass to sample text and classify the dominant script.

    Returns "arabic" or "latin".

    Strategy: use EasyOCR with both language packs on a downscaled image
    for speed, count Arabic vs Latin characters in the output.
    """
    # Downscale for fast detection (not accuracy-critical)
    h, w = img.shape[:2]
    scale = min(1.0, 640 / max(h, w))
    if scale < 1.0:
        small = cv2.resize(img, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
    else:
        small = img

    # Use a lightweight reader with both scripts just for detection
    reader = _get_detection_reader()
    results = reader.readtext(small, detail=0, paragraph=False)
    text = " ".join(results)

    arabic_count = sum(1 for ch in text if _is_arabic_char(ch))
    latin_count = sum(1 for ch in text if ch.isascii() and ch.isalpha())

    if arabic_count == 0 and latin_count == 0:
        # No text detected — default to latin (EasyOCR is more forgiving)
        return "latin"

    return "arabic" if arabic_count > latin_count else "latin"


@lru_cache(maxsize=1)
def _get_detection_reader() -> easyocr.Reader:
    """Bilingual reader used only for script detection."""
    logger.info("Loading bilingual EasyOCR reader for script detection...")
    return easyocr.Reader(["en", "ar"], gpu=True, verbose=False)


# ---------------------------------------------------------------------------
# Engine wrappers
# ---------------------------------------------------------------------------

def _ocr_easyocr(img: np.ndarray, detail: bool = False) -> list[dict]:
    """
    Run EasyOCR on a preprocessed image region.

    Returns a list of dicts:
      {"text": str, "confidence": float, "bbox": list[list[int]]}
    """
    reader = _load_easyocr()
    raw = reader.readtext(
        img,
        detail=1,
        paragraph=False,
        text_threshold=0.5,
        low_text=0.3,
    )
    results = []
    for bbox, text, conf in raw:
        entry = {"text": text, "confidence": float(conf)}
        if detail:
            entry["bbox"] = bbox
        results.append(entry)
    return results


def _ocr_trocr(img: np.ndarray) -> list[dict]:
    """
    Run TrOCR on a preprocessed image region.

    TrOCR expects a single text-line image.  If the region contains
    multiple lines, split it first (see `split_lines` helper below).

    Returns a list of dicts:
      {"text": str, "confidence": float}
    """
    processor, model = _load_trocr()
    lines = split_lines(img)
    results = []
    for line_img in lines:
        pil_img = Image.fromarray(line_img).convert("RGB")
        pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values
        generated = model.generate(
            pixel_values,
            max_new_tokens=128,
            output_scores=True,
            return_dict_in_generate=True,
        )
        text = processor.batch_decode(
            generated.sequences, skip_special_tokens=True
        )[0]

        # Approximate confidence from mean token log-probs
        if generated.scores:
            import torch
            log_probs = [
                torch.nn.functional.log_softmax(s, dim=-1).max(dim=-1).values
                for s in generated.scores
            ]
            mean_log_prob = torch.stack(log_probs).mean().item()
            confidence = min(1.0, max(0.0, np.exp(mean_log_prob)))
        else:
            confidence = 0.0

        results.append({"text": text.strip(), "confidence": confidence})
    return results


# ---------------------------------------------------------------------------
# Line segmentation for TrOCR
# ---------------------------------------------------------------------------

def split_lines(img: np.ndarray, min_line_height: int = 15) -> list[np.ndarray]:
    """
    Split a multi-line region into individual line images using
    horizontal projection profile.

    Returns a list of cropped line images.  If only one line is
    detected, returns the original image in a list.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    # Invert so text pixels are white
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Horizontal projection
    h_proj = np.sum(binary, axis=1)
    threshold = h_proj.max() * 0.05

    in_line = False
    lines = []
    start = 0
    for y, val in enumerate(h_proj):
        if val > threshold and not in_line:
            start = y
            in_line = True
        elif val <= threshold and in_line:
            if y - start >= min_line_height:
                lines.append((start, y))
            in_line = False
    if in_line and len(h_proj) - start >= min_line_height:
        lines.append((start, len(h_proj)))

    if not lines:
        return [img]

    src = img if len(img.shape) == 3 else gray
    return [src[y1:y2, :] for y1, y2 in lines]


# ---------------------------------------------------------------------------
# Unified public API
# ---------------------------------------------------------------------------

class OCRResult:
    """Container for OCR output from a single region."""

    def __init__(self, text: str, confidence: float, engine: str,
                 script: str, bbox: tuple | None = None):
        self.text = text
        self.confidence = confidence
        self.engine = engine        # "easyocr" | "trocr"
        self.script = script        # "latin"  | "arabic"
        self.bbox = bbox            # ROI coordinates in original image

    def to_dict(self) -> dict:
        d = {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "engine": self.engine,
            "script": self.script,
        }
        if self.bbox is not None:
            d["bbox"] = self.bbox
        return d

    def __repr__(self) -> str:
        return (f"OCRResult(text={self.text!r}, conf={self.confidence:.2f}, "
                f"engine={self.engine}, script={self.script})")


# Confidence below this threshold → flag for human review
CONFIDENCE_THRESHOLD = 0.6


def ocr_region(
    img: np.ndarray,
    bbox: tuple[int, int, int, int] | None = None,
    script_hint: str | None = None,
) -> list[OCRResult]:
    """
    OCR a single image region.

    Args:
        img:         Full or cropped image (numpy BGR or RGB).
        bbox:        Optional (x, y, w, h) to crop from img.
        script_hint: Force "arabic" or "latin" instead of auto-detecting.

    Returns:
        List of OCRResult objects.
    """
    if bbox is not None:
        x, y, w, h = bbox
        crop = img[y:y + h, x:x + w]
    else:
        crop = img

    if crop.size == 0:
        return []

    script = script_hint or detect_script(crop)

    if script == "arabic":
        raw = _ocr_trocr(crop)
        engine = "trocr"
    else:
        raw = _ocr_easyocr(crop)
        engine = "easyocr"

    results = []
    for entry in raw:
        r = OCRResult(
            text=entry["text"],
            confidence=entry["confidence"],
            engine=engine,
            script=script,
            bbox=bbox,
        )
        if r.confidence < CONFIDENCE_THRESHOLD:
            logger.warning("Low confidence (%.2f) on text: %r", r.confidence, r.text)
        results.append(r)
    return results


def _bbox_overlap(det_bbox, region_bbox) -> float:
    """
    Compute fraction of detected bbox area that falls inside a template region.

    det_bbox:    list of 4 corner points [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    region_bbox: (rx, ry, rw, rh)
    """
    xs = [p[0] for p in det_bbox]
    ys = [p[1] for p in det_bbox]
    dx1, dy1, dx2, dy2 = min(xs), min(ys), max(xs), max(ys)
    rx, ry, rw, rh = region_bbox
    rx2, ry2 = rx + rw, ry + rh

    ix1 = max(dx1, rx)
    iy1 = max(dy1, ry)
    ix2 = min(dx2, rx2)
    iy2 = min(dy2, ry2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    det_area = max((dx2 - dx1) * (dy2 - dy1), 1)
    return inter / det_area


def ocr_full_page(
    img: np.ndarray,
    regions: dict[str, tuple[int, int, int, int]] | None = None,
    script_hints: dict[str, str] | None = None,
) -> dict[str, list[OCRResult]]:
    """
    OCR an entire page.

    Strategy: run EasyOCR once on the full page, then assign each detected
    text fragment to the template region it overlaps most with.  This avoids
    the problem where small cropped ROIs lose context and detection fails.

    For regions hinted as "arabic", the crop is passed to TrOCR instead.

    Args:
        img:          Full page image (RGB numpy array).
        regions:      Optional dict mapping field names to (x, y, w, h) ROIs.
                      If None, the whole page is processed as one region.
        script_hints: Optional dict mapping field names to "arabic" or "latin".

    Returns:
        Dict mapping field names (or "full_page") to lists of OCRResult.
    """
    script_hints = script_hints or {}

    if regions is None:
        return {"full_page": ocr_region(img, script_hint=script_hints.get("full_page"))}

    # --- Step 1: Full-page EasyOCR detection ---
    reader = _load_easyocr()
    detections = reader.readtext(
        img,
        detail=1,
        paragraph=False,
        text_threshold=0.3,
        low_text=0.2,
    )

    # --- Step 2: Assign detections to regions by overlap ---
    output: dict[str, list[OCRResult]] = {name: [] for name in regions}
    min_overlap = 0.3  # At least 30% of detected bbox must be inside region

    for det_bbox, text, conf in detections:
        best_field = None
        best_overlap = min_overlap
        for field_name, region_bbox in regions.items():
            overlap = _bbox_overlap(det_bbox, region_bbox)
            if overlap > best_overlap:
                best_overlap = overlap
                best_field = field_name
        if best_field is not None:
            r = OCRResult(
                text=text,
                confidence=float(conf),
                engine="easyocr",
                script="latin",
                bbox=regions[best_field],
            )
            if r.confidence < CONFIDENCE_THRESHOLD:
                logger.warning("Low confidence (%.2f) on text: %r",
                               r.confidence, r.text)
            output[best_field].append(r)

    # --- Step 3: For Arabic-hinted regions with no results, try TrOCR ---
    for field_name, hint in script_hints.items():
        if hint == "arabic" and field_name in regions:
            x, y, w, h = regions[field_name]
            crop = img[y:y + h, x:x + w]
            if crop.size == 0:
                continue
            raw = _ocr_trocr(crop)
            output[field_name] = [
                OCRResult(text=e["text"], confidence=e["confidence"],
                          engine="trocr", script="arabic",
                          bbox=regions[field_name])
                for e in raw
            ]

    return output
