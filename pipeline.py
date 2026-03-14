"""
End-to-end document OCR pipeline.

Usage:
    from pipeline import process_document
    results = process_document("scan.pdf", doc_type="check")
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
import logging
from datetime import datetime

import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from pathlib import Path

from ocr_engine import ocr_full_page, OCRResult, CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Ingestion
# ---------------------------------------------------------------------------

def load_pages(path: str, dpi: int = 300) -> list[np.ndarray]:
    """Load a PDF or image file and return a list of RGB numpy arrays."""
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".pdf":
        doc = fitz.open(path)
        pages = []
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            # Convert RGBA → RGB if needed
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            pages.append(img)
        doc.close()
        return pages

    if suffix in (".tif", ".tiff"):
        # Multi-page TIFF
        pil = Image.open(path)
        pages = []
        try:
            while True:
                pages.append(np.array(pil.convert("RGB")))
                pil.seek(pil.tell() + 1)
        except EOFError:
            pass
        return pages

    # Single image (jpg, png, bmp, webp, …)
    return [np.array(Image.open(path).convert("RGB"))]


# ---------------------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------------------

def preprocess(img: np.ndarray, aggressive: bool = False) -> np.ndarray:
    """
    Preprocess a page image for OCR.

    Args:
        img:        RGB numpy array.
        aggressive: If True, apply adaptive thresholding (for noisy scans).
                    If False, only deskew and lightly denoise (for clean images).

    Returns a single-channel (grayscale) cleaned image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # --- Deskew ---
    coords = np.column_stack(np.where(gray < 200))
    if len(coords) > 500:
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        if abs(angle) > 0.3:
            h, w = gray.shape
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            gray = cv2.warpAffine(
                gray, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )

    if aggressive:
        # For noisy/low-contrast scans
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31,
            C=15,
        )
        return cv2.fastNlMeansDenoising(binary, h=10)

    # For clean images: light CLAHE contrast boost, no binarization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return enhanced


# ---------------------------------------------------------------------------
# 3. Template definitions
# ---------------------------------------------------------------------------

# (x, y, w, h) — calibrate these on your actual scanned documents.
# Tip: scan a sample at 300 DPI, open in an image editor, note pixel coords.

CHECK_TEMPLATE = {
    "date":          (870,  200, 510, 180),
    "payee":         (370,  330, 680, 140),
    "amount_digits": (1030, 350, 350, 120),
    "amount_words":  (140,  430, 1150, 150),
    "memo":          (180,  570, 550, 120),
}

# Script hints let us skip auto-detection for known fields.
CHECK_SCRIPT_HINTS = {
    "date":          "latin",
    "amount_digits": "latin",
    # payee / amount_words / memo could be either — leave unset for auto-detect
}

LOAN_APP_TEMPLATE = {
    "applicant_name_en": (400, 250, 800, 70),
    "applicant_name_ar": (400, 340, 800, 70),
    "national_id":       (400, 430, 500, 60),
    "loan_amount":       (400, 520, 400, 60),
    "purpose":           (400, 700, 1200, 120),
}

LOAN_SCRIPT_HINTS = {
    "applicant_name_en": "latin",
    "applicant_name_ar": "arabic",
    "national_id":       "latin",
    "loan_amount":       "latin",
}

TEMPLATES = {
    "check":    (CHECK_TEMPLATE, CHECK_SCRIPT_HINTS),
    "loan_app": (LOAN_APP_TEMPLATE, LOAN_SCRIPT_HINTS),
}


# ---------------------------------------------------------------------------
# 3b. Layout-free cheque field detection
# ---------------------------------------------------------------------------

# Precompile patterns used by detect_fields_layout_free
_DATE_RE         = re.compile(r'\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}')
_AMOUNT_DIGIT_RE = re.compile(r'[\d\u0660-\u0669][,،\d\u0660-\u0669]*[.\u066B][\d\u0660-\u0669]{2}')
_MICR_RE         = re.compile(r'[\d:⑆⑇⑈⑉]{9,}')  # MICR digits + symbols
_ARABIC_RE       = re.compile(r'[\u0600-\u06FF]')


def detect_fields_layout_free(img: np.ndarray) -> dict[str, list[OCRResult]]:
    """
    Detect cheque fields without fixed pixel coordinates.

    Strategy:
      1. Run EasyOCR once on the full page.
      2. Classify each detected text blob using content patterns
         (date regex, amount regex, MICR, Arabic script) combined with
         its relative vertical position on the page.
      3. Everything unclassified is placed in "other".

    This works for cheques from any bank regardless of layout.
    """
    from ocr_engine import _load_easyocr  # lazy import — model already cached

    h, w = img.shape[:2]
    reader = _load_easyocr()
    detections = reader.readtext(
        img, detail=1, paragraph=False,
        text_threshold=0.3, low_text=0.2,
    )

    # Sort top-to-bottom by centroid y
    detections.sort(key=lambda d: sum(p[1] for p in d[0]) / 4)

    fields: dict[str, list[OCRResult]] = {
        "date": [], "payee": [], "amount_digits": [],
        "amount_words": [], "memo": [], "micr": [], "other": [],
    }
    assigned: set[int] = set()

    def centroid_y(bbox):
        return sum(p[1] for p in bbox) / 4

    def centroid_x(bbox):
        return sum(p[0] for p in bbox) / 4

    def make_result(bbox, text, conf, script="latin"):
        return OCRResult(
            text=text.strip(),
            confidence=float(conf),
            engine="easyocr",
            script=script,
            bbox=(int(min(p[0] for p in bbox)),
                  int(min(p[1] for p in bbox)),
                  int(max(p[0] for p in bbox) - min(p[0] for p in bbox)),
                  int(max(p[1] for p in bbox) - min(p[1] for p in bbox))),
        )

    # Pass 1 — MICR line: bottom 22% of page, long digit/symbol run
    for i, (bbox, text, conf) in enumerate(detections):
        if centroid_y(bbox) / h > 0.78 and _MICR_RE.search(text):
            fields["micr"].append(make_result(bbox, text, conf))
            assigned.add(i)

    # Pass 2 — Date: matches date pattern anywhere on page
    for i, (bbox, text, conf) in enumerate(detections):
        if i in assigned:
            continue
        if _DATE_RE.search(text):
            fields["date"].append(make_result(bbox, text, conf))
            assigned.add(i)

    # Pass 3 — Amount digits: currency amount pattern
    for i, (bbox, text, conf) in enumerate(detections):
        if i in assigned:
            continue
        if _AMOUNT_DIGIT_RE.search(text):
            fields["amount_digits"].append(make_result(bbox, text, conf))
            assigned.add(i)

    # Pass 4 — Amount in words: longest unassigned line in middle vertical band
    # (typically 25%-72% height, width spanning >35% of page)
    candidates = [
        (i, bbox, text, conf)
        for i, (bbox, text, conf) in enumerate(detections)
        if i not in assigned
        and 0.25 < centroid_y(bbox) / h < 0.72
        and len(text.strip()) > 6
    ]
    if candidates:
        # Prefer widest line (amount-in-words spans most of the cheque width)
        i, bbox, text, conf = max(candidates, key=lambda x: (
            max(p[0] for p in x[1]) - min(p[0] for p in x[1])
        ))
        script = "arabic" if _ARABIC_RE.search(text) else "latin"
        fields["amount_words"].append(make_result(bbox, text, conf, script))
        assigned.add(i)

    # Pass 5 — Payee: longest unassigned line in upper 65% of page
    upper = [
        (i, bbox, text, conf)
        for i, (bbox, text, conf) in enumerate(detections)
        if i not in assigned
        and centroid_y(bbox) / h < 0.65
        and len(text.strip()) > 2
    ]
    if upper:
        i, bbox, text, conf = max(upper, key=lambda x: len(x[2]))
        script = "arabic" if _ARABIC_RE.search(text) else "latin"
        fields["payee"].append(make_result(bbox, text, conf, script))
        assigned.add(i)

    # Pass 6 — Memo: short unassigned line in lower half, not MICR area
    for i, (bbox, text, conf) in enumerate(detections):
        if i in assigned:
            continue
        cy = centroid_y(bbox) / h
        if 0.55 < cy < 0.80 and 3 < len(text.strip()) < 60:
            fields["memo"].append(make_result(bbox, text, conf))
            assigned.add(i)

    # Pass 7 — everything else
    for i, (bbox, text, conf) in enumerate(detections):
        if i not in assigned:
            fields["other"].append(make_result(bbox, text, conf))

    return {k: v for k, v in fields.items() if v}


# ---------------------------------------------------------------------------
# 4. Post-processing helpers
# ---------------------------------------------------------------------------

def parse_amount(raw: str) -> float | None:
    cleaned = re.sub(r"[^\d.,]", "", raw)
    cleaned = cleaned.replace(",", "").replace(" ", "")
    # Handle OCR dropping the decimal: if no dot and len > 2, try inserting
    # before last 2 digits (common for currency)
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_date(raw: str) -> str | None:
    raw = raw.strip()
    if not raw:
        return None
    # Strip common OCR-merged prefixes like "Date", "Date:"
    raw = re.sub(r'^[Dd]ate[:\s]*', '', raw)
    # Clean OCR artifacts: {5 → 15, extra commas/dots
    raw = re.sub(r'[{(\[]', '1', raw)
    raw = re.sub(r'[})\]]', '', raw)
    raw = raw.replace(',.', ', ')
    raw = raw.strip(' ,.')
    for fmt in ("%B %d, %Y", "%b %d, %Y",
                "%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d", "%m-%d-%Y",
                "%d-%m-%Y", "%m/%d/%y", "%d/%m/%y"):
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    return raw  # Return cleaned raw string if no format matched


FIELD_PARSERS = {
    "amount_digits": parse_amount,
    "loan_amount":   parse_amount,
    "date":          parse_date,
}


# ---------------------------------------------------------------------------
# 5. Pipeline entry point
# ---------------------------------------------------------------------------

def process_document(
    path: str,
    doc_type: str = "check",
    aggressive_preprocess: bool = False,
) -> list[dict]:
    """
    Process a scanned document end-to-end.

    Args:
        path:     Path to a PDF or image file.
        doc_type: One of the keys in TEMPLATES, or "generic" for
                  full-page OCR without field mapping.
        aggressive_preprocess: Use adaptive thresholding for noisy scans.

    Returns:
        List of dicts (one per page), each containing:
          - "page": page number (1-based)
          - "fields": dict of field_name → {"value": ..., "confidence": ...,
                                             "engine": ..., "needs_review": bool}
    """
    pages = load_pages(path)
    all_results = []

    for page_num, page_img in enumerate(pages, start=1):
        if aggressive_preprocess:
            processed = preprocess(page_img, aggressive=True)
            processed_rgb = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
        else:
            processed_rgb = page_img

        if doc_type == "check_auto":
            raw_ocr = detect_fields_layout_free(processed_rgb)
        elif doc_type in TEMPLATES:
            regions, hints = TEMPLATES[doc_type]
            raw_ocr = ocr_full_page(processed_rgb, regions=regions,
                                    script_hints=hints)
        else:
            raw_ocr = ocr_full_page(processed_rgb)

        fields = {}
        for field_name, ocr_results in raw_ocr.items():
            # Concatenate all text fragments for the field
            texts = [r.text for r in ocr_results if r.text]
            combined_text = " ".join(texts)

            # Aggregate confidence (minimum across fragments)
            confidences = [r.confidence for r in ocr_results]
            min_conf = min(confidences) if confidences else 0.0

            # Determine which engine was used
            engines = {r.engine for r in ocr_results}
            engine = engines.pop() if len(engines) == 1 else "mixed"

            # Apply field-specific parser if available
            parser = FIELD_PARSERS.get(field_name)
            value = parser(combined_text) if parser else combined_text

            # Store bbox from first result (for auto-mode preview overlay)
            first_bbox = ocr_results[0].bbox if ocr_results else None

            fields[field_name] = {
                "value": value,
                "raw_text": combined_text,
                "confidence": round(min_conf, 4),
                "engine": engine,
                "needs_review": min_conf < CONFIDENCE_THRESHOLD,
                "bbox": first_bbox,
            }

        all_results.append({"page": page_num, "fields": fields})

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Bank document OCR pipeline")
    parser.add_argument("file", help="Path to PDF or image")
    parser.add_argument(
        "--type", default="check",
        choices=list(TEMPLATES.keys()) + ["generic"],
        help="Document type (default: check)",
    )
    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument("--aggressive", action="store_true",
                        help="Use aggressive preprocessing for noisy scans")
    args = parser.parse_args()

    results = process_document(args.file, doc_type=args.type,
                               aggressive_preprocess=args.aggressive)

    json_str = json.dumps(results, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(json_str, encoding="utf-8")
        print(f"Results written to {args.output}")
    else:
        print(json_str)
