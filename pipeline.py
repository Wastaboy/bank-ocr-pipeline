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
_AMOUNT_DIGIT_RE = re.compile(
    r'(?:[\d\u0660-\u0669]{1,3}(?:[,،][\d\u0660-\u0669]{3})+)'   # 1,234 or 1,234,567
    r'(?:[./\u066B][\d\u0660-\u0669]{1,3})?'                      # optional decimal (1-3 places)
    r'|[\d\u0660-\u0669]{4,}'                                      # OR 4+ bare digits (e.g. 5250)
    r'|[\d\u0660-\u0669]{1,3}[./\u066B][\d\u0660-\u0669]{1,3}'   # OR decimal without comma (e.g. 50.000)
)
_MICR_RE         = re.compile(r'[\d:⑆⑇⑈⑉]{9,}')  # MICR digits + symbols
_ARABIC_RE       = re.compile(r'[\u0600-\u06FF]')
# Words that appear in amount-in-words lines (English and Arabic)
_AMOUNT_WORDS_RE = re.compile(
    r'\b(?:thousand|hundred|million|billion|dinar[s]?|fils|riyal[s]?|only|halala|baisa|dirham[s]?|pound[s]?|'
    r'and\s+\w+\s+hundred|'
    r'ألف|مائة|دينار|فلس|ريال|فقط)\b',
    re.IGNORECASE,
)
# Printed labels on cheque forms that should never be extracted as fields
_PRINTED_LABELS_RE = re.compile(
    r'^(?:authorized\s+signature|memo|signature|for\s+and\s+on\s+behalf|'
    r'signatory|sign\s+here|pay\s+to\s+the\s+order|date|amount)\s*[:\-]?\s*$',
    re.IGNORECASE,
)


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

    def _best_amount_match(text: str) -> str | None:
        """Extract the most currency-like numeric match from a text blob."""
        matches = list(_AMOUNT_DIGIT_RE.finditer(text))
        if not matches:
            return None
        # Prefer: comma-formatted > has decimal > longer
        def score(m):
            s = m.group(0)
            return ((',' in s or '،' in s), ('.' in s or '/' in s), len(s))
        return max(matches, key=score).group(0)

    # Pass 3 — Amount digits: collect all right-half candidates then keep only the best one
    amount_digit_candidates = []
    for i, (bbox, text, conf) in enumerate(detections):
        if i in assigned:
            continue
        numeric = _best_amount_match(text)
        if numeric and centroid_x(bbox) / w > 0.40:
            amount_digit_candidates.append((i, bbox, numeric, conf))

    if amount_digit_candidates:
        def _digit_score(c):
            s = c[2]
            return ((',' in s or '،' in s), ('.' in s or '/' in s), len(s))
        best = max(amount_digit_candidates, key=_digit_score)
        fields["amount_digits"].append(make_result(best[1], best[2], best[3]))
        assigned.add(best[0])

    # Pass 3b — Positional fallback: if still no amount_digits found,
    # scan the right half for the most numeric-looking token
    if not fields["amount_digits"]:
        candidates = [
            (i, bbox, text, conf)
            for i, (bbox, text, conf) in enumerate(detections)
            if i not in assigned
            and centroid_x(bbox) / w > 0.45
            and centroid_y(bbox) / h < 0.70
            and sum(ch.isdigit() or '\u0660' <= ch <= '\u0669' for ch in text) >= 2
        ]
        if candidates:
            best = max(candidates, key=lambda x: sum(
                ch.isdigit() or '\u0660' <= ch <= '\u0669' for ch in x[2]
            ) / max(len(x[2]), 1))
            i, bbox, text, conf = best
            numeric = _best_amount_match(text) or text
            fields["amount_digits"].append(make_result(bbox, numeric, conf))
            assigned.add(i)

    # Pass 4 — Amount in words: any unassigned line containing currency words.
    # Grabs all matching lines to handle multi-line amounts
    # (e.g. "Sixty Seven Thousand..." on one line and "and Seven Hundred Fifty Fils Only" on the next)
    amount_word_lines = [
        (i, bbox, text, conf)
        for i, (bbox, text, conf) in enumerate(detections)
        if i not in assigned
        and 0.15 < centroid_y(bbox) / h < 0.82
        and _AMOUNT_WORDS_RE.search(text)
    ]
    if amount_word_lines:
        for i, bbox, text, conf in amount_word_lines:
            script = "arabic" if _ARABIC_RE.search(text) else "latin"
            fields["amount_words"].append(make_result(bbox, text, conf, script))
            assigned.add(i)
    else:
        # Fallback: widest unassigned line in the middle band
        candidates = [
            (i, bbox, text, conf)
            for i, (bbox, text, conf) in enumerate(detections)
            if i not in assigned
            and 0.25 < centroid_y(bbox) / h < 0.72
            and len(text.strip()) > 6
        ]
        if candidates:
            i, bbox, text, conf = max(candidates, key=lambda x: (
                max(p[0] for p in x[1]) - min(p[0] for p in x[1])
            ))
            script = "arabic" if _ARABIC_RE.search(text) else "latin"
            fields["amount_words"].append(make_result(bbox, text, conf, script))
            assigned.add(i)

    # Pass 5 — Payee: longest unassigned line in upper 65% that does NOT
    # look like amount-in-words, a bank name header, or a printed label
    _BANK_HEADER_RE = re.compile(
        r'\b(?:bank|financial|harbour|harbor|branch|swift|iban|bhd|kwd|sar|aed|usd)\b',
        re.IGNORECASE,
    )
    upper = [
        (i, bbox, text, conf)
        for i, (bbox, text, conf) in enumerate(detections)
        if i not in assigned
        and 0.15 < centroid_y(bbox) / h < 0.65
        and len(text.strip()) > 2
        and not _AMOUNT_WORDS_RE.search(text)
        and not _BANK_HEADER_RE.search(text)
        and not _PRINTED_LABELS_RE.match(text.strip())
    ]
    if upper:
        i, bbox, text, conf = max(upper, key=lambda x: len(x[2]))
        script = "arabic" if _ARABIC_RE.search(text) else "latin"
        fields["payee"].append(make_result(bbox, text, conf, script))
        assigned.add(i)

    _ACCOUNT_RE = re.compile(
        r'\b(?:account|acc\.?|a/c)\b.*\d|\d{4}[\-\s]\d{4}[\-\s]\d{2,}',
        re.IGNORECASE,
    )

    # Pass 6 — Memo: short unassigned line in lower half, not MICR area,
    # and not a printed label or account number
    for i, (bbox, text, conf) in enumerate(detections):
        if i in assigned:
            continue
        cy = centroid_y(bbox) / h
        if (0.55 < cy < 0.80
                and 3 < len(text.strip()) < 60
                and not _PRINTED_LABELS_RE.match(text.strip())
                and not _AMOUNT_WORDS_RE.search(text)
                and not _ACCOUNT_RE.search(text)):
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
    # Convert Arabic-Indic numerals (٠١٢٣٤٥٦٧٨٩) to Western digits
    raw = raw.translate(str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789'))
    # Normalise Arabic decimal separator and OCR slash misread
    raw = raw.replace('\u066B', '.').replace('/', '.')
    # Strip currency symbols and whitespace
    cleaned = re.sub(r'[^\d.,]', '', raw)
    # Remove thousands separators (commas)
    # If multiple dots exist, keep only the last one as the decimal separator
    parts = cleaned.split('.')
    if len(parts) > 2:
        cleaned = ''.join(parts[:-1]) + '.' + parts[-1]
    cleaned = cleaned.replace(',', '')
    if not cleaned:
        return None
    try:
        value = float(cleaned)
        # Return as integer if no meaningful decimal part
        return int(value) if value == int(value) else value
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


_ONES = {
    'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,
    'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12,'thirteen':13,
    'fourteen':14,'fifteen':15,'sixteen':16,'seventeen':17,'eighteen':18,
    'nineteen':19,'twenty':20,'thirty':30,'forty':40,'fifty':50,
    'sixty':60,'seventy':70,'eighty':80,'ninety':90,
}


def words_to_amount(text: str) -> float | None:
    """
    Convert an English amount-in-words string to a float.

    Handles Gulf currency structure: dinars + fils (1000 fils = 1 dinar).
    Also handles standard cents/halalas (100 subunits = 1 major unit).

    Examples:
      "Sixty Seven Thousand Eight Hundred Ninety Dinars and Seven Hundred Fifty Fils Only"
        → 67890.750
      "Five Thousand Two Hundred Fifty Only" → 5250.0
    """
    text = text.lower()
    # Split on the subunit keyword to separate major and minor parts
    subunit_split = re.split(r'\b(?:fils?|halala[s]?|cent[s]?|baisa)\b', text, maxsplit=1)
    major_text = re.sub(r'\b(?:dinar[s]?|riyal[s]?|pound[s]?|dollar[s]?|bhd|kwd|sar|aed|usd|only|and)\b', ' ', subunit_split[0])
    minor_text = re.sub(r'\b(?:only|and)\b', ' ', subunit_split[1]) if len(subunit_split) > 1 else ''

    def _parse_chunk(words: str) -> int:
        tokens = re.findall(r'[a-z]+', words)
        total = 0
        current = 0
        for tok in tokens:
            if tok in _ONES:
                current += _ONES[tok]
            elif tok == 'hundred':
                current = current * 100 if current else 100
            elif tok in ('thousand', 'thousands'):
                total += (current or 1) * 1000
                current = 0
            elif tok in ('million', 'millions'):
                total += (current or 1) * 1_000_000
                current = 0
        return total + current

    major = _parse_chunk(major_text)
    minor = _parse_chunk(minor_text) if minor_text.strip() else 0

    if major == 0 and minor == 0:
        return None

    # Determine subunit divisor: fils/baisa → /1000, cents/halala → /100
    divisor = 1000 if re.search(r'\b(?:fils?|baisa)\b', text) else 100
    return major + minor / divisor


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

        # Cross-validate amount_digits against amount_words
        if "amount_digits" in fields and "amount_words" in fields:
            words_val = words_to_amount(fields["amount_words"]["raw_text"])
            digit_val  = fields["amount_digits"]["value"]
            if words_val is not None:
                if digit_val is None:
                    # Words parsed but digits failed — use words-derived value
                    fields["amount_digits"]["value"] = words_val
                    fields["amount_digits"]["needs_review"] = False
                else:
                    # Both present — check if they agree within 1%
                    try:
                        pct_diff = abs(float(digit_val) - words_val) / max(words_val, 1)
                        if pct_diff > 0.01:
                            # Mismatch: trust amount_words (harder to OCR-corrupt a sentence)
                            fields["amount_digits"]["value"] = words_val
                            fields["amount_digits"]["needs_review"] = True
                            fields["amount_digits"]["amount_words_override"] = True
                        else:
                            # They agree — use words_val as the canonical value (more precise)
                            fields["amount_digits"]["value"] = words_val
                    except (TypeError, ValueError):
                        pass

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
