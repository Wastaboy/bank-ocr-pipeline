# Bank OCR Pipeline

A desktop application for extracting structured data from scanned bank cheques and documents using OCR. Supports both English (Latin) and Arabic script, works with any cheque layout, and exports results to JSON.

---

## Features

- **Layout-free cheque detection** — automatically locates fields (date, payee, amount, memo) without needing fixed pixel coordinates, works across different bank formats
- **Template mode** — define exact pixel regions for known document layouts (cheques, loan applications)
- **Dual OCR engine** — EasyOCR for Latin/English text, TrOCR for Arabic handwriting, with automatic script detection per field
- **Image preprocessing** — deskewing, CLAHE contrast enhancement, and optional adaptive thresholding for noisy scans
- **GUI** — built with Tkinter; shows the document with color-coded bounding boxes overlaid on each detected field
- **Multi-page support** — handles PDFs and multi-page TIFFs with page navigation
- **Confidence scoring** — every extracted field includes a confidence score; low-confidence fields are flagged for human review
- **Export** — save all results to a JSON file
- **CLI** — can also be run headlessly from the command line

---

## Project Structure

```
ocr_pipeline/
├── gui.py            # Tkinter desktop application
├── pipeline.py       # End-to-end document processing logic
├── ocr_engine.py     # OCR engine wrappers (EasyOCR + TrOCR)
└── requirements.txt  # Python dependencies
```

---

## How It Works

### 1. Ingestion (`pipeline.py → load_pages`)
Loads the input file into a list of RGB numpy arrays. Supports PDF (via PyMuPDF), multi-page TIFF, and standard image formats (PNG, JPG, BMP, WebP).

### 2. Preprocessing (`pipeline.py → preprocess`)
- **Deskew** — detects rotation angle using `minAreaRect` and corrects it
- **Normal mode** — applies CLAHE contrast enhancement (good for clean scans)
- **Aggressive mode** — applies adaptive Gaussian thresholding + denoising (good for low-quality or faded scans)

### 3. Field Detection (two modes)

#### Auto mode (`check_auto`)
Runs EasyOCR once on the full page and classifies each detected text blob by content pattern and vertical position:

| Pass | Field | Detection method |
|------|-------|-----------------|
| 1 | MICR line | Bottom 22% of page + MICR character regex |
| 2 | Date | Date regex (`dd/mm/yyyy`, `mm-dd-yyyy`, etc.) |
| 3 | Amount (digits) | Currency amount regex (handles Arabic-Indic numerals) |
| 4 | Amount (words) | Widest unassigned text line in the middle vertical band |
| 5 | Payee | Longest unassigned line in the upper 65% of the page |
| 6 | Memo | Short unassigned line in the lower half |

#### Template mode
Defines named regions as `(x, y, w, h)` pixel coordinates. EasyOCR runs on the full page and each detected text blob is assigned to the region it overlaps with most (minimum 30% overlap). Arabic-hinted regions fall back to TrOCR if EasyOCR finds nothing.

Two built-in templates are included:
- `check` — standard cheque layout
- `loan_app` — loan application form with both English and Arabic name fields

### 4. OCR Engines (`ocr_engine.py`)

| Engine | Script | Model |
|--------|--------|-------|
| EasyOCR | Latin / English | `easyocr` (English pack) |
| TrOCR | Arabic handwriting | `microsoft/trocr-base-handwritten` |

Script is auto-detected per region using a fast bilingual EasyOCR pass that counts Arabic vs Latin characters. Both models are loaded once and cached.

TrOCR confidence is approximated from the mean token log-probability of the generated sequence.

### 5. Post-processing
- **Date parsing** — normalises raw OCR output across multiple date formats, strips common OCR artifacts (`{` → `1`, etc.)
- **Amount parsing** — strips non-numeric characters, handles comma separators
- **Review flagging** — any field with confidence below 0.60 is marked `needs_review: true`

---

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python 3.10+
- `easyocr >= 1.7`
- `pymupdf >= 1.23`
- `opencv-python-headless >= 4.8`
- `pillow >= 10.0`
- `transformers >= 4.36`
- `torch >= 2.1`
- `sentencepiece >= 0.1.99`

GPU is used automatically if available (recommended for TrOCR).

---

## Usage

### Desktop GUI

```bash
python gui.py
```

1. Click **Open File** and select a cheque image or PDF
2. Choose a document type:
   - `check_auto` — layout-free detection (recommended for mixed bank formats)
   - `check` — fixed-region template
   - `loan_app` — loan application template
   - `generic` — full-page OCR with no field mapping
3. Optionally enable **Aggressive preprocessing** for noisy or faded scans
4. Click **Run OCR**
5. Fields appear in the right panel with confidence scores. Fields flagged for review are highlighted in yellow.
6. Click **Copy** next to any field, or **Copy All Fields** to copy everything at once
7. Click **Export JSON** to save results

### Command Line

```bash
python pipeline.py scan.pdf --type check --output results.json
python pipeline.py cheque.jpg --type check_auto
python pipeline.py form.pdf --type loan_app --aggressive
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `file` | Path to image or PDF |
| `--type` | Document type: `check`, `loan_app`, `generic` |
| `--output` / `-o` | Output JSON file path (prints to stdout if omitted) |
| `--aggressive` | Enable aggressive preprocessing |

---

## Output Format

```json
[
  {
    "page": 1,
    "fields": {
      "date": {
        "value": "2024-06-15",
        "raw_text": "15/06/2024",
        "confidence": 0.94,
        "engine": "easyocr",
        "needs_review": false,
        "bbox": [870, 200, 510, 180]
      },
      "payee": {
        "value": "Mohammed Al-Rashid",
        "raw_text": "Mohammed Al-Rashid",
        "confidence": 0.87,
        "engine": "easyocr",
        "needs_review": false,
        "bbox": [370, 330, 680, 140]
      },
      "amount_digits": {
        "value": 5250.0,
        "raw_text": "5,250.00",
        "confidence": 0.91,
        "engine": "easyocr",
        "needs_review": false,
        "bbox": [1030, 350, 350, 120]
      }
    }
  }
]
```

---

## Adding a Custom Template

To support a new document layout, add an entry to `TEMPLATES` in `pipeline.py`:

```python
MY_TEMPLATE = {
    "field_name": (x, y, width, height),   # pixel coordinates at 300 DPI
    ...
}

MY_SCRIPT_HINTS = {
    "field_name": "arabic",   # or "latin" — omit for auto-detect
}

TEMPLATES["my_doc"] = (MY_TEMPLATE, MY_SCRIPT_HINTS)
```

Tip: open a sample scan in an image editor, note the pixel coordinates of each field at 300 DPI, and use those values.
