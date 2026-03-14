"""
Tkinter GUI for the bank document OCR pipeline.

Features:
  - Upload image (PNG/JPG/BMP/TIFF) or PDF files
  - Select document type (check, loan_app, generic)
  - Toggle aggressive preprocessing
  - Display uploaded image with ROI bounding boxes overlaid
  - Show extracted fields in a results table
  - Export results to JSON
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk

from pipeline import process_document, load_pages, TEMPLATES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_FILETYPES = [
    ("All supported", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp *.pdf"),
    ("Images", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp"),
    ("PDF", "*.pdf"),
]

COLORS = {
    "date":          (255, 0,   0),
    "payee":         (0,   180, 0),
    "amount_digits": (0,   0,   255),
    "amount_words":  (200, 100, 0),
    "memo":          (180, 0,   180),
}

DEFAULT_COLOR = (0, 200, 200)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class OCRApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Bank Document OCR Pipeline")
        self.geometry("1100x750")
        self.minsize(900, 600)
        self.configure(bg="#f0f0f0")

        self._file_path: str | None = None
        self._pages: list[np.ndarray] = []
        self._current_page: int = 0
        self._results: list[dict] | None = None
        self._photo_ref: ImageTk.PhotoImage | None = None  # prevent GC

        self._build_ui()

    # ----- UI construction ---------------------------------------------------

    def _build_ui(self):
        # Top control bar
        ctrl = ttk.Frame(self, padding=8)
        ctrl.pack(fill=tk.X)

        ttk.Button(ctrl, text="Open File…", command=self._on_open).pack(side=tk.LEFT)

        ttk.Label(ctrl, text="  Doc type:").pack(side=tk.LEFT)
        self._doc_type_var = tk.StringVar(value="check_auto")
        doc_combo = ttk.Combobox(
            ctrl, textvariable=self._doc_type_var, width=14, state="readonly",
            values=["check_auto"] + list(TEMPLATES.keys()) + ["generic"],
        )
        doc_combo.pack(side=tk.LEFT, padx=(4, 8))

        self._aggressive_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            ctrl, text="Aggressive preprocessing", variable=self._aggressive_var,
        ).pack(side=tk.LEFT, padx=(0, 12))

        self._run_btn = ttk.Button(ctrl, text="Run OCR", command=self._on_run,
                                   state=tk.DISABLED)
        self._run_btn.pack(side=tk.LEFT, padx=(0, 8))

        self._export_btn = ttk.Button(ctrl, text="Export JSON…",
                                      command=self._on_export, state=tk.DISABLED)
        self._export_btn.pack(side=tk.LEFT)

        self._status_var = tk.StringVar(value="No file loaded.")
        ttk.Label(ctrl, textvariable=self._status_var, foreground="gray").pack(
            side=tk.RIGHT)

        # Page navigation (for multi-page docs)
        nav = ttk.Frame(self, padding=(8, 0))
        nav.pack(fill=tk.X)

        self._prev_btn = ttk.Button(nav, text="◀ Prev", command=self._prev_page,
                                    state=tk.DISABLED)
        self._prev_btn.pack(side=tk.LEFT)

        self._page_label_var = tk.StringVar(value="")
        ttk.Label(nav, textvariable=self._page_label_var).pack(side=tk.LEFT, padx=8)

        self._next_btn = ttk.Button(nav, text="Next ▶", command=self._next_page,
                                    state=tk.DISABLED)
        self._next_btn.pack(side=tk.LEFT)

        # Main paned area: image left, results right
        pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Left: image canvas with scrollbar
        img_frame = ttk.LabelFrame(pane, text="Document Preview", padding=4)
        pane.add(img_frame, weight=3)

        self._canvas = tk.Canvas(img_frame, bg="#e0e0e0", cursor="crosshair")
        self._canvas.pack(fill=tk.BOTH, expand=True)

        # Right: results panel (scrollable field cards with copy buttons)
        result_frame = ttk.LabelFrame(pane, text="Extracted Fields", padding=4)
        pane.add(result_frame, weight=2)

        # Copy-all button at the top
        top_bar = ttk.Frame(result_frame)
        top_bar.pack(fill=tk.X, pady=(0, 4))
        self._copy_all_btn = ttk.Button(top_bar, text="Copy All Fields",
                                        command=self._on_copy_all,
                                        state=tk.DISABLED)
        self._copy_all_btn.pack(side=tk.RIGHT)

        # Scrollable area for field rows
        scroll_canvas = tk.Canvas(result_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL,
                                  command=scroll_canvas.yview)
        self._results_inner = ttk.Frame(scroll_canvas)

        self._results_inner.bind(
            "<Configure>",
            lambda e: scroll_canvas.configure(
                scrollregion=scroll_canvas.bbox("all")),
        )
        scroll_canvas.create_window((0, 0), window=self._results_inner,
                                    anchor=tk.NW)
        scroll_canvas.configure(yscrollcommand=scrollbar.set)

        scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._scroll_canvas = scroll_canvas

        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            scroll_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        scroll_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Legend
        legend_frame = ttk.LabelFrame(result_frame, text="ROI Legend", padding=4)
        legend_frame.pack(fill=tk.X, pady=(6, 0))
        self._legend_frame = legend_frame

        # Track current page fields for copy-all
        self._current_fields: dict = {}

    # ----- File handling -----------------------------------------------------

    def _on_open(self):
        path = filedialog.askopenfilename(filetypes=SUPPORTED_FILETYPES)
        if not path:
            return
        self._file_path = path
        self._results = None
        self._export_btn.config(state=tk.DISABLED)
        self._status_var.set("Loading…")
        self.update_idletasks()

        try:
            self._pages = load_pages(path)
        except Exception as e:
            messagebox.showerror("Load Error", str(e))
            self._status_var.set("Load failed.")
            return

        self._current_page = 0
        self._run_btn.config(state=tk.NORMAL)
        self._update_page_nav()
        self._show_page(draw_rois=True, results=None)
        self._clear_results_table()
        self._status_var.set(f"Loaded: {Path(path).name}  "
                             f"({len(self._pages)} page(s))")

    # ----- Page navigation ---------------------------------------------------

    def _update_page_nav(self):
        n = len(self._pages)
        self._page_label_var.set(f"Page {self._current_page + 1} / {n}")
        self._prev_btn.config(state=tk.NORMAL if self._current_page > 0
                              else tk.DISABLED)
        self._next_btn.config(state=tk.NORMAL if self._current_page < n - 1
                              else tk.DISABLED)

    def _prev_page(self):
        if self._current_page > 0:
            self._current_page -= 1
            self._update_page_nav()
            page_results = self._results[self._current_page] if self._results else None
            self._show_page(draw_rois=True, results=page_results)
            if page_results:
                self._populate_results_table(page_results)

    def _next_page(self):
        if self._current_page < len(self._pages) - 1:
            self._current_page += 1
            self._update_page_nav()
            page_results = self._results[self._current_page] if self._results else None
            self._show_page(draw_rois=True, results=page_results)
            if page_results:
                self._populate_results_table(page_results)

    # ----- Image display -----------------------------------------------------

    def _show_page(self, draw_rois: bool = True, results: dict | None = None):
        img = self._pages[self._current_page].copy()
        doc_type = self._doc_type_var.get()

        # Draw ROI boxes
        if draw_rois and doc_type == "check_auto" and results:
            self._draw_auto_rois(img, results)
            self._clear_legend()
        elif draw_rois and doc_type in TEMPLATES:
            regions, _ = TEMPLATES[doc_type]
            self._draw_rois(img, regions, results)
            self._draw_legend(regions)
        else:
            self._clear_legend()

        # Fit to canvas
        self._canvas.update_idletasks()
        cw = max(self._canvas.winfo_width(), 400)
        ch = max(self._canvas.winfo_height(), 300)
        h, w = img.shape[:2]
        scale = min(cw / w, ch / h, 1.0)
        if scale < 1.0:
            display = cv2.resize(img, (int(w * scale), int(h * scale)),
                                 interpolation=cv2.INTER_AREA)
        else:
            display = img

        pil_img = Image.fromarray(display)
        self._photo_ref = ImageTk.PhotoImage(pil_img)
        self._canvas.delete("all")
        self._canvas.create_image(cw // 2, ch // 2, image=self._photo_ref,
                                  anchor=tk.CENTER)

    def _draw_rois(self, img: np.ndarray,
                   regions: dict[str, tuple],
                   results: dict | None):
        for name, (x, y, w, h) in regions.items():
            color = COLORS.get(name, DEFAULT_COLOR)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

            # Label
            label = name.replace("_", " ").title()
            if results and name in results.get("fields", {}):
                conf = results["fields"][name]["confidence"]
                label += f" ({conf:.0%})"
            cv2.putText(img, label, (x + 4, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                        cv2.LINE_AA)

    def _draw_auto_rois(self, img: np.ndarray, results: dict):
        """Draw bounding boxes around auto-detected field regions."""
        fields = results.get("fields", {})
        for name, info in fields.items():
            bbox = info.get("bbox")
            if not bbox:
                continue
            color = COLORS.get(name, DEFAULT_COLOR)
            x, y, w, h = bbox
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            label = name.replace("_", " ").title()
            conf = info.get("confidence", 0)
            label += f" ({conf:.0%})"
            cv2.putText(img, label, (x + 4, max(y - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    def _draw_legend(self, regions: dict):
        self._clear_legend()
        for name in regions:
            color = COLORS.get(name, DEFAULT_COLOR)
            hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
            label = name.replace("_", " ").title()
            frame = ttk.Frame(self._legend_frame)
            frame.pack(anchor=tk.W, pady=1)
            swatch = tk.Canvas(frame, width=14, height=14,
                               highlightthickness=0)
            swatch.create_rectangle(1, 1, 13, 13, fill=hex_color, outline="")
            swatch.pack(side=tk.LEFT, padx=(0, 4))
            ttk.Label(frame, text=label, font=("TkDefaultFont", 8)).pack(
                side=tk.LEFT)

    def _clear_legend(self):
        for child in self._legend_frame.winfo_children():
            child.destroy()

    # ----- OCR execution -----------------------------------------------------

    def _on_run(self):
        if not self._file_path:
            return
        self._run_btn.config(state=tk.DISABLED)
        self._status_var.set("Running OCR… (this may take a moment)")
        self.update_idletasks()

        # Run in background thread to keep UI responsive
        thread = threading.Thread(target=self._run_ocr, daemon=True)
        thread.start()

    def _run_ocr(self):
        try:
            results = process_document(
                self._file_path,
                doc_type=self._doc_type_var.get(),
                aggressive_preprocess=self._aggressive_var.get(),
            )
            self.after(0, self._on_ocr_done, results, None)
        except Exception as e:
            self.after(0, self._on_ocr_done, None, e)

    def _on_ocr_done(self, results: list[dict] | None, error: Exception | None):
        self._run_btn.config(state=tk.NORMAL)
        if error:
            messagebox.showerror("OCR Error", str(error))
            self._status_var.set("OCR failed.")
            return

        self._results = results
        self._export_btn.config(state=tk.NORMAL)

        page_result = results[self._current_page]
        self._show_page(draw_rois=True, results=page_result)
        self._populate_results_table(page_result)

        total_fields = sum(len(r["fields"]) for r in results)
        review_count = sum(
            1 for r in results
            for f in r["fields"].values()
            if f.get("needs_review")
        )
        self._status_var.set(
            f"Done — {total_fields} fields extracted, "
            f"{review_count} flagged for review."
        )

    # ----- Results panel -----------------------------------------------------

    def _clear_results_table(self):
        for child in self._results_inner.winfo_children():
            child.destroy()
        self._current_fields = {}
        self._copy_all_btn.config(state=tk.DISABLED)

    def _populate_results_table(self, page_result: dict):
        self._clear_results_table()
        fields = page_result.get("fields", {})
        self._current_fields = fields
        self._copy_all_btn.config(state=tk.NORMAL if fields else tk.DISABLED)

        for name, info in fields.items():
            display_name = name.replace("_", " ").title()
            value = info.get("value", "")
            if value is None:
                value = "(none)"
            raw = info.get("raw_text", "")
            conf = info.get("confidence", 0)
            engine = info.get("engine", "?")
            needs_review = info.get("needs_review", False)

            bg = "#fff3cd" if needs_review else "#d4edda"
            card = tk.Frame(self._results_inner, bg=bg, bd=1, relief=tk.GROOVE)
            card.pack(fill=tk.X, padx=2, pady=2)

            # Row 1: field name + confidence + engine
            header = tk.Frame(card, bg=bg)
            header.pack(fill=tk.X, padx=6, pady=(4, 0))
            tk.Label(header, text=display_name, bg=bg,
                     font=("TkDefaultFont", 9, "bold")).pack(side=tk.LEFT)
            meta_text = f"{conf:.0%}  |  {engine}"
            if needs_review:
                meta_text += "  |  REVIEW"
            tk.Label(header, text=meta_text, bg=bg, fg="#666",
                     font=("TkDefaultFont", 8)).pack(side=tk.RIGHT)

            # Row 2: value + copy button
            body = tk.Frame(card, bg=bg)
            body.pack(fill=tk.X, padx=6, pady=(2, 4))

            val_str = str(value)
            val_label = tk.Label(body, text=val_str, bg=bg, anchor=tk.W,
                                 wraplength=280, justify=tk.LEFT,
                                 font=("TkDefaultFont", 10))
            val_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

            copy_btn = ttk.Button(
                body, text="Copy", width=5,
                command=lambda v=val_str, b=None: self._copy_field(v, b),
            )
            copy_btn.pack(side=tk.RIGHT, padx=(4, 0))
            # Rebind so the button reference is captured for feedback
            copy_btn.config(
                command=lambda v=val_str, btn=copy_btn: self._copy_field(v, btn),
            )

            # Row 3: raw OCR text if different from parsed value
            if str(value) != raw and raw:
                raw_label = tk.Label(body, text=f"raw: {raw}", bg=bg, fg="#888",
                                     anchor=tk.W, wraplength=280, justify=tk.LEFT,
                                     font=("TkDefaultFont", 8))
                raw_label.pack(side=tk.LEFT, fill=tk.X, pady=(0, 2))

    def _copy_field(self, value: str, btn: ttk.Button | None):
        """Copy a single field value to clipboard and show feedback."""
        self.clipboard_clear()
        self.clipboard_append(value)
        self.update()  # Required for clipboard to persist on Windows
        if btn:
            original_text = btn.cget("text")
            btn.config(text="Done!")
            self.after(1200, lambda: btn.config(text=original_text))
        self._status_var.set(f"Copied: {value}")

    def _on_copy_all(self):
        """Copy all fields as formatted text."""
        if not self._current_fields:
            return
        lines = []
        for name, info in self._current_fields.items():
            display_name = name.replace("_", " ").title()
            value = info.get("value", "")
            if value is None:
                value = "(none)"
            lines.append(f"{display_name}: {value}")
        text = "\n".join(lines)
        self.clipboard_clear()
        self.clipboard_append(text)
        self.update()
        self._copy_all_btn.config(text="Copied!")
        self.after(1200, lambda: self._copy_all_btn.config(text="Copy All Fields"))
        self._status_var.set("All fields copied to clipboard.")

    # ----- Export ------------------------------------------------------------

    def _on_export(self):
        if not self._results:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialfile="ocr_results.json",
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._results, f, indent=2, ensure_ascii=False)
        self._status_var.set(f"Exported to {Path(path).name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = OCRApp()
    app.mainloop()
