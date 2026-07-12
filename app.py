"""
app.py
======

A minimal Flask API + black-and-white web page for **Crispen**.
This deliberately replaces the old Streamlit interface with a plain
HTTP API so the engine (crispen_engine.py) can be driven from any client.

Endpoints
---------
GET  /            -> serves the monochrome HTML UI (templates/index.html)
POST /api/filter  -> multipart form upload (field name: "image", one or many)
                     returns JSON:
                       {
                         "ok": true,
                         "results": [
                           {
                             "ok": true,
                             "filename": "scan1.png",
                             "category": "white_document" | "colored_product",
                             "original": "data:image/png;base64,...",
                             "result":   "data:image/png;base64,..."
                           },
                           ...
                         ]
                       }
                     Each result has its own "ok"; a bad file reports
                     {"ok": false, "filename": ..., "error": ...}.
GET  /api/health  -> {"ok": true}

Run
---
    pip install flask opencv-python numpy pillow
    python app.py
    # open http://127.0.0.1:5000
"""

import base64
import io

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image

from crispen_engine import CrispenEngine, pil_to_bgr, bgr_to_pil

app = Flask(__name__)

# Reject uploads larger than 25 MB.
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'webp'}


def _has_allowed_extension(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _bgr_to_data_uri(bgr_image):
    """Encode a BGR OpenCV image as a base64 PNG data URI for the browser."""
    pil_image = bgr_to_pil(bgr_image)
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
    return f"data:image/png;base64,{encoded}"


@app.route('/')
def index():
    """Serve the black-and-white single-page UI."""
    return render_template('index.html')


@app.route('/api/health')
def health():
    return jsonify({'ok': True})


def _filter_one_upload(upload):
    """
    Filter a single uploaded file object.

    Returns a per-image result dict. `ok` is False for that image when the
    file is unusable, so one bad file never fails the whole batch.
    """
    filename = upload.filename or 'image'

    if filename == '':
        return {'ok': False, 'filename': filename, 'error': 'No file selected.'}

    if not _has_allowed_extension(filename):
        return {
            'ok': False,
            'filename': filename,
            'error': f'Unsupported file type. Allowed: {", ".join(sorted(ALLOWED_EXTENSIONS))}.'
        }

    # Decode the upload into an OpenCV BGR image.
    try:
        pil_image = Image.open(upload.stream).convert('RGB')
        bgr_image = pil_to_bgr(pil_image)
    except Exception as exc:
        return {'ok': False, 'filename': filename, 'error': f'Could not read image: {exc}'}

    # Run the filter (always automatic mode).
    try:
        doc_filter = CrispenEngine(auto_detect=True)
        cleaned, stages = doc_filter.run(bgr_image)
    except Exception as exc:
        return {'ok': False, 'filename': filename, 'error': f'Filtering failed: {exc}'}

    return {
        'ok': True,
        'filename': filename,
        'category': stages.get('detected_type', 'unknown'),
        'original': _bgr_to_data_uri(bgr_image),
        'result': _bgr_to_data_uri(cleaned),
    }


@app.route('/api/filter', methods=['POST'])
def filter_document():
    """
    Run Crispen on one OR many uploaded images.

    Accepts any number of files under the form field "image" (the browser
    sends one entry per selected file). Returns:
        { "ok": true, "results": [ {per-image result}, ... ] }
    Each per-image result carries its own `ok` flag, so a single bad file
    doesn't sink the batch.
    """
    uploads = request.files.getlist('image')
    uploads = [u for u in uploads if u and u.filename]

    if not uploads:
        return jsonify({'ok': False, 'error': 'No files named "image" in the request.'}), 400

    results = [_filter_one_upload(upload) for upload in uploads]

    return jsonify({'ok': True, 'results': results})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
