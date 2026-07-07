import os

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from ocr import transcribe

app = Flask(__name__)

ALLOWED = (".pdf", ".png", ".jpg", ".jpeg")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/convert", methods=["POST"])
def convert():
    """Synchronous, single-request conversion.

    One file in, transcription JSON out, all within this request. There is
    no background thread, job registry, or /status polling -- none of which
    survive Vercel's serverless model, where each request runs in its own
    short-lived, stateless function instance.
    """
    api_key = request.form.get("api_key")
    if not api_key:
        return jsonify({"error": "OpenAI API key is required"}), 400

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED:
        return jsonify({"error": f"Unsupported file type: {ext or 'unknown'}"}), 400

    file_bytes = file.read()

    try:
        pages = transcribe(file_bytes, ext, api_key)
    except Exception as e:  # surface the real reason to the UI
        return jsonify({"error": str(e)}), 500

    return jsonify({"filename": filename, "pages": pages})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
