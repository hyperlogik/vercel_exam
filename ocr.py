import os
import base64
import json
from io import BytesIO

import fitz  # PyMuPDF
from PIL import Image
from openai import OpenAI


def encode_image(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def extract_pages(file_bytes, ext):
    """Yield page images one at a time, entirely from memory (no disk).

    Vercel's filesystem is read-only apart from /tmp, so everything here
    works on in-memory bytes rather than file paths.
    """
    ext = ext.lower()

    if ext == ".pdf":
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        try:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                # 150 dpi keeps memory sane while staying legible for GPT-4o.
                pix = page.get_pixmap(dpi=150)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                pix = None  # release the pixmap buffer promptly
                img.thumbnail((2000, 2000))
                yield img
        finally:
            doc.close()
    elif ext in [".png", ".jpg", ".jpeg"]:
        img = Image.open(BytesIO(file_bytes)).convert("RGB")
        img.thumbnail((2000, 2000))
        yield img


def transcribe(file_bytes, ext, api_key):
    """Transcribe every page synchronously and return a list of pages.

    Each page is a list of blocks: {'type': 'printed'|'handwritten', 'text': str}.
    No threads, no job registry, no disk -- everything the request needs is
    produced and returned within this single call, as Vercel's model requires.
    """
    client = OpenAI(api_key=api_key)
    pages = []

    for img in extract_pages(file_bytes, ext):
        base64_image = encode_image(img)
        img = None  # free the page before waiting on the API

        response = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a transcriber converting scanned exam papers. "
                        "Return a JSON object with a 'blocks' array. Each block must "
                        "have 'type' ('printed' for standard question text, "
                        "'handwritten' for student answers/working) and 'text'. "
                        "Include crossed out text wrapped in ~~. Describe sketches "
                        "in brackets."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Transcribe this exam page."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                },
            ],
            max_tokens=4000,
        )

        try:
            content = json.loads(response.choices[0].message.content)
            blocks = content.get("blocks", [])
        except json.JSONDecodeError:
            blocks = [
                {
                    "type": "printed",
                    "text": "[Error parsing transcription JSON structure for this page]",
                }
            ]

        pages.append(blocks)

    return pages
