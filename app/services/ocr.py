# app/services/ocr.py
from PIL import Image
import easyocr
import io
from pdf2image import convert_from_bytes
import numpy as np

reader = easyocr.Reader(['pt'])


def extract_text(filename: str, content_bytes: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        images = convert_from_bytes(content_bytes)
        full_text = ""
        for image in images:
            image_array = np.array(image)
            result = reader.readtext(image_array)
            page_text = " ".join(item[1] for item in result)
            full_text += page_text + "\n"
        return full_text
    else:
        image = Image.open(io.BytesIO(content_bytes))
        image_array = np.array(image)
        result = reader.readtext(image_array)
        text = " ".join([item[1] for item in result])
        return text
