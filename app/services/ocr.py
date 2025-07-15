# app/services/ocr.py
from PIL import Image
import easyocr
import io

reader = easyocr.Reader(['pt'])


def extract_text(content_bytes):
    image = Image.open(io.BytesIO(content_bytes))
    result = reader.readtext(image)
    text = " ".join([item[1] for item in result])
    return text
