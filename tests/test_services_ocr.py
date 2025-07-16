# app/tests/test_services_ocr.py
from app.services import ocr


def test_extract_text_pdf_valid():
    with open("tests/assets/sample_resume.pdf", "rb") as f:
        content = f.read()

    text = ocr.extract_text("sample_resume.pdf", content)
    assert isinstance(text, str)
    assert len(text.strip()) > 0


def test_extract_text_pdf_empty():
    text = ocr.extract_text("empty.pdf", b"")
    assert "Erro" in text
