# tests/test_routers_analyze.py
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app
import uuid

client = TestClient(app)


@patch("app.services.ocr.extract_text")
def test_analyze_with_query(mock_ocr):
    mock_ocr.return_value = "Texto extraído simulado para consulta."

    with open("tests/assets/sample_resume.pdf", "rb") as fake_file:
        files = [("files", ("resume.pdf", fake_file, "text/plain"))]
        data = {
            "request_id": str(uuid.uuid4()),
            "user_id": "fabio",
            "query": "Qual experiência profissional?"
        }

        response = client.post("/analyze/", files=files, data=data)
        assert response.status_code == 200

        json_data = response.json()
        assert "resume.pdf" in json_data
        # Qdo tem query, o retorno será o texto extraído
        assert json_data["resume.pdf"] == ("Texto extraído simulado para"
                                           " consulta.")


@patch("app.services.summarizer.summarize_text")
def test_analyze_without_query(mock_summarize):
    mock_summarize.return_value = "Resumo simulado para teste."

    with open("tests/assets/sample_resume.pdf", "rb") as fake_file:
        files = [("files", ("resume.pdf", fake_file, "text/plain"))]
        data = {
            "request_id": str(uuid.uuid4()),
            "user_id": "fabio"
        }

        response = client.post("/analyze/", files=files, data=data)
        assert response.status_code == 200
        json_data = response.json()
        assert "resume.pdf" in json_data
        assert json_data["resume.pdf"]["summary"] == (
            "Resumo simulado para teste."
            )
