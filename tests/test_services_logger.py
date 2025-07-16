# tests/test_services_logger.py
from unittest.mock import patch
from app.services import logger


def test_save_log_calls_insert_one():
    dummy_log = {
        "request_id": "123",
        "user_id": "fabio",
        "timestamp": "2025-07-16T21:00:00",
        "query": None,
        "resultado": {"resume.pdf": {"summary": "Texto simulado"}}
    }

    with patch("app.services.logger.collection.insert_one") as mock_insert:
        logger.save_log(dummy_log)
        mock_insert.assert_called_once_with(dummy_log)
