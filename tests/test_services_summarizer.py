# tests/test_services_summarizer.py
from unittest.mock import patch
from app.services import summarizer


def test_summarize_text_normal():
    text = ("Este é um exemplo de currículo com informações relevantes para "
            "sumarização.")

    with patch("app.services.summarizer.summarizer") as mock_pipeline:
        mock_pipeline.return_value = [{"summary_text": "Resumo simulado"}]
        summary = summarizer.summarize_text(text)
        assert summary == "Resumo simulado"


def test_summarize_text_empty():
    text = "       "

    summary = summarizer.summarize_text(text)
    assert summary == "Texto vazio, não foi possível gerar resumo."


def test_summarize_text_long():
    text = "Test " * 2000

    with patch("app.services.summarizer.summarizer") as mock_pipeline:
        mock_pipeline.return_value = [{"summary_text": ("Resumo de texto muito"
                                                        " longo")}]
        summary = summarizer.summarize_text(text)
        assert summary == "Resumo de texto muito longo"

        mock_pipeline.assert_called_once()
        args, kwargs = mock_pipeline.call_args
        assert len(args[0]) <= 1000


def test_summarize_text_erro_pipeline():
    text = "Texto válido, mas o pipeline quebra"

    with patch(
        "app.services.summarizer.summarizer",
        side_effect=Exception("Falha no modelo")
    ):
        summary = summarizer.summarize_text(text)
        assert summary.startswith("Erro ao gerar resumo:")
