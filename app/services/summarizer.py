# app/services/summarizer.py
from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1
summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    device=device,
    )


def summarize_text(text: str) -> str:
    if not text.strip():
        return "Texto vazio, não foi possível gerar resum."

    if len(text) > 1000:
        text = text[:1000]
    try:
        summary = summarizer(
            text,
            max_length=130,
            min_length=30,
            do_sample=False
            )
        return summary[0]["summary_text"]
    except Exception as e:
        return f"Erro ao gerar resumo: {str(e)}"
