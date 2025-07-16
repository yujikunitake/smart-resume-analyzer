# app/services/sumarizer.py
from transformers import pipeline

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")


def summarize_text(text: str) -> str:
    if len(text) > 1000:
        text = text[:1000]
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]["summary_text"]
