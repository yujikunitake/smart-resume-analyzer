# app/routers/analyze.py
from fastapi import APIRouter, UploadFile, File, Form
from typing import List, Optional
from uuid import UUID
from datetime import datetime, timezone
from app.services import ocr, logger, summarizer

analyze_router = APIRouter()


@analyze_router.post("/")
async def analyze_files(
    files: List[UploadFile] = File(...),
    request_id: UUID = Form(...),
    user_id: str = Form(...),
    query: Optional[str] = Form(None)
):
    results = {}

    for file in files:
        content = await file.read()
        extracted_text = ocr.extract_text(file.filename, content)
        if query:
            results[file.filename] = extracted_text
        else:
            summary = summarizer.summarize_text(extracted_text)
            results[file.filename] = {"summary": summary}

    log_data = {
        "request_id": str(request_id),
        "user_id": user_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "resultado": results
    }

    logger.save_log(log_data)
    return results
