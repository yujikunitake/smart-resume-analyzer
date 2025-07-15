# app/routers/analyze.py
from fastapi import APIRouter, UploadFile, File, Form
from typing import List, Optional
from uuid import UUID
from datetime import datetime
from app.services import ocr, logger

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
        extracted_text = ocr.extract_text(content)
        results[file.filename] = extracted_text

    log_data = {
        "request_id": str(request_id),
        "user_id": user_id,
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "resultado": results
    }

    logger.save_log(log_data)
    return results
