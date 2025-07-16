# app/main.py
from fastapi import FastAPI
from routers.analyze import analyze_router


app = FastAPI(
    title="Smart Resume Analyzer",
    description="Recebe currículos, extrai informações e responde "
                "queries via IA.",
    version="1.0.0"
)

app.include_router(
    analyze_router,
    prefix="/analyze",
    tags=["Analyze"]
)
