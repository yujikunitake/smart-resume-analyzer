# app/main.py
from fastapi import FastAPI
from app.routers.analyze import analyze_router

app = FastAPI(
    title="Smart Resume Analyzer",
    description="Recebe currículos, extrai informações e responde queries via IA.",  # noqa: E501
    version="1.0.0"
)


app.include_router(
    analyze_router,
    prefix="/analyze",
    tags=["Analyze"]
)


# Rota raiz só para ver se está no ar
@app.get("/")
def root():
    return {"message": "Smart Resume Analyzer API is running"}
