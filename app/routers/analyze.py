from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional
from uuid import UUID
from datetime import datetime, timezone
from app.services import ocr, logger, summarizer, question_answering
from app.schemas.analyze import AnalyzeResponse
import logging

analyze_router = APIRouter()


logging.basicConfig(level=logging.INFO)
logger_system = logging.getLogger(__name__)


@analyze_router.post(
    "/",
    summary="Analisa currículos (PDF ou imagens)",
    response_model=AnalyzeResponse,
    responses={
        200: {"description": "Resultado da análise dos currículos"},
        400: {"description": "Requisição malformada ou sem arquivos válidos"},
        500: {"description": "Erro interno da aplicação"},
    }
)
async def analyze_files(
    files: List[UploadFile] = File(
        ...,
        description="Lista de arquivos (PDFs ou imagens JPG/PNG)"
    ),
    request_id: UUID = Form(..., description="Identificador único da requisição"),  # noqa: E501
    user_id: str = Form(..., description="ID do usuário solicitante"),
    query: Optional[str] = Form(
        None,
        description="Consulta de recrutamento. Ex: 'Quem se encaixa melhor para vaga de Engenheiro Python?'"  # noqa: E501
    )
):
    """
    Analisa currículos enviados (PDF/JPG/PNG) via OCR e IA.

    - Se **query** for enviada, retorna uma análise de aderência com
    justificativa.
    - Se **query** for omitida, retorna apenas um sumário de cada currículo.

    Args:
    - files (List[UploadFile]): Arquivos de currículo (PDF ou imagem).
    - request_id (UUID): ID da requisição (para auditoria).
    - user_id (str): ID do usuário que fez a solicitação.
    - query (Optional[str]): Pergunta ou filtro da vaga para análise.

    Returns:
    - dict: Resultado por arquivo, com resposta ou resumo.
    """
    if not files:
        raise HTTPException(status_code=400, detail="Nenhum arquivo foi enviado")  # noqa: E501

    resumes_texts = []
    filenames = []

    # Valida e processa arquivos
    allowed_types = {"application/pdf", "image/jpeg", "image/png"}

    for file in files:
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Tipo de arquivo inválido: {file.filename} ({file.content_type})"  # noqa: E501
            )

        try:
            content = await file.read()
            logger_system.info(f"Processando arquivo: {file.filename}")

            text = ocr.extract_text(file.filename, content)

            if not text or len(text.strip()) < 20:
                logger_system.warning(f"Texto extraído muito curto para {file.filename}")  # noqa: E501
                text = f"Erro: Não foi possível extrair texto suficiente de {file.filename}"  # noqa: E501

            resumes_texts.append(text)
            filenames.append(file.filename)

        except Exception as e:
            logger_system.error(f"Erro ao processar {file.filename}: {str(e)}")
            resumes_texts.append(f"Erro ao processar arquivo: {str(e)}")
            filenames.append(file.filename)

    results = {}

    if query:
        logger_system.info(f"Analisando currículos para query: {query}")

        summarized_texts = []
        for i, text in enumerate(resumes_texts):
            if text.startswith("Erro"):
                summarized_texts.append(text)
            else:
                summary = summarizer.summarize_text(text)
                summarized_texts.append(summary)
                logger_system.info(f"Resumo gerado para {filenames[i]}: {summary[:100]}...")  # noqa: E501

        for i, (summary, filename) in enumerate(zip(summarized_texts, filenames)):  # noqa: E501
            try:
                if summary.startswith("Erro"):
                    results[filename] = {
                        "answer": "Erro",
                        "justification": summary
                    }
                else:
                    analysis = question_answering.analyze_resume_for_position(summary, query)  # noqa: E501
                    results[filename] = {
                        "answer": analysis["answer"],
                        "justification": analysis["justification"],
                        "resume_summary": summary
                    }
            except Exception as e:
                logger_system.error(f"Erro na análise de {filename}: {str(e)}")
                results[filename] = {
                    "answer": "Erro",
                    "justification": f"Erro durante análise: {str(e)}"
                }

    else:
        logger_system.info("Gerando resumos dos currículos")
        for i, (text, filename) in enumerate(zip(resumes_texts, filenames)):
            try:
                if text.startswith("Erro"):
                    results[filename] = {
                        "summary": text
                    }
                else:
                    summary = summarizer.summarize_text(text)
                    results[filename] = {
                        "summary": summary
                    }
            except Exception as e:
                logger_system.error(f"Erro ao gerar resumo para {filename}: {str(e)}")  # noqa: E501
                results[filename] = {
                    "summary": f"Erro ao gerar resumo: {str(e)}"
                }

    log_data = {
        "request_id": str(request_id),
        "user_id": user_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "files_processed": len(files),
        "filenames": filenames,
        "resultado": results
    }

    try:
        logger.save_log(log_data)
        logger_system.info(f"Log salvo para request_id: {request_id}")
    except Exception as e:
        logger_system.error(f"Erro ao salvar log: {str(e)}")

    return JSONResponse(content=results)
