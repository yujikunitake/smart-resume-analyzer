# app/schemas/analyze.py
from pydantic import BaseModel, RootModel
from typing import Dict, Union


class ResumeSummary(BaseModel):
    summary: str


class ResumeAnalysis(BaseModel):
    answer: str
    justification: str
    resume_summary: str


class AnalyzeResponse(RootModel[Dict[str, Union[ResumeAnalysis, ResumeSummary]]]):  # noqa: E501
    pass
