"""Assistive LLM Sentence Generation Endpoints."""
from fastapi import APIRouter
from backend.app.schemas.prediction import GenerateSentenceRequest, GenerateSentenceResponse
from backend.app.services.llm_service import llm_service

router = APIRouter(prefix="/api", tags=["LLM Sentence Generation"])


@router.post(
    "/generate-sentence",
    response_model=GenerateSentenceResponse,
    summary="Convert a single imagined word into a natural first-person assistive sentence",
)
def generate_sentence(req: GenerateSentenceRequest):
    result = llm_service.generate_sentence(req.word, use_memory=bool(req.use_memory))
    return GenerateSentenceResponse(
        word=req.word,
        sentence=result["sentence"],
        generation_mode=result["generation_mode"],
    )
