from fastapi import APIRouter

from src.app.schemas.voice import InstructionPayload, InstructionRequest
from src.app.services.groq_service import get_instruction_from_llm

router = APIRouter(tags=["instruction"])


@router.post("/instruction", response_model=InstructionPayload)
def route_instruction(payload: InstructionRequest) -> InstructionPayload:
    result = get_instruction_from_llm(payload.transcription)
    return InstructionPayload(**result)