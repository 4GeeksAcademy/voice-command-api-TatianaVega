from fastapi import APIRouter, HTTPException, Request, status
from groq import Groq
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from src.app.api.routes import tasks as tasks_module
from src.app.core.config import get_settings
from src.app.schemas.voice import (
    InstructionPayload,
    TaskCreate,
    TaskReplace,
    TaskUpdate,
    TranscribeFlowResponse,
)
from src.app.services.groq_service import get_instruction_from_llm

router = APIRouter(tags=["transcribe"])


@router.get("/")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


def transcribe_audio(audio_bytes: bytes, filename: str, language: str) -> str:
    """Convierte audio en texto usando Whisper (de Groq)."""
    settings = get_settings()
    client = Groq(
        api_key=settings.groq_api_key,
        timeout=settings.request_timeout_seconds,
    )
    options = {}
    if language:
        options["language"] = language
    try:
        result = client.audio.transcriptions.create(
            file=(filename, audio_bytes),
            model=settings.groq_transcription_model,
            **options,
        )
        return result.text.strip()
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Error al transcribir con Whisper: {error}",
        )


def execute_instruction(instruction: InstructionPayload):
    """Hace lo que pidió la IA, usando las funciones de tasks.py."""
    endpoint = instruction.endpoint.rstrip("/")
    method = instruction.method.upper()
    params = instruction.params

    try:
        if endpoint == "/tasks":
            if method == "GET":
                return tasks_module.get_tasks()
            if method == "POST":
                return tasks_module.create_task(TaskCreate(**params))

        elif endpoint.startswith("/tasks/"):
            try:
                task_id = int(endpoint.split("/")[2])
            except (IndexError, ValueError):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Id de tarea inválido en {instruction.endpoint}",
                )
            if method == "PATCH":
                return tasks_module.update_task(task_id, TaskUpdate(**params))
            if method == "PUT":
                return tasks_module.replace_task(task_id, TaskReplace(**params))
            if method == "DELETE":
                return tasks_module.delete_task(task_id)
    except ValidationError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Parámetros inválidos: {error.errors()[0]['msg']}",
        )

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"La IA pidió algo que no existe: {method} {instruction.endpoint}",
    )


@router.post("/transcribe", response_model=TranscribeFlowResponse)
async def transcribe_and_run_flow(request: Request) -> TranscribeFlowResponse:
    content_type = request.headers.get("content-type", "")

    # Caso 1: llegó un audio (formulario)
    if "multipart/form-data" in content_type:
        form = await request.form()
        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Falta el archivo de audio en el campo 'file'",
            )
        audio_bytes = await upload.read()
        if not audio_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El audio llegó vacío",
            )
        language = str(form.get("language") or "").strip()
        transcription = await run_in_threadpool(
            transcribe_audio, audio_bytes, upload.filename or "audio.webm", language
        )

    # Caso 2: llegó texto escrito a mano (JSON)
    else:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El cuerpo no es JSON válido",
            )
        transcription = str(body.get("transcription", "")).strip()

    if not transcription:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No se pudo entender ninguna frase",
        )

    raw_instruction = await run_in_threadpool(get_instruction_from_llm, transcription)
    try:
        instruction = InstructionPayload(**raw_instruction)
    except ValidationError:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="La IA devolvió un JSON con formato incorrecto",
        )

    result = execute_instruction(instruction)

    return TranscribeFlowResponse(
        transcription=transcription,
        instruction=instruction,
        result=result,
    )
