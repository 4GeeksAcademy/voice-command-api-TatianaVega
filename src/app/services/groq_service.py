import json

from fastapi import HTTPException, status
from groq import Groq

from src.app.core.config import get_settings

SYSTEM_PROMPT = """Eres un traductor de comandos de voz para una app de lista de tareas.
Recibes una frase del usuario y respondes ÚNICAMENTE con un objeto JSON válido.
No escribas texto antes ni después. No uses markdown. No uses ```.

El JSON debe tener exactamente estas tres claves: "endpoint", "method", "params".

Acciones posibles:

1. Crear una tarea:
{"endpoint": "/tasks", "method": "POST", "params": {"title": "<texto de la tarea>"}}

2. Ver / listar todas las tareas:
{"endpoint": "/tasks", "method": "GET", "params": {}}

3. Marcar una tarea como completada (id es un número entero):
{"endpoint": "/tasks/<id>", "method": "PATCH", "params": {"done": true}}

4. Cambiar el título de una tarea:
{"endpoint": "/tasks/<id>", "method": "PATCH", "params": {"title": "<nuevo título>"}}

5. Eliminar una tarea:
{"endpoint": "/tasks/<id>", "method": "DELETE", "params": {}}

Reglas:
- Reemplaza <id> por el número que diga el usuario (por ejemplo "/tasks/3").
- Si el usuario dice "la tarea tres", usa el número 3.
- Responde siempre con el JSON y nada más.
"""


def get_instruction_from_llm(transcription: str) -> dict:
    settings = get_settings()
    client = Groq(
        api_key=settings.groq_api_key,
        timeout=settings.request_timeout_seconds,
    )

    try:
        completion = client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": transcription},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content
        return json.loads(content)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="La IA no devolvió un JSON válido",
        )
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Error al hablar con Groq: {error}",
        )