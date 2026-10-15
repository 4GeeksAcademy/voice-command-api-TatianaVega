from fastapi import APIRouter, HTTPException, status

from src.app.schemas.voice import Task, TaskCreate, TaskReplace, TaskUpdate

router = APIRouter(prefix="/tasks", tags=["tasks"])

# La "libreta" donde se guardan las tareas (vive en la memoria)
tasks: list[dict] = []

# Contador para dar un número único a cada tarea
next_id = 1


def find_task(task_id: int) -> dict:
    """Busca una tarea por su id. Si no existe, avisa con un error 404."""
    for task in tasks:
        if task["id"] == task_id:
            return task
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"No existe la tarea con id {task_id}",
    )


@router.get("", response_model=list[Task])
def get_tasks() -> list[dict]:
    return tasks


@router.post("", response_model=Task, status_code=status.HTTP_201_CREATED)
def create_task(payload: TaskCreate) -> dict:
    global next_id
    new_task = {"id": next_id, "title": payload.title, "done": payload.done}
    next_id += 1
    tasks.append(new_task)
    return new_task


@router.put("/{task_id}", response_model=Task)
def replace_task(task_id: int, payload: TaskReplace) -> dict:
    task = find_task(task_id)
    task["title"] = payload.title
    task["done"] = payload.done
    return task


@router.patch("/{task_id}", response_model=Task)
def update_task(task_id: int, payload: TaskUpdate) -> dict:
    task = find_task(task_id)
    if payload.title is not None:
        task["title"] = payload.title
    if payload.done is not None:
        task["done"] = payload.done
    return task


@router.delete("/{task_id}")
def delete_task(task_id: int) -> dict[str, str]:
    task = find_task(task_id)
    tasks.remove(task)
    return {"message": f"Tarea {task_id} eliminada"}