from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool

from lrage.webapp import presets, registry_meta

router = APIRouter(prefix="/api/meta", tags=["meta"])


@router.get("/tasks")
async def get_tasks():
    # First call imports the task stack and indexes ~2.5k YAMLs (tens of
    # seconds); run off the event loop so /health stays responsive.
    tasks = await run_in_threadpool(registry_meta.list_tasks)
    return {"tasks": tasks}


@router.get("/registries")
def get_registries():
    return {
        "model_types": registry_meta.MODEL_TYPES,
        "retrievers": registry_meta.RETRIEVERS,
        "retriever_types": registry_meta.RETRIEVER_TYPES,
        "rerankers": registry_meta.RERANKERS,
        "reranker_types": registry_meta.RERANKER_TYPES,
    }


@router.get("/presets")
def get_presets():
    return presets.load_presets()


@router.get("/devices")
def get_devices():
    return {"devices": registry_meta.list_devices()}
