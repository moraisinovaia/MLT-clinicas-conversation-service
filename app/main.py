import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.v1 import conversation, feedback, health, eval_retrieval
from app.integrations.supabase_client import get_pool, close_pool
from app.integrations.gt_inova import GTInovaClient
from app.core.config import settings

logging.basicConfig(level=settings.log_level)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await get_pool()   # aquece o pool na inicialização
    if settings.gt_inova_base_url and settings.gt_inova_api_key:
        app.state.gt_inova = GTInovaClient(settings.gt_inova_base_url, settings.gt_inova_api_key)
    else:
        app.state.gt_inova = None
    yield
    await close_pool()


app = FastAPI(
    title="MLT Clínicas — Conversation Service",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(conversation.router, prefix="/api/v1")
app.include_router(feedback.router, prefix="/api/v1")
app.include_router(eval_retrieval.router, prefix="/api/v1")
