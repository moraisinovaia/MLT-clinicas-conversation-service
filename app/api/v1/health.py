from fastapi import APIRouter
from app.integrations.supabase_client import get_pool

router = APIRouter()


@router.get("/health")
async def health():
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        db_ok = True
    except Exception:
        db_ok = False

    return {"status": "ok" if db_ok else "degraded", "db": db_ok}
