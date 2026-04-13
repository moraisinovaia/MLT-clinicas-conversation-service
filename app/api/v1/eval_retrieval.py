"""
Endpoint de debug para avaliação de retrieval.

Exposto apenas quando EVAL_RETRIEVAL_ENABLED=true (nunca em produção).
Permite que o framework de eval (run_eval.py --mode e2e) meça chunk_recall@k
sem acesso direto ao banco — chama hybrid_search_v2 e devolve os IDs dos chunks.
"""
from __future__ import annotations
import os
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.integrations.supabase_client import get_pool
from app.routes.rag import _get_query_embedding, RAG_EXCLUDED_SOURCE_TYPES

router = APIRouter()


def _check_enabled():
    if os.getenv("EVAL_RETRIEVAL_ENABLED", "false").lower() != "true":
        raise HTTPException(status_code=404, detail="Not found")


class RetrievalRequest(BaseModel):
    query:        str
    cliente_id:   str
    risk_max:     str = "high"
    source_types: list[str] = []
    k:            int = 6


class RetrievalResponse(BaseModel):
    chunk_ids:    list[str]
    source_types: list[str]
    scores:       list[float]


@router.post("/eval/retrieval", response_model=RetrievalResponse)
async def eval_retrieval(req: RetrievalRequest, _: None = Depends(_check_enabled)):
    """
    Chama hybrid_search_v2 e devolve os IDs dos chunks recuperados.
    Usado pelo run_eval.py --mode e2e para medir chunk_recall@k.

    Requer EVAL_RETRIEVAL_ENABLED=true no ambiente.
    """
    pool = await get_pool()

    try:
        embedding = await _get_query_embedding(req.query)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"embedding error: {e}")

    embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"
    source_types_param = req.source_types if req.source_types else None

    async with pool.acquire() as db:
        rows = await db.fetch(
            "SELECT * FROM hybrid_search_v2($1, $2, $3, $4, $5, $6, $7, $8)",
            req.query,
            embedding_str,
            req.cliente_id,
            req.k,
            req.risk_max,
            source_types_param,
            None,   # procedure_id
            None,   # doctor_id
        )

    rows = [r for r in rows if r["source_type"] not in RAG_EXCLUDED_SOURCE_TYPES]

    return RetrievalResponse(
        chunk_ids=[str(r["id"])[:8] for r in rows],
        source_types=[r["source_type"] for r in rows],
        scores=[float(r.get("rrf_score", 0.0)) for r in rows],
    )
