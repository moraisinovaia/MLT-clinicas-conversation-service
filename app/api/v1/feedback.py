"""
PATCH /api/v1/feedback/{feedback_id}

Atualiza knowledge_feedback após o paciente ou equipe clínica dar retorno:
  - was_helpful     → paciente respondeu sim/não (via N8N após interação WhatsApp)
  - human_corrected → equipe revisou e corrigiu a resposta
  - correction_note → nota da correção humana
"""
from __future__ import annotations
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.integrations.supabase_client import get_pool

logger = logging.getLogger(__name__)
router = APIRouter()


class FeedbackUpdate(BaseModel):
    was_helpful:      bool | None = None
    human_corrected:  bool | None = None
    correction_note:  str | None = None


@router.patch("/feedback/{feedback_id}", status_code=204)
async def update_feedback(feedback_id: str, body: FeedbackUpdate):
    """
    Atualiza os campos de feedback de uma resposta RAG.

    Chamado pelo N8N quando o paciente responde à pergunta de satisfação,
    ou pela equipe clínica via painel de revisão.
    Retorna 404 se o feedback_id não existir.
    """
    # Monta SET dinâmico só com os campos enviados
    updates: dict[str, object] = {}
    if body.was_helpful is not None:
        updates["was_helpful"] = body.was_helpful
    if body.human_corrected is not None:
        updates["human_corrected"] = body.human_corrected
    if body.correction_note is not None:
        updates["correction_note"] = body.correction_note

    if not updates:
        return  # nada a atualizar → 204 sem erro

    set_clause = ", ".join(
        f"{col} = ${i + 2}" for i, col in enumerate(updates.keys())
    )
    values = [feedback_id] + list(updates.values())

    pool = await get_pool()
    async with pool.acquire() as db:
        result = await db.execute(
            f"UPDATE knowledge_feedback SET {set_clause} WHERE id = $1",
            *values,
        )

    # asyncpg retorna "UPDATE N" — se N=0, o ID não existe
    if result == "UPDATE 0":
        raise HTTPException(status_code=404, detail="feedback_id não encontrado")
