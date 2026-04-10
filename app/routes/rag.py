"""
Rota rag — Fase 2.

Pipeline:
  1. Gerar embedding da query via OpenAI
  2. Chamar hybrid_search_v2() no Supabase (vetor + FTS + RRF)
  3. Verificar vw_knowledge_conflicts (HIGH risk)
  4. Compor resposta com LLM
  5. Escrever retrieval_logs
"""
from __future__ import annotations
import json
import uuid
import logging
from datetime import datetime, timezone
import httpx
import asyncpg

from app.models.intent import ParsedIntent
from app.models.conversation import OutboundMessage
from app.core.policy_engine import RagFilters
from app.core.config import settings
from app.integrations.openrouter import call_llm

logger = logging.getLogger(__name__)

# source_types que contêm dado operacional vivo e NUNCA devem ser retornados pelo RAG.
# Elegibilidade de convênio, serviços ativos e horários são autoridade da GT Inova.
# O RAG pode explicar contexto, mas não decide "aceita/não aceita".
RAG_EXCLUDED_SOURCE_TYPES: frozenset[str] = frozenset({
    "insurance_info",
})

# Resposta padrão quando não há documento aprovado e vigente
NO_APPROVED_DOC = (
    "Para orientações específicas sobre preparo, entre em contato com a clínica "
    "pelo telefone da recepção. Nossa equipe vai te ajudar."
)

CONFLICT_RESPONSE = (
    "Temos informações sobre esse assunto que precisam de revisão pela equipe clínica. "
    "Para garantir que você receba a orientação correta, entre em contato com a recepção."
)

COMPOSE_SYSTEM = """\
Você é um assistente de clínica médica que responde via WhatsApp.

Estrutura obrigatória da resposta (3 partes, sem marcadores visuais):
1. Resposta direta — responda a pergunta em 1-2 frases objetivas.
2. Instrução prática — oriente o que o paciente deve fazer, observar ou evitar (só se aplicável ao contexto).
3. Próximo passo — indique o contato com a recepção ou equipe clínica quando a situação exigir acompanhamento.

Regras invioláveis:
- Use APENAS as informações dos trechos fornecidos. Nunca invente ou infira além do texto.
- Sem markdown: sem **, ##, listas com -, emojis ou qualquer formatação especial.
- Linguagem clara, acolhedora e direta — o paciente lê no celular.
- Máximo 3 parágrafos curtos. Nunca ultrapasse 150 palavras.
- Para trechos marcados com risco "high": seja preciso e conservador; sempre ofereça contato com a equipe clínica ao final.
- Se a informação estiver incompleta nos trechos, diga que a equipe pode detalhar — nunca complete lacunas com suposições.
"""


async def _get_query_embedding(query: str) -> list[float]:
    """Gera embedding da query via OpenAI text-embedding-3-small."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            json={
                "model":      "text-embedding-3-small",
                "input":      query,
                "dimensions": 1536,
            },
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]


async def _get_conflicting_subjects(
    chunk_ids:  list[str],
    cliente_id: str,
    db:         asyncpg.Connection,
) -> set[str]:
    """
    Fix 7: verifica conflitos usando canonical_subject de knowledge_documents,
    não source_type do chunk. A view vw_knowledge_conflicts agrupa por
    canonical_subject — usar source_type era semanticamente errado.
    """
    if not chunk_ids:
        return set()
    rows = await db.fetch(
        """
        SELECT DISTINCT kd.canonical_subject
        FROM knowledge_chunks kc
        JOIN knowledge_documents kd ON kd.id = kc.document_id
        JOIN vw_knowledge_conflicts vc
          ON vc.cliente_id        = kd.cliente_id
         AND vc.canonical_subject = kd.canonical_subject
        WHERE kc.id        = ANY($1::uuid[])
          AND kd.cliente_id = $2
        """,
        chunk_ids,
        cliente_id,
    )
    return {r["canonical_subject"] for r in rows}


async def _write_retrieval_log(
    db:               asyncpg.Connection,
    cliente_id:       str,
    session_id:       str,
    query:            str,
    intent:           ParsedIntent,
    filters:          RagFilters,
    vector_candidates: list[dict],
    fts_candidates:   list[dict],
    final_chunks:     list[dict],
    confidence:       float,
    route:            str,
    fallback_reason:  str | None,
) -> None:
    try:
        await db.execute(
            """
            INSERT INTO retrieval_logs (
                id, cliente_id, session_id,
                query_text, normalized_query,
                intent_detected, risk_level,
                filters_applied,
                vector_candidates, fts_candidates,
                reranked_candidates, final_chunks,
                confidence_score, route_selected, fallback_reason,
                created_at
            ) VALUES (
                $1, $2, $3,
                $4, $5,
                $6, $7,
                $8::jsonb,
                $9::jsonb, $10::jsonb,
                $11::jsonb, $12::jsonb,
                $13, $14, $15,
                NOW()
            )
            """,
            str(uuid.uuid4()),
            cliente_id,
            session_id,
            query,
            query.lower().strip(),
            intent.intent.value,
            intent.risk_level,
            json.dumps({"risk_max": filters.risk_max, "source_types": filters.source_types}),
            json.dumps(vector_candidates),
            json.dumps(fts_candidates),
            json.dumps(final_chunks),   # reranked = final após RRF (já feito no SQL)
            json.dumps(final_chunks),
            confidence,
            route,
            fallback_reason,
        )
    except Exception as e:
        logger.warning("retrieval_log write failed: %s", e)


async def _write_knowledge_feedback(
    db:           asyncpg.Connection,
    cliente_id:   str,
    session_id:   str,
    query:        str,
    route:        str,
    chunks_used:  list[dict],
    answer_text:  str,
    confidence:   float,
) -> str:
    """
    Insere linha em knowledge_feedback e retorna o UUID gerado.
    was_helpful começa NULL — preenchido via PATCH /feedback/{id} depois.
    """
    feedback_id = str(uuid.uuid4())
    try:
        await db.execute(
            """
            INSERT INTO knowledge_feedback (
                id, cliente_id, session_id,
                query_text, route_selected,
                chunks_used, answer_text, confidence_score,
                was_helpful, human_corrected
            ) VALUES (
                $1, $2, $3,
                $4, $5,
                $6::jsonb, $7, $8,
                NULL, false
            )
            """,
            feedback_id,
            cliente_id,
            session_id or None,
            query,
            route,
            json.dumps(chunks_used),
            answer_text,
            confidence,
        )
    except Exception as e:
        logger.warning("knowledge_feedback write failed: %s", e)
    return feedback_id


async def execute_rag(
    intent:     ParsedIntent,
    filters:    RagFilters,
    cliente_id: str,
    db:         asyncpg.Connection,
    session_id: str = "",
    query:      str = "",
) -> tuple[list[OutboundMessage], str | None]:
    """
    Executa a rota RAG completa:
      embedding → hybrid_search_v2 → conflict check → LLM compose → log → feedback

    Retorna (messages, feedback_id). feedback_id é None em caso de erro antecipado.
    """
    if not query:
        query = intent.mensagem_usuario or ""

    # ── 1. Gerar embedding ────────────────────────────────────────────────────
    try:
        embedding = await _get_query_embedding(query)
    except Exception as e:
        logger.error("embedding_error: %s", e)
        await _write_retrieval_log(
            db, cliente_id, session_id, query, intent, filters,
            [], [], [], 0.0, "rag", f"embedding_error: {e}"
        )
        return [OutboundMessage(text=NO_APPROVED_DOC)], None

    # ── 2. hybrid_search_v2 ───────────────────────────────────────────────────
    # pgvector via asyncpg exige string '[f1,f2,...]', não list Python
    embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"

    # source_types: quando explícito, já vem sem insurance_info (policy_engine não inclui).
    # Quando vazio (= todos), passa None ao SQL e filtramos pós-busca para garantir.
    source_types_param = filters.source_types if filters.source_types else None
    rows = await db.fetch(
        "SELECT * FROM hybrid_search_v2($1, $2, $3, $4, $5, $6, $7, $8)",
        query,
        embedding_str,
        cliente_id,
        6,                           # limit
        filters.risk_max,
        source_types_param,
        filters.procedure_id or None,
        filters.doctor_id    or None,
    )

    # Exclui source_types operacionais que nunca devem ser autoridade no RAG.
    # Garante exclusão mesmo quando source_types_param era None (= todos).
    rows = [r for r in rows if r["source_type"] not in RAG_EXCLUDED_SOURCE_TYPES]

    if not rows:
        await _write_retrieval_log(
            db, cliente_id, session_id, query, intent, filters,
            [], [], [], 0.0, "rag", "no_chunks_found"
        )
        return [OutboundMessage(text=NO_APPROVED_DOC)], None

    # ── 3. Verificar conflitos de alto risco (Fix 7) ─────────────────────────
    # Bloqueia resposta se houver canonical_subjects conflitantes.
    # Determinístico — não usa LLM para decidir segurança clínica.
    if intent.risk_level == "high":
        chunk_ids = [str(r["chunk_id"]) for r in rows]
        conflicting = await _get_conflicting_subjects(chunk_ids, cliente_id, db)
        if conflicting:
            await _write_retrieval_log(
                db, cliente_id, session_id, query, intent, filters,
                [], [], [], 0.0, "rag",
                f"conflict_detected:{','.join(conflicting)}"
            )
            return [OutboundMessage(text=CONFLICT_RESPONSE)], None

    # ── 4. Montar contexto para o LLM ─────────────────────────────────────────
    chunks_for_log = [
        {
            "chunk_id":   str(r["chunk_id"]),
            "rrf_score":  r["rrf_score"],
            "source_type": r["source_type"],
        }
        for r in rows
    ]

    context_parts = []
    for i, r in enumerate(rows, 1):
        context_parts.append(
            f"[Trecho {i} | {r['source_type']} | risco:{r['risk_level']}]\n{r['chunk_text']}"
        )
    context = "\n\n".join(context_parts)

    user_content = (
        f"Tipo de dúvida: {intent.intent.value} | Risco: {intent.risk_level}\n\n"
        f"Trechos de conhecimento da clínica:\n{context}\n\n"
        f"Pergunta do paciente: {query}"
    )

    # ── 5. Compor resposta com LLM ────────────────────────────────────────────
    confidence = float(rows[0]["rrf_score"]) if rows else 0.0
    try:
        resposta = await call_llm(system=COMPOSE_SYSTEM, user=user_content)
    except Exception as e:
        logger.error("llm_compose_error: %s", e)
        resposta = NO_APPROVED_DOC
        confidence = 0.0

    # ── 6. Escrever retrieval_log ─────────────────────────────────────────────
    await _write_retrieval_log(
        db, cliente_id, session_id, query, intent, filters,
        chunks_for_log,   # vector_candidates (aproximado — RRF já fundiu)
        chunks_for_log,   # fts_candidates
        chunks_for_log,   # final_chunks
        confidence,
        "rag",
        None,
    )

    # ── 7. Escrever knowledge_feedback (was_helpful = NULL até o paciente responder) ──
    feedback_id = await _write_knowledge_feedback(
        db          = db,
        cliente_id  = cliente_id,
        session_id  = session_id,
        query       = query,
        route       = "rag",
        chunks_used = chunks_for_log,
        answer_text = resposta,
        confidence  = confidence,
    )

    return [OutboundMessage(text=resposta)], feedback_id
