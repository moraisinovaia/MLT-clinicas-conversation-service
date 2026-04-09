"""
POST /api/v1/conversation — ponto de entrada único do FastAPI.

Pipeline (ordem obrigatória):
  1. acquire_conversation_lock
  2. load_session
  3. pre_parser (sem LLM)
  4. semantic_parse (LLM → ParsedIntent)
  5. alias_lookup (convenio → convenio_canonico)
  6. merge_entities (acumula entidades entre requests — nunca sobrescreve com None)
  7. compute_missing_fields (dados_faltantes real, baseado no intent)
  8. policy_engine (decide rota)
  9. executor da rota
 10. resolve_next_state
 11. save_session
 12. release_conversation_lock (sempre no finally — com fetchval + log)
"""
from __future__ import annotations
import uuid
import logging
from fastapi import APIRouter, Request

from app.models.conversation import ConversationRequest, ConversationResponse, OutboundMessage
from app.models.state import ConversationState
from app.models.intent import ParseError, IntentType, EntitySet, ParsedIntent
from app.core.pre_parser import pre_parse
from app.core.semantic_parser import semantic_parse
from app.core.alias_lookup import canonicalize_entities
from app.core.policy_engine import decide_route
from app.core.state_machine import resolve_next_state
from app.core.session import (
    load_session, save_session, build_context_string,
    merge_entities, compute_missing_fields,
)
from app.integrations.supabase_client import get_pool
from app.routes import clarify, direct, rag, sql_route, workflow_route

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/conversation", response_model=ConversationResponse)
async def conversation(req: ConversationRequest, request: Request):
    trace_id = str(uuid.uuid4())
    pool     = await get_pool()

    async with pool.acquire() as db:
        # ── 1. Lock atômico ───────────────────────────────────────────────
        # Garante que a linha existe antes de tentar o lock (upsert).
        await db.execute(
            """
            INSERT INTO n8n_status_atendimento (session_id, cliente_id)
            VALUES ($1, $2)
            ON CONFLICT (session_id, cliente_id) DO NOTHING
            """,
            req.session_id,
            req.cliente_id,
        )
        locked = await db.fetchval(
            "SELECT acquire_conversation_lock($1, $2, $3)",
            req.cliente_id,
            req.session_id,
            "conversation-service",
        )
        if not locked:
            return ConversationResponse(
                messages=[], action="none",
                new_state=ConversationState.TRIAGEM,
                session_id=req.session_id,
                cliente_id=req.cliente_id,
                trace_id=trace_id,
            )

        messages:    list[OutboundMessage] = []
        action       = "send"
        handoff      = None
        novo_estado  = ConversationState.TRIAGEM   # fallback seguro

        try:
            # ── 2a. Configuração da clínica (transbordo, fallback) ────────
            clinica_cfg = await db.fetchrow(
                """
                SELECT transbordo_humano_ativo,
                       mensagem_transbordo,
                       mensagem_fallback_sem_humano
                FROM configuracoes_clinica
                WHERE id = $1
                """,
                req.cliente_id,
            )
            transbordo_ativo = bool(clinica_cfg["transbordo_humano_ativo"]) if clinica_cfg else False
            msg_transbordo   = (clinica_cfg["mensagem_transbordo"] or "") if clinica_cfg else ""
            msg_fallback_sem_humano = (
                clinica_cfg["mensagem_fallback_sem_humano"]
                or "No momento não temos atendentes disponíveis. Posso continuar te ajudando aqui."
            ) if clinica_cfg else "No momento não temos atendentes disponíveis. Posso continuar te ajudando aqui."

            # ── 2b. Carregar sessão ───────────────────────────────────────
            ctx = await load_session(req.session_id, req.cliente_id, db)
            novo_estado = ctx.estado_atual

            # ── 3. Pre-parser (sem LLM) ───────────────────────────────────
            parsed = pre_parse(req.message, ctx.estado_atual, req.media_type)

            # ── 4. Semantic parse (LLM) ───────────────────────────────────
            if parsed is None:
                context_str = build_context_string(ctx)
                try:
                    parsed = await semantic_parse(
                        message=req.message,
                        context=context_str,
                        cliente_info=req.cliente_id,
                        media_type=req.media_type,
                    )
                except ParseError as e:
                    logger.warning("parse_error trace=%s err=%s", trace_id, e)
                    parsed = ParsedIntent(
                        intent=IntentType.DUVIDA,
                        confidence=0.0,
                        entities=EntitySet(),
                        risk_level="low",
                        needs_clarification=True,
                    )

            # ── 5. Alias lookup (convenio → convenio_canonico) ───────────
            parsed.entities = await canonicalize_entities(
                parsed.entities, req.cliente_id, db
            )

            # ── 6. Merge de entidades (Fix 1 + 5) ────────────────────────
            # Acumula entidades entre requests sem sobrescrever com None.
            # Persiste no ctx para o save_session.
            ctx.entities_coletadas = merge_entities(ctx.entities_coletadas, parsed.entities)
            # Usa as entidades mescladas daqui para frente
            parsed.entities = ctx.entities_coletadas

            # ── 7. Dados faltantes (Fix 4) ────────────────────────────────
            # Cálculo real baseado no intent + entidades já coletadas.
            # Alimenta o clarify para pedir 1 campo por vez.
            ctx.dados_faltantes = compute_missing_fields(parsed, ctx.entities_coletadas)

            # ── 8. Policy engine ──────────────────────────────────────────
            decision = decide_route(parsed, ctx.estado_atual)

            # ── 9. Executor da rota ───────────────────────────────────────
            if decision.route == "direct":
                messages = direct.build_direct_response(parsed)

            elif decision.route == "clarify":
                messages = clarify.build_clarify_response(parsed, ctx.dados_faltantes)
                # Se o clarify foi para coletar dados de intent transacional,
                # avança para coletando_dados para que o próximo turno seja processado
                # pelo workflow (não pelo clarify novamente).
                _transactional_collecting = {
                    IntentType.AGENDAR, IntentType.REMARCAR,
                    IntentType.CANCELAR, IntentType.CONFIRMAR, IntentType.FILA,
                }
                _missing = compute_missing_fields(parsed, parsed.entities)
                if parsed.intent in _transactional_collecting and _missing:
                    novo_estado = ConversationState.COLETANDO_DADOS

            elif decision.route == "rag":
                messages = await rag.execute_rag(
                    parsed, decision.filters, req.cliente_id, db,
                    session_id=req.session_id, query=req.message,
                )

            elif decision.route == "sql":
                messages = await sql_route.execute_sql(parsed, req.cliente_id, db)

            elif decision.route == "hybrid":
                sql_msgs = await sql_route.execute_sql(parsed, req.cliente_id, db)
                rag_msgs = await rag.execute_rag(
                    parsed, decision.filters, req.cliente_id, db,
                    session_id=req.session_id, query=req.message,
                )
                messages = sql_msgs + rag_msgs

            elif decision.route == "workflow":
                gt_inova = getattr(request.app.state, "gt_inova", None)
                messages, next_state_val = await workflow_route.execute_workflow(
                    parsed, ctx.estado_atual, ctx.dados_faltantes,
                    req.cliente_id, req.session_id, db, gt_inova,
                )
                if next_state_val:
                    novo_estado = resolve_next_state(
                        ctx.estado_atual,
                        parsed.intent,
                        ConversationState(next_state_val),
                    )

            # ── 10. Resolve estado final ──────────────────────────────────
            # workflow e clarify-transacional já definiram novo_estado acima.
            if decision.route not in ("workflow", "clarify"):
                novo_estado = resolve_next_state(ctx.estado_atual, parsed.intent)

            # ── 10b. Resolve action para o N8N (decisão por clínica) ─────
            # action é o contrato externo com o N8N.
            # Handoff só ocorre se a clínica tem transbordo_humano_ativo=true.
            if novo_estado == ConversationState.TRANSBORDO:
                if transbordo_ativo:
                    action = "handoff"
                    if msg_transbordo:
                        messages = [OutboundMessage(text=msg_transbordo)]
                else:
                    # Clínica 100% IA: não avança para TRANSBORDO, usa fallback
                    action = "send"
                    messages = [OutboundMessage(text=msg_fallback_sem_humano)]
                    novo_estado = ctx.estado_atual
            else:
                action = "send"

            # ── 11. Salvar sessão ─────────────────────────────────────────
            resposta_texto = " ".join(m.text for m in messages)
            await save_session(ctx, db, novo_estado, resposta_texto, req.message)

        except Exception as e:
            logger.error("conversation_error trace=%s err=%s", trace_id, e, exc_info=True)
            messages = [OutboundMessage(
                text="Ocorreu um erro interno. Tente novamente em instantes."
            )]

        finally:
            # ── 12. Liberar lock (Fix 3) ──────────────────────────────────
            # fetchval em vez de execute; log se falhar — nunca propaga exceção.
            try:
                await db.fetchval(
                    "SELECT release_conversation_lock($1, $2, $3)",
                    req.cliente_id,
                    req.session_id,
                    "conversation-service",
                )
            except Exception as e:
                logger.error(
                    "lock_release_failed trace=%s session=%s err=%s",
                    trace_id, req.session_id, e,
                )

    return ConversationResponse(
        messages=messages,
        action=action,
        handoff_data=handoff,
        new_state=novo_estado,
        session_id=req.session_id,
        cliente_id=req.cliente_id,
        trace_id=trace_id,
    )
