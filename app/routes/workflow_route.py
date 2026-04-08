"""
Rota workflow — executor de agendamento determinístico (Fase 3).

Zero chamadas ao LLM. Cada decisão é baseada em:
  - intent (o que o paciente quer)
  - estado_atual (em que ponto do fluxo está)
  - dados_faltantes (o que ainda precisa coletar)
  - resposta da API (sucesso ou código de erro específico)

Tratamento de erros: error.message da GT Inova usado diretamente,
sem reformatação — a API já formata para WhatsApp.
"""
from __future__ import annotations
import json
import logging
import uuid
import asyncpg

from app.models.intent import IntentType, ParsedIntent, EntitySet
from app.models.state import ConversationState
from app.models.conversation import OutboundMessage
from app.integrations.gt_inova import GTInovaClient, GTInovaOk, GTInovaError

logger = logging.getLogger(__name__)

# Resposta quando API está indisponível
_API_UNAVAILABLE = (
    "O sistema de agendamento está temporariamente indisponível. "
    "Por favor, tente novamente em alguns minutos ou ligue para a recepção."
)

# Perguntas de confirmação por intent
_CONFIRM_QUESTION = "Confirme com SIM para prosseguir ou NÃO para cancelar."


# ── Helpers de formatação ─────────────────────────────────────────────────────

def _format_confirmation(intent: IntentType, e: EntitySet) -> str:
    """Monta resumo de confirmação baseado no intent e entidades coletadas."""
    if intent == IntentType.AGENDAR:
        lines = ["Confirme os dados do agendamento:"]
        if e.medico_nome:
            lines.append(f"Médico: {e.medico_nome}")
        if e.atendimento_nome:
            lines.append(f"Tipo: {e.atendimento_nome}")
        if e.data_preferida:
            lines.append(f"Data: {e.data_preferida}")
        if e.hora_consulta:
            lines.append(f"Horário: {e.hora_consulta}")
        if e.periodo and not e.hora_consulta:
            lines.append(f"Período: {e.periodo.capitalize()}")
        if e.convenio_canonico or e.convenio:
            lines.append(f"Convênio: {e.convenio_canonico or e.convenio}")
        lines.append("")
        lines.append(_CONFIRM_QUESTION)
        return "\n".join(lines)

    if intent == IntentType.REMARCAR:
        lines = ["Confirme a remarcação:"]
        if e.agendamento_id:
            lines.append(f"Agendamento: {e.agendamento_id}")
        if e.data_preferida:
            lines.append(f"Nova data preferida: {e.data_preferida}")
        lines.append("")
        lines.append(_CONFIRM_QUESTION)
        return "\n".join(lines)

    if intent == IntentType.CANCELAR:
        lines = ["Confirme o cancelamento:"]
        if e.agendamento_id:
            lines.append(f"Agendamento: {e.agendamento_id}")
        lines.append("")
        lines.append(_CONFIRM_QUESTION)
        return "\n".join(lines)

    return _CONFIRM_QUESTION


def _format_availability(data: dict) -> str:
    """Formata slots disponíveis retornados pela API."""
    # A API já retorna `message` formatado — usar se disponível
    if "message" in data:
        return data["message"]
    slots = data.get("slots", [])
    if not slots:
        return "Não há vagas disponíveis no momento. Deseja entrar na lista de espera?"
    lines = ["Vagas disponíveis:"]
    for s in slots[:5]:   # máximo 5 opções
        lines.append(f"  {s}")
    return "\n".join(lines)


# ── Logging em workflow_runs ──────────────────────────────────────────────────

async def _log_workflow_run(
    db:             asyncpg.Connection,
    session_id:     str,
    cliente_id:     str,
    intent:         str,
    step:           str,
    status:         str,         # "running" | "completed" | "failed" | "waiting_confirm"
    agendamento_id: str | None,
    payload:        dict | None,
    error_code:     str | None,
) -> None:
    try:
        await db.execute(
            """
            INSERT INTO workflow_runs (
                id, session_id, cliente_id,
                intent, current_step, status,
                agendamento_id, payload, error_code,
                created_at, updated_at
            ) VALUES (
                $1, $2, $3,
                $4, $5, $6,
                $7, $8::jsonb, $9,
                NOW(), NOW()
            )
            ON CONFLICT (session_id, cliente_id)
            DO UPDATE SET
                current_step   = EXCLUDED.current_step,
                status         = EXCLUDED.status,
                agendamento_id = COALESCE(EXCLUDED.agendamento_id, workflow_runs.agendamento_id),
                payload        = EXCLUDED.payload,
                error_code     = EXCLUDED.error_code,
                updated_at     = NOW()
            """,
            str(uuid.uuid4()),
            session_id, cliente_id,
            intent, step, status,
            agendamento_id,
            json.dumps(payload or {}, ensure_ascii=False),
            error_code,
        )
    except Exception as e:
        logger.warning("workflow_run_log_failed session=%s err=%s", session_id, e)


# ── Executor principal ────────────────────────────────────────────────────────

async def execute_workflow(
    parsed:          ParsedIntent,
    estado_atual:    ConversationState,
    dados_faltantes: list[str],
    cliente_id:      str,
    session_id:      str,
    db:              asyncpg.Connection,
    gt_inova:        GTInovaClient | None,
) -> tuple[list[OutboundMessage], str | None]:
    """
    Executa o intent transacional de forma determinística.

    Retorna (messages, próximo_estado | None).
    None = manter o estado atual.
    """
    intent  = parsed.intent
    entities = parsed.entities

    # ── TRANSBORDO ────────────────────────────────────────────────────────────
    if intent == IntentType.TRANSBORDO:
        return (
            [OutboundMessage(text="Vou te transferir para um de nossos atendentes. Um momento! 😊")],
            ConversationState.TRANSBORDO.value,
        )

    # ── CONFIRMAR em estado CONFIRMANDO → executar o agendamento pendente ────
    # "SIM" pode ser classificado como CONFIRMAR ou RESPOSTA_FILA dependendo do LLM.
    # Em ambos os casos, se estiver em CONFIRMANDO, o correto é chamar /schedule.
    if estado_atual == ConversationState.CONFIRMANDO and intent in (
        IntentType.CONFIRMAR,
        IntentType.RESPOSTA_FILA,
    ):
        return await _execute_schedule(entities, cliente_id, session_id, db, gt_inova)

    # ── RESPOSTA_FILA ─────────────────────────────────────────────────────────
    if intent == IntentType.RESPOSTA_FILA:
        return await _handle_resposta_fila(
            entities, estado_atual, cliente_id, session_id, db, gt_inova
        )

    # ── FILA (adicionar) ──────────────────────────────────────────────────────
    if intent == IntentType.FILA:
        return await _handle_fila(
            entities, dados_faltantes, cliente_id, session_id, db, gt_inova
        )

    # ── AGENDAR ───────────────────────────────────────────────────────────────
    if intent == IntentType.AGENDAR:
        return await _handle_agendar(
            entities, estado_atual, dados_faltantes, cliente_id, session_id, db, gt_inova
        )

    # ── REMARCAR ──────────────────────────────────────────────────────────────
    if intent == IntentType.REMARCAR:
        return await _handle_remarcar(
            entities, estado_atual, dados_faltantes, cliente_id, session_id, db, gt_inova
        )

    # ── CANCELAR ──────────────────────────────────────────────────────────────
    if intent == IntentType.CANCELAR:
        return await _handle_cancelar(
            entities, estado_atual, dados_faltantes, cliente_id, session_id, db, gt_inova
        )

    # ── CONFIRMAR (de agendamento existente, ex: reminder) ───────────────────
    if intent == IntentType.CONFIRMAR:
        return await _handle_confirmar(
            entities, dados_faltantes, cliente_id, session_id, db, gt_inova
        )

    # Fallback — não deveria chegar aqui
    return (
        [OutboundMessage(text="Não entendi sua solicitação. Pode repetir?")],
        None,
    )


# ── Handlers por intent ───────────────────────────────────────────────────────

async def _handle_agendar(
    entities:        EntitySet,
    estado_atual:    ConversationState,
    dados_faltantes: list[str],
    cliente_id:      str,
    session_id:      str,
    db:              asyncpg.Connection,
    gt_inova:        GTInovaClient | None,
) -> tuple[list[OutboundMessage], str | None]:

    # 3.2 — Coleta progressiva: pede 1 campo por vez
    if dados_faltantes:
        from app.routes.clarify import QUESTIONS, AMBIGUOUS_RESPONSE, COLLECTION_ORDER
        for field in COLLECTION_ORDER:
            if field in dados_faltantes:
                return ([OutboundMessage(text=QUESTIONS[field])],
                        ConversationState.COLETANDO_DADOS.value)
        return ([OutboundMessage(text=AMBIGUOUS_RESPONSE)], None)

    # Dados completos + sem slot escolhido → buscar disponibilidade primeiro
    if (estado_atual in (ConversationState.COLETANDO_DADOS, ConversationState.TRIAGEM)
            and not entities.hora_consulta):
        return await _handle_offer_availability(entities, cliente_id, session_id, db, gt_inova)

    # Dados completos + slot escolhido → mostrar confirmação
    if estado_atual == ConversationState.COLETANDO_DADOS or \
       estado_atual == ConversationState.TRIAGEM:
        return (
            [OutboundMessage(text=_format_confirmation(IntentType.AGENDAR, entities))],
            ConversationState.CONFIRMANDO.value,
        )

    # Paciente está em CONFIRMANDO — re-confirmar (paciente mandou outra mensagem)
    if estado_atual == ConversationState.CONFIRMANDO and not _is_confirming(entities):
        return (
            [OutboundMessage(text=_format_confirmation(IntentType.AGENDAR, entities))],
            None,  # permanece em CONFIRMANDO
        )

    # Paciente confirmou (intent CONFIRMAR ou estado CONFIRMANDO com afirmação)
    if estado_atual == ConversationState.CONFIRMANDO:
        return await _execute_schedule(entities, cliente_id, session_id, db, gt_inova)

    return ([OutboundMessage(text=_format_confirmation(IntentType.AGENDAR, entities))],
            ConversationState.CONFIRMANDO.value)


def _is_confirming(entities: EntitySet) -> bool:
    """True quando o paciente confirmou explicitamente (resposta_fila SIM ou intent CONFIRMAR)."""
    return entities.resposta_fila == "SIM"


async def _execute_schedule(
    entities:   EntitySet,
    cliente_id: str,
    session_id: str,
    db:         asyncpg.Connection,
    gt_inova:   GTInovaClient | None,
) -> tuple[list[OutboundMessage], str | None]:
    """Chama /schedule e trata todos os códigos de erro."""
    await _log_workflow_run(db, session_id, cliente_id, "agendar", "calling_api",
                            "running", None, None, None)

    if gt_inova is None:
        return ([OutboundMessage(text=_API_UNAVAILABLE)],
                ConversationState.TRIAGEM.value)

    result = await gt_inova.schedule(
        medico_nome      = entities.medico_nome or "",
        atendimento_nome = entities.atendimento_nome or "",
        data_preferida   = entities.data_preferida or "",
        convenio         = entities.convenio_canonico or entities.convenio or "",
        cliente_id       = cliente_id,
        paciente_nome    = entities.paciente_nome,
        paciente_celular = entities.paciente_celular,
        data_nascimento  = entities.data_nascimento,
        periodo          = entities.periodo,
    )

    if isinstance(result, GTInovaOk):
        agendamento_id = result.data.get("agendamento_id")
        msg = result.data.get("message", "Agendamento realizado com sucesso! ✅")
        await _log_workflow_run(db, session_id, cliente_id, "agendar", "completed",
                                "completed", agendamento_id, result.data, None)
        return ([OutboundMessage(text=msg)], ConversationState.CONCLUIDO.value)

    # 3.4 — error.message diretamente, sem reformatar
    error: GTInovaError = result
    await _log_workflow_run(db, session_id, cliente_id, "agendar", "api_error",
                            "failed", None, None, error.error_code)

    if error.error_code == "SLOT_TAKEN":
        # 3.3 — Oferecer novas datas via /availability
        return await _handle_slot_taken(entities, cliente_id, session_id, db, gt_inova,
                                        error.message)

    if error.error_code == "DUPLICATE_BOOKING":
        return await _handle_duplicate(entities, cliente_id, gt_inova, error.message)

    # Todos os outros erros: usar error.message direto
    return ([OutboundMessage(text=error.message)], ConversationState.TRIAGEM.value)


async def _handle_offer_availability(
    entities:   EntitySet,
    cliente_id: str,
    session_id: str,
    db:         asyncpg.Connection,
    gt_inova:   GTInovaClient | None,
) -> tuple[list[OutboundMessage], str | None]:
    """
    Chama /availability, mostra disponibilidade e vai direto para CONFIRMANDO.

    A GT Inova usa ordem de chegada (sem slot fixo): retorna data real e hora_inicio
    do período disponível. Capturamos esses valores em entities para usar no /schedule.
    """
    if gt_inova is None:
        return (
            [OutboundMessage(text=_format_confirmation(IntentType.AGENDAR, entities))],
            ConversationState.CONFIRMANDO.value,
        )

    avail = await gt_inova.get_availability(
        medico_nome      = entities.medico_nome or "",
        atendimento_nome = entities.atendimento_nome or "",
        cliente_id       = cliente_id,
        periodo          = entities.periodo,
    )

    if isinstance(avail, GTInovaError):
        logger.warning("availability_failed session=%s err=%s", session_id, avail.error_code)
        # Pula disponibilidade: vai para confirmação com os dados que temos
        return (
            [OutboundMessage(text=_format_confirmation(IntentType.AGENDAR, entities))],
            ConversationState.CONFIRMANDO.value,
        )

    # Extrai data real e hora_inicio do primeiro período disponível
    data_real = avail.data.get("data") or entities.data_preferida
    hora_real = None
    for periodo in avail.data.get("periodos", []):
        if periodo.get("disponivel") and periodo.get("vagas_disponiveis", 0) > 0:
            hora_real = periodo.get("hora_inicio")
            break

    # Atualiza entities com dados reais (persiste via save_session no caller)
    if data_real:
        entities.data_preferida = data_real
    if hora_real:
        entities.hora_consulta = hora_real

    # Mensagem de disponibilidade já formatada para WhatsApp
    avail_text = avail.data.get("message") or avail.data.get("mensagem_whatsapp") or _format_availability(avail.data)

    # Exibe disponibilidade + confirmação em sequência
    messages = [
        OutboundMessage(text=avail_text),
        OutboundMessage(text=_format_confirmation(IntentType.AGENDAR, entities), delay_ms=1200),
    ]
    return (messages, ConversationState.CONFIRMANDO.value)


async def _handle_slot_taken(
    entities:   EntitySet,
    cliente_id: str,
    session_id: str,
    db:         asyncpg.Connection,
    gt_inova:   GTInovaClient | None,
    slot_msg:   str,
) -> tuple[list[OutboundMessage], str | None]:
    """3.3 — SLOT_TAKEN: busca disponibilidade e oferece novas datas."""
    messages = [OutboundMessage(text=slot_msg)]

    if gt_inova:
        avail = await gt_inova.get_availability(
            medico_nome      = entities.medico_nome or "",
            atendimento_nome = entities.atendimento_nome or "",
            cliente_id       = cliente_id,
            periodo          = entities.periodo,
        )
        if isinstance(avail, GTInovaOk):
            messages.append(OutboundMessage(
                text=_format_availability(avail.data),
                delay_ms=1000,
            ))

    # Volta para coletando_dados — data_preferida precisa ser recoletada
    return (messages, ConversationState.COLETANDO_DADOS.value)


async def _handle_duplicate(
    entities:   EntitySet,
    cliente_id: str,
    gt_inova:   GTInovaClient | None,
    error_msg:  str,
) -> tuple[list[OutboundMessage], str | None]:
    """DUPLICATE_BOOKING: mostra agendamento existente e pergunta se quer remarcar."""
    messages = [OutboundMessage(text=error_msg)]
    if gt_inova and entities.paciente_celular:
        check = await gt_inova.check_patient(entities.paciente_celular, cliente_id)
        if isinstance(check, GTInovaOk) and "message" in check.data:
            messages.append(OutboundMessage(
                text=check.data["message"] + "\n\nDeseja remarcar esse agendamento?",
                delay_ms=800,
            ))
    return (messages, ConversationState.TRIAGEM.value)


async def _handle_remarcar(
    entities:        EntitySet,
    estado_atual:    ConversationState,
    dados_faltantes: list[str],
    cliente_id:      str,
    session_id:      str,
    db:              asyncpg.Connection,
    gt_inova:        GTInovaClient | None,
) -> tuple[list[OutboundMessage], str | None]:
    if dados_faltantes:
        from app.routes.clarify import QUESTIONS, COLLECTION_ORDER
        for f in COLLECTION_ORDER:
            if f in dados_faltantes:
                return ([OutboundMessage(text=QUESTIONS[f])],
                        ConversationState.COLETANDO_DADOS.value)

    if estado_atual in (ConversationState.COLETANDO_DADOS, ConversationState.TRIAGEM):
        return ([OutboundMessage(text=_format_confirmation(IntentType.REMARCAR, entities))],
                ConversationState.CONFIRMANDO.value)

    if estado_atual == ConversationState.CONFIRMANDO:
        if gt_inova is None:
            return ([OutboundMessage(text=_API_UNAVAILABLE)], ConversationState.TRIAGEM.value)
        result = await gt_inova.reschedule(
            agendamento_id = entities.agendamento_id or "",
            data_preferida = entities.data_preferida or "",
            cliente_id     = cliente_id,
        )
        if isinstance(result, GTInovaOk):
            msg = result.data.get("message", "Consulta remarcada com sucesso! ✅")
            return ([OutboundMessage(text=msg)], ConversationState.CONCLUIDO.value)
        error: GTInovaError = result
        if error.error_code == "SLOT_TAKEN":
            return await _handle_slot_taken(entities, cliente_id, session_id, db, gt_inova, error.message)
        return ([OutboundMessage(text=error.message)], ConversationState.TRIAGEM.value)

    return ([OutboundMessage(text=_format_confirmation(IntentType.REMARCAR, entities))],
            ConversationState.CONFIRMANDO.value)


async def _handle_cancelar(
    entities:        EntitySet,
    estado_atual:    ConversationState,
    dados_faltantes: list[str],
    cliente_id:      str,
    session_id:      str,
    db:              asyncpg.Connection,
    gt_inova:        GTInovaClient | None,
) -> tuple[list[OutboundMessage], str | None]:
    if dados_faltantes:
        # agendamento_id faltando: listar agendamentos para o paciente escolher
        if "agendamento_id" in dados_faltantes and gt_inova and entities.paciente_celular:
            result = await gt_inova.list_appointments(entities.paciente_celular, cliente_id)
            if isinstance(result, GTInovaOk):
                msg = result.data.get("message", "Qual agendamento deseja cancelar?")
                return ([OutboundMessage(text=msg)], ConversationState.COLETANDO_DADOS.value)
        return ([OutboundMessage(text="Qual é o número do agendamento que deseja cancelar?")],
                ConversationState.COLETANDO_DADOS.value)

    if estado_atual in (ConversationState.COLETANDO_DADOS, ConversationState.TRIAGEM):
        return ([OutboundMessage(text=_format_confirmation(IntentType.CANCELAR, entities))],
                ConversationState.CONFIRMANDO.value)

    if estado_atual == ConversationState.CONFIRMANDO:
        if gt_inova is None:
            return ([OutboundMessage(text=_API_UNAVAILABLE)], ConversationState.TRIAGEM.value)
        result = await gt_inova.cancel(
            agendamento_id = entities.agendamento_id or "",
            cliente_id     = cliente_id,
        )
        if isinstance(result, GTInovaOk):
            msg = result.data.get("message", "Consulta cancelada com sucesso.")
            return ([OutboundMessage(text=msg)], ConversationState.CONCLUIDO.value)
        error: GTInovaError = result
        if error.error_code == "INVALID_STATUS_TRANSITION":
            return ([OutboundMessage(text=error.message)], ConversationState.TRIAGEM.value)
        return ([OutboundMessage(text=error.message)], ConversationState.TRIAGEM.value)

    return ([OutboundMessage(text=_format_confirmation(IntentType.CANCELAR, entities))],
            ConversationState.CONFIRMANDO.value)


async def _handle_confirmar(
    entities:        EntitySet,
    dados_faltantes: list[str],
    cliente_id:      str,
    session_id:      str,
    db:              asyncpg.Connection,
    gt_inova:        GTInovaClient | None,
) -> tuple[list[OutboundMessage], str | None]:
    """Confirmação de agendamento existente (ex: reminder da GT Inova)."""
    if not entities.agendamento_id:
        if gt_inova and entities.paciente_celular:
            result = await gt_inova.list_appointments(entities.paciente_celular, cliente_id)
            if isinstance(result, GTInovaOk):
                return ([OutboundMessage(text=result.data.get("message", "Qual agendamento deseja confirmar?"))],
                        ConversationState.COLETANDO_DADOS.value)
        return ([OutboundMessage(text="Qual é o número do agendamento que deseja confirmar?")],
                ConversationState.COLETANDO_DADOS.value)

    if gt_inova is None:
        return ([OutboundMessage(text=_API_UNAVAILABLE)], ConversationState.TRIAGEM.value)

    result = await gt_inova.confirm(
        agendamento_id = entities.agendamento_id,
        cliente_id     = cliente_id,
    )
    if isinstance(result, GTInovaOk):
        msg = result.data.get("message", "Agendamento confirmado! ✅")
        return ([OutboundMessage(text=msg)], ConversationState.CONCLUIDO.value)
    error: GTInovaError = result
    return ([OutboundMessage(text=error.message)], ConversationState.TRIAGEM.value)


async def _handle_fila(
    entities:        EntitySet,
    dados_faltantes: list[str],
    cliente_id:      str,
    session_id:      str,
    db:              asyncpg.Connection,
    gt_inova:        GTInovaClient | None,
) -> tuple[list[OutboundMessage], str | None]:
    if dados_faltantes:
        from app.routes.clarify import QUESTIONS, COLLECTION_ORDER
        for f in COLLECTION_ORDER:
            if f in dados_faltantes:
                return ([OutboundMessage(text=QUESTIONS[f])],
                        ConversationState.COLETANDO_DADOS.value)

    if gt_inova is None:
        return ([OutboundMessage(text=_API_UNAVAILABLE)], ConversationState.TRIAGEM.value)

    result = await gt_inova.adicionar_fila(
        medico_nome      = entities.medico_nome or "",
        atendimento_nome = entities.atendimento_nome or "",
        cliente_id       = cliente_id,
        convenio         = entities.convenio_canonico or entities.convenio,
    )
    if isinstance(result, GTInovaOk):
        fila_id = result.data.get("fila_id")
        msg     = result.data.get("message", "Você foi adicionado à lista de espera! Avisaremos quando houver uma vaga. 🔔")
        # fila_id será salvo em ia_contexto_sessao.fila_id pelo caller
        return ([OutboundMessage(text=msg)], ConversationState.AGUARDANDO_FILA.value)
    error: GTInovaError = result
    return ([OutboundMessage(text=error.message)], ConversationState.TRIAGEM.value)


async def _handle_resposta_fila(
    entities:     EntitySet,
    estado_atual: ConversationState,
    cliente_id:   str,
    session_id:   str,
    db:           asyncpg.Connection,
    gt_inova:     GTInovaClient | None,
) -> tuple[list[OutboundMessage], str | None]:
    """3.5 — RESPOSTA_FILA: SIM → /responder-fila | NÃO → volta para triagem."""
    fila_id = await db.fetchval(
        "SELECT fila_id FROM ia_contexto_sessao WHERE session_id=$1 AND cliente_id=$2",
        session_id, cliente_id,
    )

    if entities.resposta_fila == "NAO":
        return (
            [OutboundMessage(text="Tudo bem! Se precisar de mais alguma coisa, estou aqui. 😊")],
            ConversationState.TRIAGEM.value,
        )

    # SIM
    if not fila_id:
        return (
            [OutboundMessage(text="Não encontrei sua posição na fila. Entre em contato com a recepção.")],
            ConversationState.TRIAGEM.value,
        )

    if gt_inova is None:
        return ([OutboundMessage(text=_API_UNAVAILABLE)], ConversationState.TRIAGEM.value)

    result = await gt_inova.responder_fila(
        fila_id    = str(fila_id),
        resposta   = "SIM",
        cliente_id = cliente_id,
    )
    if isinstance(result, GTInovaOk):
        agendamento_id = result.data.get("agendamento_id")
        msg = result.data.get("message", "Vaga confirmada! Seu agendamento foi realizado. ✅")
        await _log_workflow_run(db, session_id, cliente_id, "resposta_fila", "completed",
                                "completed", agendamento_id, result.data, None)
        return ([OutboundMessage(text=msg)], ConversationState.CONCLUIDO.value)

    error: GTInovaError = result
    return ([OutboundMessage(text=error.message)], ConversationState.TRIAGEM.value)
