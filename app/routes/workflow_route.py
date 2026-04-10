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
from dataclasses import dataclass
import json
import logging
import uuid
import asyncpg

from app.core.pre_parser import normalize_text
from app.core.session import resolve_rag_ids
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


@dataclass
class CombinationDecision:
    action: str = "none"              # none | allow | deny
    message: str | None = None
    decision_source: str | None = None
    rule_table: str | None = None
    rule_id: str | None = None
    notes: str | None = None


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


def _doctor_name_matches(expected: str | None, actual: str | None) -> bool:
    if not expected or not actual:
        return False
    expected_norm = normalize_text(expected)
    actual_norm = normalize_text(actual)
    return expected_norm in actual_norm or actual_norm in expected_norm


def _service_name_matches(expected: str | None, actual: str | None) -> bool:
    if not expected or not actual:
        return False
    expected_norm = normalize_text(expected)
    actual_norm = normalize_text(actual)
    return expected_norm in actual_norm or actual_norm in expected_norm


def _extract_schedule_doctor(payload: dict, medico_nome: str | None) -> dict | None:
    medicos = payload.get("medicos") or []
    if not medicos:
        return None
    if not medico_nome:
        return medicos[0]
    for doctor in medicos:
        if _doctor_name_matches(medico_nome, doctor.get("nome")):
            return doctor
    return medicos[0]


def _doctor_accepts_convenio(doctor: dict, convenio: str) -> bool:
    convenios = doctor.get("convenios_aceitos") or []
    convenio_norm = normalize_text(convenio)
    return any(normalize_text(str(item)) == convenio_norm for item in convenios)


def _doctor_has_service(doctor: dict, atendimento_nome: str) -> bool:
    servicos = doctor.get("servicos") or []
    for item in servicos:
        if isinstance(item, str) and _service_name_matches(atendimento_nome, item):
            return True
        if isinstance(item, dict) and _service_name_matches(atendimento_nome, item.get("nome")):
            return True
    return False


def _format_doctor_schedule_summary(doctor: dict) -> str:
    nome = doctor.get("nome") or "O médico"
    servicos = doctor.get("servicos") or []
    if not servicos:
        return (
            f"Consultei a GT Inova agora, mas nao encontrei a grade operacional de {nome}. "
            "Posso verificar disponibilidade se voce me disser o atendimento desejado."
        )

    lines = [f"Segundo a agenda atual da GT Inova, {nome} atende assim:"]
    for servico in servicos[:2]:
        if isinstance(servico, str):
            lines.append(servico)
            continue
        nome_servico = servico.get("nome") or "Atendimento"
        dias = servico.get("dias") or "dias nao informados"
        periodos = servico.get("periodos") or []
        if periodos:
            resumo_periodos = []
            for periodo in periodos[:2]:
                horario = periodo.get("horario") or ""
                limite = periodo.get("limite_pacientes")
                trecho = horario if horario else str(periodo.get("periodo") or "")
                if limite is not None:
                    trecho = f"{trecho} (limite {limite})".strip()
                resumo_periodos.append(trecho.strip())
            lines.append(f"{nome_servico}: {dias}. {'; '.join(p for p in resumo_periodos if p)}")
        else:
            lines.append(f"{nome_servico}: {dias}.")
    return "\n".join(lines)


def _format_doctor_convenios_summary(doctor: dict) -> str:
    nome = doctor.get("nome") or "O médico"
    convenios = doctor.get("convenios_aceitos") or []
    if not convenios:
        return (
            f"Consultei a GT Inova agora, mas nao encontrei convenios ativos para {nome}. "
            "Se quiser, posso verificar outro medico."
        )
    lista = ", ".join(str(item) for item in convenios[:12])
    return f"Segundo a GT Inova agora, {nome} atende pelos seguintes convenios: {lista}."


def _format_doctor_services_summary(doctor: dict) -> str:
    nome = doctor.get("nome") or "O médico"
    servicos = doctor.get("servicos") or []
    nomes: list[str] = []
    for item in servicos:
        if isinstance(item, str):
            nomes.append(item)
        elif isinstance(item, dict) and item.get("nome"):
            nomes.append(str(item["nome"]))
    if not nomes:
        return (
            f"Consultei a GT Inova agora, mas nao encontrei servicos ativos para {nome}. "
            "Se quiser, posso verificar outro medico."
        )
    lista = ", ".join(nomes[:12])
    return f"Segundo a GT Inova agora, {nome} aparece com estes servicos ativos: {lista}."


def _message_asks_for_limit(message_norm: str) -> bool:
    limit_keywords = {
        "limite", "quantidade", "quantos", "cota", "cotas", "pacientes por turno",
        "por turno", "quantos pacientes",
    }
    return any(keyword in message_norm for keyword in limit_keywords)


async def _finalize_operational_query_log(
    db: asyncpg.Connection,
    session_id: str,
    cliente_id: str,
    status: str,
    payload: dict | None = None,
    error_code: str | None = None,
) -> None:
    await _log_workflow_run(
        db=db,
        session_id=session_id,
        cliente_id=cliente_id,
        intent="duvida_operacional",
        step="gt_inova_result",
        status=status,
        agendamento_id=None,
        payload=payload,
        error_code=error_code,
    )


async def _handle_operational_live_question(
    parsed: ParsedIntent,
    cliente_id: str,
    session_id: str,
    db: asyncpg.Connection,
    gt_inova: GTInovaClient | None,
) -> tuple[list[OutboundMessage], str | None]:
    """
    Dúvidas operacionais vivas devem ser respondidas somente após consulta
    à GT Inova.
    """
    entities = parsed.entities
    message_norm = normalize_text(parsed.mensagem_usuario or "")

    await _log_workflow_run(
        db=db,
        session_id=session_id,
        cliente_id=cliente_id,
        intent="duvida_operacional",
        step="consulting_gt_inova",
        status="running",
        agendamento_id=None,
        payload={
            "medico_nome": entities.medico_nome,
            "atendimento_nome": entities.atendimento_nome,
            "convenio": entities.convenio_canonico or entities.convenio,
        },
        error_code=None,
    )

    if gt_inova is None:
        await _finalize_operational_query_log(
            db, session_id, cliente_id, "failed",
            payload={"source": "gt_inova", "detail": "unavailable"},
            error_code="GT_INOVA_UNAVAILABLE",
        )
        return (
            [OutboundMessage(
                text="Preciso consultar a agenda da GT Inova para confirmar essa informacao operacional. Tente novamente em instantes."
            )],
            None,
        )

    schedule_keywords = {"agenda", "disponibilidade", "horario", "horário", "vaga", "vagas", "data", "dia", "dias"}
    convenio_keywords = {"convenio", "convênio", "convenios", "convênios", "aceita", "aceito", "particular", "hgu"}
    service_keywords = {"faz", "realiza", "atende", "servico", "serviço", "procedimento", "procedimentos"}

    if any(keyword in message_norm for keyword in {"vaga", "vagas", "disponibilidade", "horario", "horário"}) and not entities.atendimento_nome:
        await _finalize_operational_query_log(
            db, session_id, cliente_id, "failed",
            payload={"source": "workflow", "detail": "missing_atendimento_for_availability"},
            error_code="MISSING_ATENDIMENTO",
        )
        return ([OutboundMessage(
            text="Para eu consultar a disponibilidade correta na GT Inova, me diga qual atendimento voce quer verificar."
        )], ConversationState.COLETANDO_DADOS.value)

    if (
        entities.medico_nome
        and entities.atendimento_nome
        and any(keyword in message_norm for keyword in schedule_keywords)
    ):
        avail = await gt_inova.get_availability(
            medico_nome=entities.medico_nome,
            atendimento_nome=entities.atendimento_nome,
            cliente_id=cliente_id,
            periodo=entities.periodo,
        )
        if isinstance(avail, GTInovaError):
            await _finalize_operational_query_log(
                db, session_id, cliente_id, "failed",
                payload={"source": "availability"},
                error_code=avail.error_code,
            )
            return ([OutboundMessage(text=avail.message)], None)
        avail_text = avail.data.get("message") or avail.data.get("mensagem_formatada") or _format_availability(avail.data)
        await _finalize_operational_query_log(
            db, session_id, cliente_id, "completed",
            payload={"source": "availability"},
        )
        return ([OutboundMessage(text=avail_text)], None)

    if entities.medico_nome:
        schedules = await gt_inova.doctor_schedules(cliente_id, entities.medico_nome)
        if isinstance(schedules, GTInovaError):
            await _finalize_operational_query_log(
                db, session_id, cliente_id, "failed",
                payload={"source": "doctor_schedules"},
                error_code=schedules.error_code,
            )
            return ([OutboundMessage(text=schedules.message)], None)

        doctor = _extract_schedule_doctor(schedules.data, entities.medico_nome)
        if not doctor:
            await _finalize_operational_query_log(
                db, session_id, cliente_id, "failed",
                payload={"source": "doctor_schedules", "detail": "doctor_not_found"},
                error_code="DOCTOR_NOT_FOUND",
            )
            return ([OutboundMessage(
                text="Consultei a GT Inova agora, mas nao encontrei esse medico na agenda ativa. Posso verificar outro nome para voce?"
            )], None)

        convenio = entities.convenio_canonico or entities.convenio
        if convenio:
            if _message_asks_for_limit(message_norm):
                await _finalize_operational_query_log(
                    db, session_id, cliente_id, "completed",
                    payload={"source": "doctor_schedules", "detail": "convenio_limit_requires_transaction"},
                )
                return ([OutboundMessage(
                    text=f"Segundo a GT Inova agora, {doctor.get('nome') or entities.medico_nome} aparece com o convenio {convenio} ativo. O limite desse convenio por turno e validado na propria GT Inova no momento da disponibilidade ou do agendamento."
                )], None)

            accepts = _doctor_accepts_convenio(doctor, convenio)
            if accepts:
                await _finalize_operational_query_log(
                    db, session_id, cliente_id, "completed",
                    payload={"source": "doctor_schedules", "detail": "convenio_yes"},
                )
                return ([OutboundMessage(
                    text=f"Segundo a GT Inova agora, {doctor.get('nome') or entities.medico_nome} atende pelo convenio {convenio}. Se quiser, posso verificar a disponibilidade."
                )], None)
            await _finalize_operational_query_log(
                db, session_id, cliente_id, "completed",
                payload={"source": "doctor_schedules", "detail": "convenio_no"},
            )
            return ([OutboundMessage(
                text=f"Segundo a GT Inova agora, {doctor.get('nome') or entities.medico_nome} nao aparece atendendo pelo convenio {convenio}. Posso verificar outro medico ou convenio para voce?"
            )], None)

        if any(keyword in message_norm for keyword in convenio_keywords):
            await _finalize_operational_query_log(
                db, session_id, cliente_id, "completed",
                payload={"source": "doctor_schedules", "detail": "convenio_list"},
            )
            return ([OutboundMessage(text=_format_doctor_convenios_summary(doctor))], None)

        if entities.atendimento_nome and any(keyword in message_norm for keyword in service_keywords):
            has_service = _doctor_has_service(doctor, entities.atendimento_nome)
            if has_service:
                await _finalize_operational_query_log(
                    db, session_id, cliente_id, "completed",
                    payload={"source": "doctor_schedules", "detail": "service_yes"},
                )
                return ([OutboundMessage(
                    text=f"Segundo a GT Inova agora, {doctor.get('nome') or entities.medico_nome} realiza {entities.atendimento_nome}. Se quiser, posso consultar a disponibilidade."
                )], None)
            await _finalize_operational_query_log(
                db, session_id, cliente_id, "completed",
                payload={"source": "doctor_schedules", "detail": "service_no"},
            )
            return ([OutboundMessage(
                text=f"Segundo a GT Inova agora, {doctor.get('nome') or entities.medico_nome} nao aparece com {entities.atendimento_nome} ativo na agenda. Posso verificar outro medico ou atendimento para voce?"
            )], None)

        if any(keyword in message_norm for keyword in service_keywords):
            await _finalize_operational_query_log(
                db, session_id, cliente_id, "completed",
                payload={"source": "doctor_schedules", "detail": "service_list"},
            )
            return ([OutboundMessage(text=_format_doctor_services_summary(doctor))], None)

        await _finalize_operational_query_log(
            db, session_id, cliente_id, "completed",
            payload={"source": "doctor_schedules", "detail": "schedule_summary"},
        )
        return ([OutboundMessage(text=_format_doctor_schedule_summary(doctor))], None)

    doctors = await gt_inova.list_doctors(cliente_id)
    if isinstance(doctors, GTInovaError):
        await _finalize_operational_query_log(
            db, session_id, cliente_id, "failed",
            payload={"source": "list_doctors"},
            error_code=doctors.error_code,
        )
        return ([OutboundMessage(text=doctors.message)], None)

    medicos = doctors.data.get("medicos") or []
    convenio = entities.convenio_canonico or entities.convenio
    if convenio:
        matching = [
            doctor.get("nome")
            for doctor in medicos
            if _doctor_accepts_convenio(doctor, convenio)
        ]
        if matching:
            nomes = ", ".join(matching[:5])
            await _finalize_operational_query_log(
                db, session_id, cliente_id, "completed",
                payload={"source": "list_doctors", "detail": "convenio_global_list"},
            )
            return ([OutboundMessage(
                text=f"Segundo a GT Inova agora, o convenio {convenio} aparece ativo para: {nomes}."
            )], None)
        await _finalize_operational_query_log(
            db, session_id, cliente_id, "completed",
            payload={"source": "list_doctors", "detail": "convenio_global_empty"},
        )
        return ([OutboundMessage(
            text=f"Consultei a GT Inova agora e nao encontrei medicos com o convenio {convenio} ativo."
        )], None)

    if entities.atendimento_nome:
        matching = [
            doctor.get("nome")
            for doctor in medicos
            if _doctor_has_service(doctor, entities.atendimento_nome)
        ]
        if matching:
            nomes = ", ".join(matching[:5])
            await _finalize_operational_query_log(
                db, session_id, cliente_id, "completed",
                payload={"source": "list_doctors", "detail": "service_global_list"},
            )
            return ([OutboundMessage(
                text=f"Segundo a GT Inova agora, {entities.atendimento_nome} aparece ativo para: {nomes}."
            )], None)
        await _finalize_operational_query_log(
            db, session_id, cliente_id, "completed",
            payload={"source": "list_doctors", "detail": "service_global_empty"},
        )
        return ([OutboundMessage(
            text=f"Consultei a GT Inova agora e nao encontrei {entities.atendimento_nome} como servico ativo."
        )], None)

    await _finalize_operational_query_log(
        db, session_id, cliente_id, "failed",
        payload={"source": "workflow", "detail": "insufficient_entities"},
        error_code="INSUFFICIENT_ENTITIES",
    )
    return ([OutboundMessage(
        text="Para confirmar essa informacao operacional, preciso do medico, convenio ou atendimento que voce quer consultar."
    )], None)


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
    # Sinaliza TRANSBORDO para o conversation.py — a mensagem e o action
    # são decididos lá, com base em transbordo_humano_ativo da clínica.
    if intent == IntentType.TRANSBORDO:
        return (
            [],  # mensagem composta pelo conversation.py
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

    # ── DÚVIDA operacional viva ──────────────────────────────────────────────
    if intent == IntentType.DUVIDA:
        return await _handle_operational_live_question(
            parsed, cliente_id, session_id, db, gt_inova
        )

    # Fallback — não deveria chegar aqui
    return (
        [OutboundMessage(text="Não entendi sua solicitação. Pode repetir?")],
        None,
    )


# ── Validação de convênio × médico ───────────────────────────────────────────

async def _check_convenio_medico(
    entities:   EntitySet,
    cliente_id: str,
    db:         asyncpg.Connection,
) -> str | None:
    """
    Consulta `convenios_medico` para verificar se o médico atende o convênio.

    Returns:
        None  → combinação válida, sem registro (deixa GT Inova decidir),
                ou dados insuficientes para validar.
        str   → mensagem de erro pronta para WhatsApp.
    """
    convenio = entities.convenio_canonico or entities.convenio
    medico   = entities.medico_nome

    if not convenio or not medico:
        return None

    row = await db.fetchrow(
        """
        SELECT cm.status
        FROM convenios_medico cm
        JOIN medicos m ON m.id = cm.medico_id
        WHERE cm.cliente_id = $1
          AND m.nome ILIKE $2
          AND cm.convenio_nome = $3
        LIMIT 1
        """,
        cliente_id,
        f"%{medico}%",
        convenio,
    )

    if row is None:
        return None  # sem cadastro → GT Inova é a autoridade final

    if row["status"] == "nao_atende":
        return (
            f"O médico solicitado não atende pelo convênio *{convenio}*. "
            "Posso verificar outro médico ou convênio para você?"
        )

    return None  # status == "atende"


async def _evaluate_atendimento_medico(
    entities:   EntitySet,
    cliente_id: str,
    db:         asyncpg.Connection,
) -> CombinationDecision:
    """
    Valida a matriz global de atendimento × médico.
    """
    if not entities.medico_nome or not entities.atendimento_nome:
        return CombinationDecision()

    doctor_id, procedure_id = await resolve_rag_ids(
        medico_nome=entities.medico_nome,
        atendimento_nome=entities.atendimento_nome,
        cliente_id=cliente_id,
        db=db,
    )
    if not doctor_id or not procedure_id:
        return CombinationDecision()

    row = await db.fetchrow(
        """
        SELECT id, rule_action, mensagem_bloqueio, notes
        FROM atendimentos_medico
        WHERE cliente_id   = $1
          AND medico_id    = $2
          AND procedure_id = $3
          AND is_active    = TRUE
          AND (valid_from IS NULL OR valid_from <= NOW())
          AND (valid_to   IS NULL OR valid_to   >= NOW())
        ORDER BY priority ASC, created_at DESC
        LIMIT 1
        """,
        cliente_id,
        doctor_id,
        procedure_id,
    )

    if row is None:
        return CombinationDecision()

    action = row["rule_action"]
    if action == "deny":
        return CombinationDecision(
            action="deny",
            message=row["mensagem_bloqueio"] or (
                "O medico solicitado nao realiza esse atendimento. "
                "Posso verificar outro medico ou atendimento para voce?"
            ),
            decision_source="atendimento_medico",
            rule_table="atendimentos_medico",
            rule_id=str(row["id"]),
            notes=row["notes"],
        )

    return CombinationDecision(
        action="allow",
        decision_source="atendimento_medico",
        rule_table="atendimentos_medico",
        rule_id=str(row["id"]),
        notes=row["notes"],
    )


async def _check_atendimento_medico(
    entities:   EntitySet,
    cliente_id: str,
    db:         asyncpg.Connection,
) -> str | None:
    """
    Wrapper compatível para testes e uso legado.
    """
    return (await _evaluate_atendimento_medico(entities, cliente_id, db)).message


async def _evaluate_convenio_atendimento_medico(
    entities:   EntitySet,
    cliente_id: str,
    db:         asyncpg.Connection,
) -> CombinationDecision:
    """
    Valida exceções da matriz global convênio × atendimento × médico.
    """
    convenio = entities.convenio_canonico or entities.convenio
    if not convenio or not entities.medico_nome or not entities.atendimento_nome:
        return CombinationDecision()

    doctor_id, procedure_id = await resolve_rag_ids(
        medico_nome=entities.medico_nome,
        atendimento_nome=entities.atendimento_nome,
        cliente_id=cliente_id,
        db=db,
    )
    if not doctor_id or not procedure_id:
        return CombinationDecision()  # sem IDs → sem regra → GT Inova decide

    row = await db.fetchrow(
        """
        SELECT id, rule_action, mensagem_bloqueio, notes
        FROM regras_convenio_atendimento_medico
        WHERE cliente_id            = $1
          AND medico_id             = $2
          AND procedure_id          = $3
          AND LOWER(convenio_nome)  = LOWER($4)
          AND is_active             = TRUE
          AND (valid_from IS NULL OR valid_from <= NOW())
          AND (valid_to   IS NULL OR valid_to   >= NOW())
        ORDER BY priority ASC, created_at DESC
        LIMIT 1
        """,
        cliente_id,
        doctor_id,
        procedure_id,
        convenio,
    )

    if row is None:
        return CombinationDecision()  # sem regra cadastrada → GT Inova decide

    action = row["rule_action"]
    if action == "deny":
        return CombinationDecision(
            action="deny",
            message=row["mensagem_bloqueio"] or (
                f"O convenio {convenio} nao permite esse atendimento com o medico solicitado. "
                "Posso verificar outra opcao para voce?"
            ),
            decision_source="convenio_atendimento_medico",
            rule_table="regras_convenio_atendimento_medico",
            rule_id=str(row["id"]),
            notes=row["notes"],
        )

    return CombinationDecision(
        action="allow",
        decision_source="convenio_atendimento_medico",
        rule_table="regras_convenio_atendimento_medico",
        rule_id=str(row["id"]),
        notes=row["notes"],
    )


async def _check_convenio_atendimento_medico(
    entities:   EntitySet,
    cliente_id: str,
    db:         asyncpg.Connection,
) -> str | None:
    """
    Wrapper compatível para testes e uso legado.
    """
    return (await _evaluate_convenio_atendimento_medico(entities, cliente_id, db)).message


async def _precheck_gt_inova(
    entities:   EntitySet,
    cliente_id: str,
    gt_inova:   "GTInovaClient | None",
) -> CombinationDecision:
    """
    Valida convenio e servico contra a GT Inova antes do /availability.

    Substitui os checks locais convenios_medico + atendimentos_medico como
    fonte de bloqueio no fluxo de agendamento. A GT Inova é a autoridade —
    os dados locais eram informativos e podiam divergir.

    Comportamento:
    - GT Inova indisponivel → retorna action='none' (nao bloqueia, deixa /schedule decidir)
    - Medico nao encontrado na agenda → deny com mensagem de confirmação de nome
    - Convenio nao aceito → deny com lista dos convenios aceitos
    - Servico nao ativo → deny com lista dos servicos disponíveis
    - Tudo ok → action='allow'
    """
    from app.integrations.gt_inova import GTInovaError

    if gt_inova is None or not entities.medico_nome:
        return CombinationDecision()

    schedules = await gt_inova.doctor_schedules(cliente_id, entities.medico_nome)
    if isinstance(schedules, GTInovaError):
        # GT Inova indisponivel: nao bloquear — /availability e /schedule decidem
        return CombinationDecision()

    doctor = _extract_schedule_doctor(schedules.data, entities.medico_nome)
    if not doctor:
        return CombinationDecision(
            action="deny",
            message=f"Nao encontrei {entities.medico_nome} na agenda ativa. Pode confirmar o nome do medico?",
            decision_source="gt_inova_precheck",
            rule_table="doctor_schedules",
        )

    convenio = entities.convenio_canonico or entities.convenio
    if convenio and not _doctor_accepts_convenio(doctor, convenio):
        convenios_aceitos = doctor.get("convenios_aceitos") or []
        lista = ", ".join(convenios_aceitos[:6]) if convenios_aceitos else "nenhum cadastrado"
        return CombinationDecision(
            action="deny",
            message=(
                f"{doctor.get('nome') or entities.medico_nome} nao atende pelo convenio {convenio}. "
                f"Convenios aceitos: {lista}. Posso verificar outro convenio ou medico?"
            ),
            decision_source="gt_inova_precheck",
            rule_table="doctor_schedules",
            notes="CONVENIO_NAO_ACEITO via doctor_schedules pre-check",
        )

    if entities.atendimento_nome and not _doctor_has_service(doctor, entities.atendimento_nome):
        raw_servicos = doctor.get("servicos") or []
        nomes = [
            (s if isinstance(s, str) else s.get("nome", ""))
            for s in raw_servicos
        ]
        lista = ", ".join(filter(None, nomes[:5])) if nomes else "nenhum cadastrado"
        return CombinationDecision(
            action="deny",
            message=(
                f"{doctor.get('nome') or entities.medico_nome} nao aparece com "
                f"{entities.atendimento_nome} ativo na agenda. "
                f"Atendimentos disponíveis: {lista}. Posso verificar outra opcao?"
            ),
            decision_source="gt_inova_precheck",
            rule_table="doctor_schedules",
            notes="SERVICO_NAO_ATIVO via doctor_schedules pre-check",
        )

    return CombinationDecision(
        action="allow",
        decision_source="gt_inova_precheck",
        rule_table="doctor_schedules",
    )


async def _log_combination_decision(
    db: asyncpg.Connection,
    session_id: str,
    cliente_id: str,
    decision: CombinationDecision,
) -> None:
    if decision.action == "none" or not decision.decision_source:
        return

    payload = {
        "decision_source": decision.decision_source,
        "rule_table": decision.rule_table,
        "rule_id": decision.rule_id,
        "decision_action": decision.action,
        "notes": decision.notes,
    }
    await _log_workflow_run(
        db=db,
        session_id=session_id,
        cliente_id=cliente_id,
        intent="agendar",
        step="policy_rule",
        status="failed" if decision.action == "deny" else "running",
        agendamento_id=None,
        payload=payload,
        error_code="POLICY_RULE_BLOCKED" if decision.action == "deny" else None,
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

    # Dados completos → validar em ordem:
    #   1. convenio x atendimento x medico  (regra tripla local — o que a API nao sabe)
    #   2. GT Inova precheck via doctor_schedules  (convenio + servico ativos)
    #   3. /availability → /schedule
    if estado_atual in (ConversationState.COLETANDO_DADOS, ConversationState.TRIAGEM):
        decision_tripla = await _evaluate_convenio_atendimento_medico(entities, cliente_id, db)
        await _log_combination_decision(db, session_id, cliente_id, decision_tripla)
        if decision_tripla.action == "deny":
            return ([OutboundMessage(text=decision_tripla.message or "")], ConversationState.TRIAGEM.value)

        # Tripla nao tem regra especifica → pré-checar com GT Inova (fonte de verdade)
        if decision_tripla.action != "allow":
            decision_precheck = await _precheck_gt_inova(entities, cliente_id, gt_inova)
            await _log_combination_decision(db, session_id, cliente_id, decision_precheck)
            if decision_precheck.action == "deny":
                return ([OutboundMessage(text=decision_precheck.message or "")], ConversationState.TRIAGEM.value)

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
