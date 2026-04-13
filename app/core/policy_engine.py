from __future__ import annotations
import re
from dataclasses import dataclass, field
from app.models.intent import IntentType, ParsedIntent, EntitySet
from app.models.state import ConversationState


# ── Conjuntos de intents por categoria ──────────────────────────────────────

TRANSACTIONAL_INTENTS = {
    IntentType.AGENDAR,
    IntentType.REMARCAR,
    IntentType.CANCELAR,
    IntentType.CONFIRMAR,
    IntentType.FILA,
    IntentType.RESPOSTA_FILA,
    IntentType.TRANSBORDO,
}

SOCIAL_INTENTS = {
    IntentType.SOCIAL,
    IntentType.SAUDACAO,
    IntentType.AGRADECIMENTO,
    IntentType.DESPEDIDA,
    IntentType.FORA_ESCOPO,
}

# Intents que buscam explicação clínica → rota rag
EXPLANATORY_INTENTS = {
    IntentType.DUVIDA_PREPARO,
    IntentType.DUVIDA_ORIENTACAO,
    IntentType.DUVIDA_POS_PROCEDIMENTO,
}

# Keywords de localização/contato — respondidas pelo SQL (configuracoes_clinica)
_LOCATION_KEYWORDS = (
    "endereço", "endereco", "localização", "localizacao",
    "onde fica", "como chegar", "horário", "horario",
    "funcionamento", "abre", "fecha", "aberto", "telefone",
    "contato", "número", "numero",
)

_DOCTOR_CONTEXT_PATTERNS = (
    r"\bele\b",
    r"\bela\b",
    r"\bdele\b",
    r"\bdela\b",
    r"\bcom ele\b",
    r"\bcom ela\b",
    r"\besse medico\b",
    r"\bessa medica\b",
    r"\besse doutor\b",
    r"\bessa doutora\b",
)

_DOCTOR_STRUCTURED_FACT_KEYWORDS = (
    "crm",
    "rqe",
)


# ── Modelo de retorno ────────────────────────────────────────────────────────

@dataclass
class RagFilters:
    risk_max:     str = "high"
    source_types: list[str] = field(default_factory=list)
    doctor_id:    str | None = None    # uuid — preenchido por resolve_rag_ids()
    procedure_id: str | None = None   # uuid — preenchido por resolve_rag_ids()


@dataclass
class RouteDecision:
    route:   str                  # sql | rag | workflow | clarify | direct
    reason:  str = ""
    filters: RagFilters | None = None


# ── Policy engine (Python puro — sem I/O, 100% testável) ────────────────────

def decide_route(
    intent:  ParsedIntent,
    state:   ConversationState,
    session_entities: EntitySet | None = None,
) -> RouteDecision:
    """
    Recebe um ParsedIntent validado pelo Pydantic e o estado atual
    e devolve a rota a executar.

    Princípio central:
      - Pergunta OPERACIONAL (tem/aceita/faz/qual agenda) → workflow (GT Inova é autoridade)
      - Pergunta EXPLICATIVA (como/o que é/qual preparo)  → rag (knowledge base aprovado)
      - Dado ESTRUTURADO (médico, endereço, horário)      → sql (banco local)
      - Ambíguo ou baixa confiança                        → clarify

    DEFAULT = clarify. Rota hybrid foi eliminada: SQL retorna dados estruturados de
    contato, RAG retorna conhecimento clínico — misturá-los produz duas mensagens
    contraditórias. Perguntas sobre procedimentos vão direto ao RAG.
    """

    # Regra 0: intents transacionais → workflow antes de qualquer clarify.
    # O workflow tem sua própria state machine para coletar dados faltantes (médico,
    # data, procedimento). Chamar clarify aqui seria uma volta desnecessária.
    # Alta prioridade: o LLM pode setar needs_clarification=True por falta de detalhe,
    # mas isso não significa que o paciente está sendo ambíguo — só que precisa de mais
    # perguntas, o que é responsabilidade do workflow.
    if intent.intent in TRANSACTIONAL_INTENTS:
        return RouteDecision(route="workflow", reason="intent transacional")

    # Regra 1: clarificação — depois de transacionais, antes de tudo mais
    if intent.needs_clarification or intent.confidence < 0.70:
        return RouteDecision(route="clarify", reason="baixa confiança ou ambiguidade")

    # Regra 2: social/saudação/fora_escopo → direct (sem busca)
    if intent.intent in SOCIAL_INTENTS:
        return RouteDecision(route="direct", reason="intent social")

    # Regra 3: emergência → direct imediato (protocolo, sem LLM adicional)
    if intent.intent == IntentType.EMERGENCIA:
        return RouteDecision(route="direct", reason="emergência — protocolo imediato")

    # Regra 4: dúvida sobre dado operacional vivo → workflow via GT Inova.
    # is_operational_query é sinalizado pelo LLM semântico (semantic_parser).
    # Cobre: agenda, disponibilidade, elegibilidade, serviço ativo, lista de médicos.
    if intent.intent == IntentType.DUVIDA and intent.is_operational_query:
        return RouteDecision(
            route="workflow",
            reason="dado operacional vivo via GT Inova",
        )

    # Regra 4b: convenio explicitamente extraído pelo LLM → workflow.
    # GT Inova é a única fonte autorizada para elegibilidade de convênio.
    if intent.intent == IntentType.DUVIDA and intent.entities.convenio:
        return RouteDecision(
            route="workflow",
            reason="convênio detectado — elegibilidade via GT Inova",
        )

    # Regra 4c: localização/contato da clínica → SQL ANTES do fallback operacional.
    # Cobre DUVIDA e EXPLANATORY_INTENTS: o LLM às vezes classifica "qual o horário de
    # funcionamento?" como duvida_orientacao. A verificação de keywords aqui garante a
    # rota correta mesmo com misclassificação de intent.
    # Precede Regra 4d e Regra 5 porque "horário" aparece tanto em "horário de
    # funcionamento" (clínica → SQL) quanto em "horário disponível" (agenda → workflow).
    if (
        intent.intent in ({IntentType.DUVIDA} | EXPLANATORY_INTENTS)
        and any(kw in intent.mensagem_usuario.lower() for kw in _LOCATION_KEYWORDS)
        and not intent.entities.medico_nome
    ):
        return RouteDecision(route="sql", reason="localização/contato da clínica — SQL direto")

    # Regra 4d: fallback de segurança operacional via keywords.
    # Garante que perguntas operacionais que o LLM não sinalizou como is_operational_query
    # (ex: "tem vaga?", "aceita Unimed?") ainda vão para o workflow em vez do RAG.
    # EntitySet.touches_live_operational_context() usa listas curadas de keywords.
    contextual_entities = build_contextual_entities(
        current_turn_entities=intent.entities,
        session_entities=session_entities,
        message=intent.mensagem_usuario,
    )
    if (
        intent.intent == IntentType.DUVIDA
        and contextual_entities.touches_live_operational_context(intent.mensagem_usuario)
    ):
        return RouteDecision(
            route="workflow",
            reason="contexto operacional detectado por keywords — GT Inova",
        )

    # Regra 5: dúvidas explicativas clínicas com intent específico → rag com filtros de risco.
    # DEVE preceder qualquer verificação de atendimento_nome para garantir a rota correta.
    # Se o LLM classificou corretamente (duvida_preparo, duvida_orientacao, etc.), vai aqui.
    #
    # source_types escolhidos para cobrir o que existe na knowledge base da clínica:
    #   duvida_preparo        → exam_prep (protocolos de dilatação, preparo de exames)
    #   duvida_orientacao     → policy + procedure_info + operational_script
    #                           (operational_script tem orientações de exames, prazo, acompanhante)
    #   duvida_pos_procedimento → exam_prep + operational_script
    #                           (post_procedure não tem chunks ainda; estes são os proxies mais
    #                            próximos até que conteúdo pós-procedimento seja adicionado)
    if intent.intent in EXPLANATORY_INTENTS:
        source_map = {
            IntentType.DUVIDA_PREPARO:          ["exam_prep"],
            IntentType.DUVIDA_ORIENTACAO:        ["policy", "procedure_info", "operational_script"],
            IntentType.DUVIDA_POS_PROCEDIMENTO:  ["exam_prep", "operational_script"],
        }
        return RouteDecision(
            route="rag",
            reason=f"dúvida clínica: {intent.intent.value}",
            filters=RagFilters(
                risk_max=intent.risk_level,
                source_types=source_map.get(intent.intent, []),
            ),
        )

    # Regra 6: dúvida sobre procedimento com intent genérico (DUVIDA + atendimento_nome).
    # O LLM classificou como 'duvida' mas extraiu um procedimento → pergunta explicativa.
    # Ex: "o que é fundo de olho?", "como funciona a OCT?", "precisa de preparo?"
    # RAG direto — hybrid foi eliminado pois gerava 2 mensagens (SQL vazio + RAG real).
    if intent.intent == IntentType.DUVIDA and contextual_entities.atendimento_nome:
        return RouteDecision(
            route="rag",
            reason="dúvida explicativa sobre procedimento — RAG direto",
            filters=RagFilters(
                risk_max=intent.risk_level,
                source_types=["exam_prep", "procedure_info"],
            ),
        )

    # Regra 6b: fatos estruturados do médico (CRM/RQE) → SQL local.
    # Mesmo quando o médico vem por anáfora contextual ("crm dele"), a fonte correta
    # continua sendo dado cadastral estruturado, não conhecimento aprovado via RAG.
    if (
        intent.intent == IntentType.DUVIDA
        and contextual_entities.medico_nome
        and any(keyword in intent.mensagem_usuario.lower() for keyword in _DOCTOR_STRUCTURED_FACT_KEYWORDS)
    ):
        return RouteDecision(route="sql", reason="fato estruturado do médico — SQL local")

    # Regra 7: perfil estável do médico (CRM, especialidade, formação) → RAG
    if (
        intent.intent == IntentType.DUVIDA
        and contextual_entities.touches_doctor_profile_context(intent.mensagem_usuario)
    ):
        return RouteDecision(
            route="rag",
            reason="perfil estável do médico via conhecimento aprovado",
            filters=RagFilters(
                risk_max=intent.risk_level,
                source_types=["doctor_bio", "procedure_info"],
            ),
        )

    # Regra 8: dúvida factual simples (médico sem procedimento) → SQL local.
    # Convênio nunca chega aqui (Regra 4/4b garantem). Só medico_nome sem atendimento_nome.
    if intent.intent == IntentType.DUVIDA and contextual_entities.is_factual_only():
        return RouteDecision(route="sql", reason="fato estruturado — SQL local")

    # DEFAULT: clarify — nunca improvisamos em contexto clínico
    return RouteDecision(route="clarify", reason="fallback: intenção não mapeada")


def build_contextual_entities(
    current_turn_entities: EntitySet,
    session_entities: EntitySet | None,
    message: str = "",
) -> EntitySet:
    """
    current_turn_entities é a fonte primária.
    session_entities só entra quando a mensagem traz referência contextual
    explícita a médico/atendimento anterior.
    """
    if session_entities is None or not _has_doctor_context_reference(message):
        return current_turn_entities

    contextual = current_turn_entities.model_copy(deep=True)
    for field in (
        "medico_nome",
        "atendimento_nome",
        "convenio",
        "convenio_canonico",
    ):
        if getattr(contextual, field) is None:
            session_value = getattr(session_entities, field)
            if session_value is not None:
                setattr(contextual, field, session_value)
    return contextual


def _build_contextual_entities(
    current_turn_entities: EntitySet,
    session_entities: EntitySet | None,
    message: str = "",
) -> EntitySet:
    """
    Alias legado para manter compatibilidade interna enquanto o helper é usado
    pelo conversation.py e pelos testes novos.
    """
    return build_contextual_entities(
        current_turn_entities=current_turn_entities,
        session_entities=session_entities,
        message=message,
    )


def _has_doctor_context_reference(message: str = "") -> bool:
    normalized = (message or "").lower()
    return any(re.search(pattern, normalized) for pattern in _DOCTOR_CONTEXT_PATTERNS)
