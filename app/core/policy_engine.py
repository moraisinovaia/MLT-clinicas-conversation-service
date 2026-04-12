from __future__ import annotations
from dataclasses import dataclass, field
from app.models.intent import IntentType, ParsedIntent
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

    # Regra 0: clarificação tem prioridade absoluta
    if intent.needs_clarification or intent.confidence < 0.70:
        return RouteDecision(route="clarify", reason="baixa confiança ou ambiguidade")

    # Regra 1: intents transacionais → workflow (state machine + GT Inova API)
    if intent.intent in TRANSACTIONAL_INTENTS:
        return RouteDecision(route="workflow", reason="intent transacional")

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

    # Regra 4c: fallback de segurança operacional via keywords.
    # Garante que perguntas operacionais que o LLM não sinalizou como is_operational_query
    # (ex: "tem vaga?", "aceita Unimed?") ainda vão para o workflow em vez do RAG.
    # EntitySet.touches_live_operational_context() usa listas curadas de keywords.
    if (
        intent.intent == IntentType.DUVIDA
        and intent.entities.touches_live_operational_context(intent.mensagem_usuario)
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
    if intent.intent == IntentType.DUVIDA and intent.entities.atendimento_nome:
        return RouteDecision(
            route="rag",
            reason="dúvida explicativa sobre procedimento — RAG direto",
            filters=RagFilters(
                risk_max=intent.risk_level,
                source_types=["exam_prep", "procedure_info"],
            ),
        )

    # Regra 7: perfil estável do médico (CRM, especialidade, formação) → RAG
    if (
        intent.intent == IntentType.DUVIDA
        and intent.entities.touches_doctor_profile_context(intent.mensagem_usuario)
    ):
        return RouteDecision(
            route="rag",
            reason="perfil estável do médico via conhecimento aprovado",
            filters=RagFilters(
                risk_max=intent.risk_level,
                source_types=["doctor_bio", "procedure_info"],
            ),
        )

    # Regra 8: dúvida sobre localização, endereço, telefone, horário → SQL
    # configuracoes_clinica tem colunas endereco e horario_funcionamento.
    if (
        intent.intent == IntentType.DUVIDA
        and any(kw in intent.mensagem_usuario.lower() for kw in _LOCATION_KEYWORDS)
        and not intent.entities.medico_nome
    ):
        return RouteDecision(route="sql", reason="localização/contato da clínica — SQL direto")

    # Regra 9: dúvida factual simples (médico sem procedimento) → SQL local.
    # Convênio nunca chega aqui (Regra 4/4b garantem). Só medico_nome sem atendimento_nome.
    if intent.intent == IntentType.DUVIDA and intent.entities.is_factual_only():
        return RouteDecision(route="sql", reason="fato estruturado — SQL local")

    # DEFAULT: clarify — nunca improvisamos em contexto clínico
    return RouteDecision(route="clarify", reason="fallback: intenção não mapeada")
