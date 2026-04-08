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


# ── Modelo de retorno ────────────────────────────────────────────────────────

@dataclass
class RagFilters:
    risk_max:     str = "high"
    source_types: list[str] = field(default_factory=list)


@dataclass
class RouteDecision:
    route:   str                  # sql | rag | hybrid | workflow | clarify | direct
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

    DEFAULT = clarify. Nunca rag como fallback em contexto clínico.
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

    # Regra 4: dúvida sobre disponibilidade/agenda → workflow (/availability)
    # Nunca RAG para agenda — dados mudam em tempo real
    if intent.intent == IntentType.DUVIDA and intent.entities.has_schedule_context():
        return RouteDecision(route="workflow", reason="disponibilidade via GT Inova /availability")

    # Regra 5: dúvida factual simples (médico ou convênio, sem procedimento) → sql
    if intent.intent == IntentType.DUVIDA and intent.entities.is_factual_only():
        return RouteDecision(route="sql", reason="fato estruturado — SQL local")

    # Regra 6: dúvidas explicativas clínicas → rag com filtros de risco
    if intent.intent in EXPLANATORY_INTENTS:
        source_map = {
            IntentType.DUVIDA_PREPARO:          ["exam_prep", "medication_guide"],
            IntentType.DUVIDA_ORIENTACAO:        ["policy", "procedure_info"],
            IntentType.DUVIDA_POS_PROCEDIMENTO:  ["post_procedure", "medication_guide"],
        }
        return RouteDecision(
            route="rag",
            reason=f"dúvida clínica: {intent.intent.value}",
            filters=RagFilters(
                risk_max=intent.risk_level,
                source_types=source_map.get(intent.intent, []),
            ),
        )

    # Regra 7: dúvida com procedimento mencionado → hybrid (sql + rag)
    if intent.intent == IntentType.DUVIDA and intent.entities.atendimento_nome:
        return RouteDecision(
            route="hybrid",
            reason="procedimento mencionado: SQL + RAG combinados",
            filters=RagFilters(risk_max=intent.risk_level),
        )

    # DEFAULT: clarify — nunca improvisamos em contexto clínico
    return RouteDecision(route="clarify", reason="fallback: intenção não mapeada")
