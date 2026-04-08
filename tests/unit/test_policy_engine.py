import pytest
from app.core.policy_engine import decide_route
from app.models.intent import IntentType, ParsedIntent, EntitySet
from app.models.state import ConversationState as S


def make_intent(
    intent=IntentType.DUVIDA,
    confidence=0.95,
    needs_clarification=False,
    risk_level="low",
    **entity_kwargs,
) -> ParsedIntent:
    return ParsedIntent(
        intent=intent,
        confidence=confidence,
        entities=EntitySet(**entity_kwargs),
        risk_level=risk_level,
        needs_clarification=needs_clarification,
    )


# ── Regra 0: clarificação ────────────────────────────────────────────────────

def test_needs_clarification_returns_clarify():
    p = make_intent(needs_clarification=True)
    assert decide_route(p, S.TRIAGEM).route == "clarify"

def test_low_confidence_returns_clarify():
    p = make_intent(confidence=0.69)
    assert decide_route(p, S.TRIAGEM).route == "clarify"

def test_exactly_070_is_not_clarify():
    p = make_intent(confidence=0.70, intent=IntentType.SOCIAL)
    assert decide_route(p, S.TRIAGEM).route == "direct"


# ── Regra 1: transacionais → workflow ────────────────────────────────────────

@pytest.mark.parametrize("intent", [
    IntentType.AGENDAR, IntentType.REMARCAR, IntentType.CANCELAR,
    IntentType.CONFIRMAR, IntentType.FILA, IntentType.RESPOSTA_FILA,
    IntentType.TRANSBORDO,
])
def test_transactional_intents_route_workflow(intent):
    p = make_intent(intent=intent)
    assert decide_route(p, S.TRIAGEM).route == "workflow"


# ── Regra 2: social → direct ─────────────────────────────────────────────────

@pytest.mark.parametrize("intent", [
    IntentType.SOCIAL, IntentType.SAUDACAO, IntentType.AGRADECIMENTO,
    IntentType.DESPEDIDA, IntentType.FORA_ESCOPO,
])
def test_social_intents_route_direct(intent):
    p = make_intent(intent=intent)
    assert decide_route(p, S.TRIAGEM).route == "direct"


# ── Regra 3: emergência → direct ─────────────────────────────────────────────

def test_emergency_routes_direct():
    p = make_intent(intent=IntentType.EMERGENCIA, risk_level="high")
    assert decide_route(p, S.TRIAGEM).route == "direct"


# ── Regra 5: dúvida factual → sql ────────────────────────────────────────────

def test_duvida_with_medico_no_procedure_routes_sql():
    p = make_intent(medico_nome="Dr. Marcelo")
    assert decide_route(p, S.TRIAGEM).route == "sql"

def test_duvida_with_convenio_no_procedure_routes_sql():
    p = make_intent(convenio="Unimed")
    assert decide_route(p, S.TRIAGEM).route == "sql"

def test_duvida_with_medico_and_procedure_routes_hybrid():
    """Médico + procedimento → hybrid (não sql)."""
    p = make_intent(medico_nome="Dr. Marcelo", atendimento_nome="MAPA 24h")
    assert decide_route(p, S.TRIAGEM).route == "hybrid"


# ── Regra 6: explicativos clínicos → rag ─────────────────────────────────────

@pytest.mark.parametrize("intent,expected_sources", [
    (IntentType.DUVIDA_PREPARO,          ["exam_prep", "medication_guide"]),
    (IntentType.DUVIDA_ORIENTACAO,       ["policy", "procedure_info"]),
    (IntentType.DUVIDA_POS_PROCEDIMENTO, ["post_procedure", "medication_guide"]),
])
def test_explanatory_intents_route_rag(intent, expected_sources):
    p = make_intent(intent=intent, risk_level="high")
    d = decide_route(p, S.TRIAGEM)
    assert d.route == "rag"
    assert d.filters is not None
    assert d.filters.source_types == expected_sources
    assert d.filters.risk_max == "high"


# ── Regra 7: dúvida com procedimento → hybrid ────────────────────────────────

def test_duvida_with_procedure_routes_hybrid():
    p = make_intent(atendimento_nome="Colonoscopia")
    assert decide_route(p, S.TRIAGEM).route == "hybrid"


# ── DEFAULT: clarify ──────────────────────────────────────────────────────────

def test_unmapped_intent_returns_clarify():
    """Qualquer combinação não mapeada cai em clarify — nunca rag."""
    p = make_intent(intent=IntentType.DUVIDA)  # sem entidades
    result = decide_route(p, S.TRIAGEM)
    assert result.route == "clarify"
    assert result.route != "rag"
