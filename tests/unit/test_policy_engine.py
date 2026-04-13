import pytest
from app.core.policy_engine import decide_route
from app.models.intent import IntentType, ParsedIntent, EntitySet
from app.models.state import ConversationState as S


def make_intent(
    intent=IntentType.DUVIDA,
    confidence=0.95,
    needs_clarification=False,
    risk_level="low",
    is_operational_query=False,
    **entity_kwargs,
) -> ParsedIntent:
    return ParsedIntent(
        intent=intent,
        confidence=confidence,
        entities=EntitySet(**entity_kwargs),
        risk_level=risk_level,
        needs_clarification=needs_clarification,
        is_operational_query=is_operational_query,
    )


# ── Regra 0: transacionais → workflow (antes de clarify) ────────────────────

@pytest.mark.parametrize("intent", [
    IntentType.AGENDAR, IntentType.REMARCAR, IntentType.CANCELAR,
    IntentType.CONFIRMAR, IntentType.FILA, IntentType.RESPOSTA_FILA,
    IntentType.TRANSBORDO,
])
def test_transactional_intents_route_workflow(intent):
    p = make_intent(intent=intent)
    assert decide_route(p, S.TRIAGEM).route == "workflow"


def test_transactional_with_needs_clarification_still_routes_workflow():
    """needs_clarification=True não bloqueia transacionais — workflow coleta o que falta."""
    p = make_intent(intent=IntentType.AGENDAR, needs_clarification=True)
    assert decide_route(p, S.TRIAGEM).route == "workflow"


def test_transactional_with_low_confidence_still_routes_workflow():
    """Confiança baixa não bloqueia transacionais."""
    p = make_intent(intent=IntentType.AGENDAR, confidence=0.50)
    assert decide_route(p, S.TRIAGEM).route == "workflow"


# ── Regra 1: clarificação ────────────────────────────────────────────────────

def test_needs_clarification_returns_clarify():
    p = make_intent(needs_clarification=True)
    assert decide_route(p, S.TRIAGEM).route == "clarify"

def test_low_confidence_returns_clarify():
    p = make_intent(confidence=0.69)
    assert decide_route(p, S.TRIAGEM).route == "clarify"

def test_exactly_070_is_not_clarify():
    p = make_intent(confidence=0.70, intent=IntentType.SOCIAL)
    assert decide_route(p, S.TRIAGEM).route == "direct"


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


# ── Regra 4c: localização (defensivo para EXPLANATORY_INTENTS misclassificados) ──

def test_duvida_orientacao_with_location_keyword_routes_sql():
    """LLM pode classificar 'horário de funcionamento' como duvida_orientacao.
    Regra 4c deve capturar isso e redirecionar para sql."""
    p = make_intent(intent=IntentType.DUVIDA_ORIENTACAO)
    p.mensagem_usuario = "qual o horário de funcionamento da clínica?"
    assert decide_route(p, S.TRIAGEM).route == "sql"


def test_duvida_orientacao_with_location_keyword_and_medico_does_not_route_sql():
    """Com médico mencionado, não aplica localização — é outra dúvida."""
    p = make_intent(intent=IntentType.DUVIDA_ORIENTACAO, medico_nome="Dr. Marcelo")
    p.mensagem_usuario = "qual o horário do Dr. Marcelo?"
    # não deve ir para sql, cai para rag (Regra 5)
    assert decide_route(p, S.TRIAGEM).route == "rag"


# ── Regra 5: dúvida factual → sql ────────────────────────────────────────────

def test_duvida_with_medico_no_procedure_routes_sql():
    p = make_intent(medico_nome="Dr. Marcelo")
    assert decide_route(p, S.TRIAGEM).route == "sql"

def test_duvida_with_convenio_no_procedure_routes_workflow():
    p = make_intent(convenio="Unimed")
    assert decide_route(p, S.TRIAGEM).route == "workflow"

def test_duvida_with_medico_and_procedure_routes_rag():
    """Médico + procedimento sem contexto operacional → rag (não hybrid, não sql)."""
    p = make_intent(medico_nome="Dr. Marcelo", atendimento_nome="MAPA 24h")
    d = decide_route(p, S.TRIAGEM)
    assert d.route == "rag"
    assert d.filters is not None
    assert "procedure_info" in d.filters.source_types


# ── Regra 6: explicativos clínicos → rag ─────────────────────────────────────

@pytest.mark.parametrize("intent,expected_sources", [
    (IntentType.DUVIDA_PREPARO,          ["exam_prep"]),
    (IntentType.DUVIDA_ORIENTACAO,       ["policy", "procedure_info", "operational_script"]),
    (IntentType.DUVIDA_POS_PROCEDIMENTO, ["exam_prep", "operational_script"]),
])
def test_explanatory_intents_route_rag(intent, expected_sources):
    p = make_intent(intent=intent, risk_level="high")
    d = decide_route(p, S.TRIAGEM)
    assert d.route == "rag"
    assert d.filters is not None
    assert d.filters.source_types == expected_sources
    assert d.filters.risk_max == "high"


# ── Regra 6: dúvida com procedimento (DUVIDA genérico) → rag ─────────────────

def test_duvida_with_procedure_routes_rag():
    """atendimento_nome sem contexto operacional → rag direto (não hybrid)."""
    p = make_intent(atendimento_nome="Colonoscopia")
    d = decide_route(p, S.TRIAGEM)
    assert d.route == "rag"
    assert d.filters is not None
    assert d.filters.source_types == ["exam_prep", "procedure_info"]


def test_duvida_operacional_convenio_routes_workflow():
    # Regra 4b: entities.convenio set → sempre workflow, independente de is_operational_query
    p = make_intent(medico_nome="Dr. Hermann", convenio="HGU")
    assert decide_route(p, S.TRIAGEM).route == "workflow"


def test_duvida_operacional_servico_ativo_routes_workflow():
    # Regra 4: is_operational_query=True (LLM detecta "faz X?" como operacional)
    p = make_intent(
        medico_nome="Dr. Hermann",
        atendimento_nome="Gonioscopia",
        is_operational_query=True,
    )
    assert decide_route(p, S.TRIAGEM).route == "workflow"


def test_duvida_operacional_lista_de_convenios_por_medico_routes_workflow():
    # Regra 4: is_operational_query=True (LLM detecta "quais convênios?" como operacional)
    p = make_intent(medico_nome="Dr. Hermann", is_operational_query=True)
    assert decide_route(p, S.TRIAGEM).route == "workflow"


def test_duvida_operacional_lista_de_servicos_por_medico_routes_workflow():
    # Regra 4: is_operational_query=True (LLM detecta "quais procedimentos?" como operacional)
    p = make_intent(medico_nome="Dr. Hermann", is_operational_query=True)
    assert decide_route(p, S.TRIAGEM).route == "workflow"


def test_duvida_perfil_medico_idade_nao_vai_para_workflow():
    # is_operational_query=False (perfil estável) → Regra 5 → rag
    p = make_intent(medico_nome="Dr. Hermann")
    p.mensagem_usuario = "Dr. Hermann atende criancas?"
    d = decide_route(p, S.TRIAGEM)
    assert d.route == "rag"
    assert d.filters is not None
    assert d.filters.source_types == ["doctor_bio", "procedure_info"]


def test_duvida_explicativa_sem_contexto_operacional_vai_para_rag():
    # is_operational_query=False → não dispara Regra 4 → Regra 6 → rag (não hybrid)
    p = make_intent(atendimento_nome="Gonioscopia")
    d = decide_route(p, S.TRIAGEM)
    assert d.route == "rag"
    assert d.filters.source_types == ["exam_prep", "procedure_info"]


# ── DEFAULT: clarify ──────────────────────────────────────────────────────────

def test_unmapped_intent_returns_clarify():
    """Qualquer combinação não mapeada cai em clarify — nunca rag."""
    p = make_intent(intent=IntentType.DUVIDA)  # sem entidades
    result = decide_route(p, S.TRIAGEM)
    assert result.route == "clarify"
    assert result.route != "rag"
