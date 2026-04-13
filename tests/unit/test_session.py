"""Testa os helpers de sessão sem banco."""
import pytest
from app.core.session import safe_json_load, merge_entities, compute_missing_fields
from app.models.intent import EntitySet, IntentType, ParsedIntent


# ── safe_json_load (Fix 2) ───────────────────────────────────────────────────

def test_safe_json_load_str():
    assert safe_json_load('["a", "b"]', []) == ["a", "b"]

def test_safe_json_load_already_list():
    """asyncpg pode entregar já deserializado."""
    assert safe_json_load(["a", "b"], []) == ["a", "b"]

def test_safe_json_load_already_dict():
    assert safe_json_load({"k": "v"}, {}) == {"k": "v"}

def test_safe_json_load_none():
    assert safe_json_load(None, []) == []

def test_safe_json_load_invalid_str():
    assert safe_json_load("nao-e-json", {}) == {}

def test_safe_json_load_empty_str():
    assert safe_json_load("", []) == []


# ── merge_entities (Fix 1 + 5) ───────────────────────────────────────────────

def test_merge_fills_empty_fields():
    existing = EntitySet(medico_nome="Dr. Marcelo")
    new      = EntitySet(data_preferida="2026-04-15", periodo="manha")
    merged   = merge_entities(existing, new)
    assert merged.medico_nome    == "Dr. Marcelo"
    assert merged.data_preferida == "2026-04-15"
    assert merged.periodo        == "manha"

def test_merge_does_not_overwrite_with_none():
    """Novo parse retorna None num campo já coletado — deve preservar."""
    existing = EntitySet(medico_nome="Dr. Marcelo", convenio="Unimed")
    new      = EntitySet(medico_nome=None, data_preferida="2026-04-15")
    merged   = merge_entities(existing, new)
    assert merged.medico_nome    == "Dr. Marcelo"   # preservado
    assert merged.convenio       == "Unimed"         # preservado
    assert merged.data_preferida == "2026-04-15"    # novo campo

def test_merge_overwrites_with_non_none():
    """Se novo parse traz valor diferente (correção pelo paciente), atualiza."""
    existing = EntitySet(medico_nome="Dr. Marcelo")
    new      = EntitySet(medico_nome="Dr. João")
    merged   = merge_entities(existing, new)
    assert merged.medico_nome == "Dr. João"

def test_merge_both_empty_returns_empty():
    merged = merge_entities(EntitySet(), EntitySet())
    assert merged.medico_nome is None


def test_merge_preserves_transacional_context_for_partial_followup():
    existing = EntitySet(medico_nome="Dr. Guilherme")
    new = EntitySet(data_preferida="amanhã", periodo="tarde")
    merged = merge_entities(existing, new)
    assert merged.medico_nome == "Dr. Guilherme"
    assert merged.data_preferida == "amanhã"
    assert merged.periodo == "tarde"


def test_merge_preserves_workflow_context_even_after_informational_turn():
    start = EntitySet(medico_nome="Dr. Guilherme")
    after_info_turn = merge_entities(start, EntitySet())
    after_followup = merge_entities(after_info_turn, EntitySet(data_preferida="amanhã"))
    assert after_followup.medico_nome == "Dr. Guilherme"
    assert after_followup.data_preferida == "amanhã"


# ── compute_missing_fields (Fix 4) ───────────────────────────────────────────

def make_intent(intent_type, **entities):
    return ParsedIntent(
        intent=intent_type,
        confidence=0.9,
        entities=EntitySet(**entities),
        risk_level="low",
        needs_clarification=False,
    )

def test_agendar_all_missing():
    p = make_intent(IntentType.AGENDAR)
    missing = compute_missing_fields(p, p.entities)
    assert set(missing) == {"medico_nome", "atendimento_nome", "data_preferida", "convenio"}

def test_agendar_partial():
    p = make_intent(IntentType.AGENDAR, medico_nome="Dr. Marcelo", convenio="Unimed")
    missing = compute_missing_fields(p, p.entities)
    assert set(missing) == {"atendimento_nome", "data_preferida"}

def test_agendar_complete():
    p = make_intent(
        IntentType.AGENDAR,
        medico_nome="Dr. Marcelo",
        atendimento_nome="Consulta",
        data_preferida="2026-04-15",
        convenio="Unimed",
    )
    assert compute_missing_fields(p, p.entities) == []

def test_cancelar_needs_agendamento_id():
    p = make_intent(IntentType.CANCELAR)
    assert compute_missing_fields(p, p.entities) == ["agendamento_id"]

def test_cancelar_complete():
    p = make_intent(IntentType.CANCELAR, agendamento_id="abc-123")
    assert compute_missing_fields(p, p.entities) == []

def test_duvida_has_no_required_fields():
    """Dúvidas não têm campos obrigatórios — clarify por ambiguidade, não por campos."""
    p = make_intent(IntentType.DUVIDA)
    assert compute_missing_fields(p, p.entities) == []

def test_social_has_no_required_fields():
    p = make_intent(IntentType.SOCIAL)
    assert compute_missing_fields(p, p.entities) == []
