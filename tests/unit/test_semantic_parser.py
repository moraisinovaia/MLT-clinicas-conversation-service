"""Testa o parser sem chamar o LLM — valida extração e tratamento de erros."""
import pytest
from app.core.semantic_parser import _parse_raw, _extract_json
from app.models.intent import IntentType, ParseError


VALID_JSON = """{
  "intent": "agendar",
  "confidence": 0.92,
  "entities": {
    "medico_nome": "Dr. Marcelo",
    "atendimento_nome": null,
    "data_preferida": "2026-04-15",
    "periodo": "manha",
    "convenio": "unimed",
    "agendamento_id": null,
    "resposta_fila": null,
    "paciente_nome": null,
    "paciente_celular": null,
    "data_nascimento": null
  },
  "risk_level": "low",
  "needs_clarification": false
}"""

VALID_JSON_WITH_FENCES = f"```json\n{VALID_JSON}\n```"


def test_parse_valid_json():
    result = _parse_raw(VALID_JSON)
    assert result.intent == IntentType.AGENDAR
    assert result.confidence == 0.92
    assert result.entities.medico_nome == "Dr. Marcelo"
    assert result.entities.periodo == "manha"
    assert result.entities.convenio == "unimed"
    assert result.needs_clarification is False

def test_parse_json_with_fences():
    """LLM frequentemente envolve o JSON em cercas de código."""
    result = _parse_raw(VALID_JSON_WITH_FENCES)
    assert result.intent == IntentType.AGENDAR

def test_unknown_intent_falls_back_to_duvida():
    """Intent desconhecido → duvida (graceful fallback)."""
    raw = VALID_JSON.replace('"intent": "agendar"', '"intent": "intencao_inventada"')
    result = _parse_raw(raw)
    assert result.intent == IntentType.DUVIDA

def test_invalid_json_raises_parse_error():
    with pytest.raises(ParseError):
        _parse_raw("isso não é json")

def test_empty_response_raises_parse_error():
    with pytest.raises(ParseError):
        _parse_raw("")

def test_json_missing_intent_field():
    """Sem campo intent → duvida como fallback."""
    raw = '{"confidence": 0.8, "entities": {}, "risk_level": "low", "needs_clarification": false}'
    result = _parse_raw(raw)
    assert result.intent == IntentType.DUVIDA

def test_confidence_below_threshold_is_preserved():
    """O parser não altera confidence — quem aplica o threshold é o policy engine."""
    raw = VALID_JSON.replace('"confidence": 0.92', '"confidence": 0.5')
    result = _parse_raw(raw)
    assert result.confidence == 0.5

def test_extract_json_from_text_with_preamble():
    text = 'Aqui está o JSON solicitado: {"intent": "duvida", "confidence": 0.9, "entities": {}, "risk_level": "low", "needs_clarification": false}'
    data = _extract_json(text)
    assert data["intent"] == "duvida"
