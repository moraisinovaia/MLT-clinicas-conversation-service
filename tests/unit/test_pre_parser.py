import pytest
from app.core.pre_parser import pre_parse, normalize_text
from app.models.intent import IntentType
from app.models.state import ConversationState as S


# ── normalize_text ───────────────────────────────────────────────────────────

def test_normalize_removes_accents():
    assert normalize_text("Ação") == "acao"

def test_normalize_lowercase():
    assert normalize_text("URGÊNCIA") == "urgencia"

def test_normalize_collapses_spaces():
    assert normalize_text("  olá   mundo  ") == "ola mundo"


# ── Emergência ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("msg", [
    "SOCORRO estou passando mal",
    "urgencia preciso de ajuda",
    "avc meu marido teve um avc",
    "estou com muita dor no peito",
])
def test_emergency_detected(msg):
    result = pre_parse(msg, S.TRIAGEM)
    assert result is not None
    assert result.intent == IntentType.EMERGENCIA
    assert result.confidence == 1.0

def test_emergency_detected_in_any_state():
    for state in S:
        result = pre_parse("socorro infarto", state)
        assert result is not None
        assert result.intent == IntentType.EMERGENCIA


# ── Humano ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("msg", [
    "quero falar com atendente",
    "me transfere para uma pessoa",
    "preciso de atendimento humano",
    "fala com alguem por favor",
])
def test_human_request_detected(msg):
    result = pre_parse(msg, S.TRIAGEM)
    assert result is not None
    assert result.intent == IntentType.TRANSBORDO
    assert result.confidence == 1.0


# ── SIM/NÃO na fila ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("msg", ["sim", "s", "quero", "pode", "claro", "confirmo"])
def test_sim_in_fila_state(msg):
    result = pre_parse(msg, S.AGUARDANDO_FILA)
    assert result is not None
    assert result.intent == IntentType.RESPOSTA_FILA
    assert result.entities.resposta_fila == "SIM"

@pytest.mark.parametrize("msg", ["nao", "n", "cancela", "desisto", "nao obrigada"])
def test_nao_in_fila_state(msg):
    result = pre_parse(msg, S.AGUARDANDO_FILA)
    assert result is not None
    assert result.entities.resposta_fila == "NAO"

def test_sim_outside_fila_returns_none():
    """SIM fora do estado AGUARDANDO_FILA → LLM classifica."""
    result = pre_parse("sim", S.TRIAGEM)
    assert result is None

def test_ambiguous_fila_returns_none():
    """Mensagem ambígua na fila → None → LLM."""
    result = pre_parse("talvez dependendo do horário", S.AGUARDANDO_FILA)
    assert result is None


# ── Casos que NÃO devem bater no pre-parser ──────────────────────────────────

@pytest.mark.parametrize("msg", [
    "quero marcar uma consulta",
    "qual o preparo do MAPA 24h?",
    "boa tarde",
    "o Dr. Marcelo atende Unimed?",
])
def test_normal_messages_return_none(msg):
    result = pre_parse(msg, S.TRIAGEM)
    assert result is None
