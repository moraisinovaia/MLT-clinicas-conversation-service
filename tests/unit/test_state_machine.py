import pytest
from app.core.state_machine import (
    validate_transition, resolve_next_state,
    InvalidTransitionError, TRANSACTIONAL_INTENTS,
)
from app.models.state import ConversationState as S
from app.models.intent import IntentType


# ── validate_transition ──────────────────────────────────────────────────────

def test_valid_novo_to_triagem():
    assert validate_transition(S.NOVO, S.TRIAGEM) == S.TRIAGEM

def test_valid_triagem_to_coletando():
    assert validate_transition(S.TRIAGEM, S.COLETANDO_DADOS) == S.COLETANDO_DADOS

def test_valid_executando_slot_taken():
    """SLOT_TAKEN retorna ao coletando_dados."""
    assert validate_transition(S.EXECUTANDO, S.COLETANDO_DADOS) == S.COLETANDO_DADOS

def test_invalid_novo_to_executando():
    with pytest.raises(InvalidTransitionError):
        validate_transition(S.NOVO, S.EXECUTANDO)

def test_invalid_concluido_to_coletando():
    with pytest.raises(InvalidTransitionError):
        validate_transition(S.CONCLUIDO, S.COLETANDO_DADOS)

def test_invalid_transbordo_to_executando():
    with pytest.raises(InvalidTransitionError):
        validate_transition(S.TRANSBORDO, S.EXECUTANDO)


# ── resolve_next_state — regra de ouro ──────────────────────────────────────

@pytest.mark.parametrize("intent", [
    IntentType.DUVIDA,
    IntentType.DUVIDA_PREPARO,
    IntentType.DUVIDA_ORIENTACAO,
    IntentType.DUVIDA_POS_PROCEDIMENTO,
    IntentType.SOCIAL,
    IntentType.SAUDACAO,
    IntentType.AGRADECIMENTO,
    IntentType.DESPEDIDA,
    IntentType.EMERGENCIA,
    IntentType.FORA_ESCOPO,
])
def test_informational_intent_preserves_state(intent):
    """
    Critério 1.3: intent informacional em QUALQUER estado
    não altera o estado atual — bug original corrigido.
    """
    for state in [S.TRIAGEM, S.COLETANDO_DADOS, S.CONFIRMANDO, S.AGUARDANDO_FILA]:
        result = resolve_next_state(state, intent, proposed=S.TRIAGEM)
        assert result == state, (
            f"Intent {intent.value} em {state.value} não deveria mudar estado. "
            f"Obtido: {result.value}"
        )

def test_duvida_in_coletando_stays_coletando():
    """Teste específico do bug reportado: duvida durante coletando_dados."""
    result = resolve_next_state(S.COLETANDO_DADOS, IntentType.DUVIDA, proposed=S.TRIAGEM)
    assert result == S.COLETANDO_DADOS

def test_agendar_can_transition():
    result = resolve_next_state(S.TRIAGEM, IntentType.AGENDAR, proposed=S.COLETANDO_DADOS)
    assert result == S.COLETANDO_DADOS

def test_invalid_transactional_transition_returns_current():
    """Transição inválida não lança exceção em produção — retorna estado atual."""
    result = resolve_next_state(S.NOVO, IntentType.AGENDAR, proposed=S.EXECUTANDO)
    assert result == S.NOVO  # NOVO→EXECUTANDO é inválido

def test_none_proposed_returns_current():
    result = resolve_next_state(S.CONFIRMANDO, IntentType.CONFIRMAR, proposed=None)
    assert result == S.CONFIRMANDO
