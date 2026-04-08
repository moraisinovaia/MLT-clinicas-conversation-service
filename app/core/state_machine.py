from app.models.state import ConversationState, VALID_TRANSITIONS
from app.models.intent import IntentType


class InvalidTransitionError(Exception):
    """Tentativa de transição de estado não permitida por VALID_TRANSITIONS."""
    pass


# Intents que PODEM alterar o estado da conversa.
# Qualquer intent fora desta lista responde e PERMANECE no estado atual.
TRANSACTIONAL_INTENTS = {
    IntentType.AGENDAR,
    IntentType.REMARCAR,
    IntentType.CANCELAR,
    IntentType.CONFIRMAR,
    IntentType.FILA,
    IntentType.RESPOSTA_FILA,
    IntentType.TRANSBORDO,
}


def validate_transition(
    current: ConversationState,
    target: ConversationState,
) -> ConversationState:
    """
    Valida e retorna o próximo estado.
    Lança InvalidTransitionError se a transição não for permitida.
    """
    allowed = VALID_TRANSITIONS.get(current, [])
    if target not in allowed:
        raise InvalidTransitionError(
            f"Transição inválida: {current.value} → {target.value}. "
            f"Permitidas: {[s.value for s in allowed]}"
        )
    return target


def resolve_next_state(
    current: ConversationState,
    intent:  IntentType,
    proposed: ConversationState | None = None,
) -> ConversationState:
    """
    Determina o próximo estado dado intent e estado atual.

    Regra de ouro: intents informacionais (duvida, social, emergencia, etc.)
    respondem e PERMANECEM no estado atual — nunca resetam o fluxo.

    Se proposed for None ou intent não for transacional, devolve current.
    Se proposed não for uma transição válida, devolve current com log de aviso
    (não lança exceção — segurança em produção).
    """
    if intent not in TRANSACTIONAL_INTENTS:
        return current  # mantém estado — regra de ouro

    if proposed is None:
        return current

    try:
        return validate_transition(current, proposed)
    except InvalidTransitionError:
        # Em produção: logar o aviso mas não travar o paciente
        return current
