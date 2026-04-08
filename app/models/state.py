from enum import Enum


class ConversationState(str, Enum):
    NOVO            = "novo"
    TRIAGEM         = "triagem"
    COLETANDO_DADOS = "coletando_dados"
    CONFIRMANDO     = "confirmando"
    EXECUTANDO      = "executando"
    CONCLUIDO       = "concluido"
    AGUARDANDO_FILA = "aguardando_fila"
    TRANSBORDO      = "transbordo"


# Transições válidas — qualquer outra é rejeitada pela state machine.
# Regra de ouro: intents informacionais (duvida, social, emergencia)
# NÃO aparecem aqui — eles respondem sem mudar de estado.
VALID_TRANSITIONS: dict[ConversationState, list[ConversationState]] = {
    # NOVO é apenas estado inicial de sessão — nunca deve ser destino
    ConversationState.NOVO: [
        ConversationState.TRIAGEM,
        ConversationState.COLETANDO_DADOS,
        ConversationState.CONFIRMANDO,
        ConversationState.TRANSBORDO,
        ConversationState.AGUARDANDO_FILA,
        ConversationState.CONCLUIDO,
    ],
    ConversationState.TRIAGEM: [
        ConversationState.COLETANDO_DADOS,
        ConversationState.CONFIRMANDO,      # dados completos na 1ª msg → pede confirmação
        ConversationState.EXECUTANDO,       # intent completo e já confirmado
        ConversationState.TRANSBORDO,
        ConversationState.AGUARDANDO_FILA,
        ConversationState.CONCLUIDO,
    ],
    ConversationState.COLETANDO_DADOS: [
        ConversationState.CONFIRMANDO,
        ConversationState.TRIAGEM,          # paciente mudou de assunto
        ConversationState.TRANSBORDO,
    ],
    ConversationState.CONFIRMANDO: [
        ConversationState.EXECUTANDO,
        ConversationState.CONCLUIDO,        # API executou direto após confirmação
        ConversationState.COLETANDO_DADOS,  # paciente corrigiu dado
        ConversationState.TRIAGEM,
    ],
    ConversationState.EXECUTANDO: [
        ConversationState.CONCLUIDO,
        ConversationState.COLETANDO_DADOS,  # SLOT_TAKEN → pedir nova data
        ConversationState.TRIAGEM,
    ],
    ConversationState.CONCLUIDO: [
        ConversationState.TRIAGEM,
    ],
    ConversationState.AGUARDANDO_FILA: [
        ConversationState.CONCLUIDO,        # respondeu SIM → executa
        ConversationState.TRIAGEM,          # respondeu NÃO → volta
    ],
    ConversationState.TRANSBORDO: [
        ConversationState.TRIAGEM,          # Chatwoot resolved → volta para IA
    ],
}
