"""
Rota clarify — gera pergunta para 1 entidade faltante por vez.

Nunca pede 2 informações na mesma mensagem.
Ordem de prioridade: medico_nome → atendimento_nome → data_preferida
                     → periodo → convenio.
"""
from __future__ import annotations
from app.models.intent import ParsedIntent
from app.models.conversation import OutboundMessage

# Ordem de coleta: quem está primeiro é perguntado primeiro
COLLECTION_ORDER = [
    "medico_nome",
    "atendimento_nome",
    "data_preferida",
    "periodo",
    "convenio",
]

QUESTIONS: dict[str, str] = {
    "medico_nome":      "Com qual médico você gostaria de agendar?",
    "atendimento_nome": "Qual tipo de atendimento ou exame você precisa?",
    "data_preferida":   "Qual data você prefere? (ex: segunda-feira, 15/05...)",
    "periodo":          "Prefere horário pela manhã ou à tarde?",
    "convenio":         "Qual é o seu convênio? (ou particular?)",
}

AMBIGUOUS_RESPONSE = (
    "Não entendi muito bem. Pode me explicar como posso ajudar? "
    "Estou aqui para agendamentos, exames e informações sobre a clínica. 😊"
)


def build_clarify_response(
    intent:          ParsedIntent,
    dados_faltantes: list[str],
) -> list[OutboundMessage]:
    """
    Se há entidades faltantes específicas → pergunta a próxima na ordem.
    Se é ambiguidade geral (needs_clarification / baixa confiança) → resposta genérica.
    """
    # Baixa confiança ou ambiguidade geral
    if intent.needs_clarification and not dados_faltantes:
        return [OutboundMessage(text=AMBIGUOUS_RESPONSE)]

    # Pergunta a próxima entidade faltante (só 1 por vez)
    for field in COLLECTION_ORDER:
        if field in dados_faltantes:
            return [OutboundMessage(text=QUESTIONS[field])]

    # Fallback: não sabe o que perguntar → pede reformulação
    return [OutboundMessage(text=AMBIGUOUS_RESPONSE)]
