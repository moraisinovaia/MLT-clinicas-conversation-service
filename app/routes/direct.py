"""
Rota direct — social, saudação, emergência, fora_escopo.
Sem busca. Resposta imediata.
"""
from __future__ import annotations
from app.models.intent import IntentType, ParsedIntent
from app.models.conversation import OutboundMessage

EMERGENCY_RESPONSE = (
    "🚨 *Atenção:* Para emergências médicas, ligue imediatamente para o *SAMU 192* "
    "ou vá ao pronto-socorro mais próximo.\n\n"
    "Nossa equipe não consegue oferecer atendimento de urgência pelo WhatsApp."
)

OUT_OF_SCOPE_RESPONSE = (
    "Posso ajudar com agendamentos, informações sobre exames e preparo, "
    "convênios aceitos e dúvidas sobre a clínica. "
    "Para outras solicitações, entre em contato pelo telefone da recepção."
)


_SAUDACAO_RESPONSES = [
    "Olá! Tudo bem? Estou aqui para ajudar com agendamentos, informações sobre exames ou qualquer dúvida sobre a clínica.",
    "Oi! Seja bem-vindo. Como posso te ajudar hoje?",
    "Olá! Como posso ajudar?",
]

_AGRADECIMENTO_RESPONSES = [
    "Fico feliz em ter ajudado! Se precisar de mais alguma coisa, é só falar.",
    "De nada! Qualquer dúvida, estou por aqui.",
    "Disponha! Se precisar de algo mais, pode perguntar.",
]

_DESPEDIDA_RESPONSES = [
    "Até logo! Se precisar de algo, estamos à disposição.",
    "Tchau! Cuide-se bem.",
    "Até mais! Qualquer dúvida é só chamar.",
]


def build_direct_response(intent: ParsedIntent) -> list[OutboundMessage]:
    if intent.intent == IntentType.EMERGENCIA:
        return [OutboundMessage(text=EMERGENCY_RESPONSE, delay_ms=0)]

    if intent.intent == IntentType.FORA_ESCOPO:
        return [OutboundMessage(text=OUT_OF_SCOPE_RESPONSE)]

    if intent.intent in {IntentType.AGRADECIMENTO}:
        return [OutboundMessage(text=_AGRADECIMENTO_RESPONSES[0])]

    if intent.intent == IntentType.DESPEDIDA:
        return [OutboundMessage(text=_DESPEDIDA_RESPONSES[0])]

    # Saudação e social genérico
    return [OutboundMessage(text=_SAUDACAO_RESPONSES[0])]
