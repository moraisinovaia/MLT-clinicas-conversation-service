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


def build_direct_response(intent: ParsedIntent) -> list[OutboundMessage]:
    if intent.intent == IntentType.EMERGENCIA:
        return [OutboundMessage(text=EMERGENCY_RESPONSE, delay_ms=0)]

    if intent.intent in {IntentType.FORA_ESCOPO}:
        return [OutboundMessage(text=OUT_OF_SCOPE_RESPONSE)]

    # Social/saudação/agradecimento/despedida:
    # mensagem_usuario é preenchida pelo LLM no semantic parse
    text = intent.mensagem_usuario or "Como posso ajudar?"
    return [OutboundMessage(text=text)]
