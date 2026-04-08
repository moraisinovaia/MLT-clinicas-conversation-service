"""
Camada determinística pré-LLM.

Roda ANTES do LLM. Cobre casos óbvios com custo zero e latência mínima.
Retorna None se nenhuma regra bater → prosseguir para LLM.
"""
from __future__ import annotations
import unicodedata
from app.models.intent import IntentType, ParsedIntent, EntitySet
from app.models.state import ConversationState


# ── Vocabulários determinísticos ─────────────────────────────────────────────

EMERGENCY_KEYWORDS = {
    "socorro", "emergencia", "urgencia", "infarto", "avc", "derrame",
    "desmaiei", "desmaio", "nao consigo respirar", "muita dor",
    "chame ambulancia", "chama ambulancia", "estou passando mal",
    "nao to bem", "nao estou bem",
}

HUMAN_KEYWORDS = {
    "falar com atendente", "falar com pessoa", "falar com humano",
    "quero atendente", "me transfere", "atendimento humano",
    "fala com alguem", "preciso de ajuda humana", "quero falar com alguem",
    "me passa pra atendente", "quero falar com atendimento",
}

SIM_TOKENS = {
    "sim", "s", "yes", "quero", "pode", "ok", "tudo bem", "ta bom",
    "claro", "com certeza", "aceito", "confirmo", "pode ser",
    "quero sim", "pode sim", "ta otimo", "otimo", "ótimo",
}

NAO_TOKENS = {
    "nao", "n", "no", "nao quero", "nao preciso", "cancela",
    "desisto", "deixa", "pode nao", "nao obrigada", "nao obrigado",
    "nao quero mais", "recuso", "dispensado", "nao precisa",
}


# ── Normalização ─────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Remove acentos, lowercase, colapsa espaços."""
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_str = nfkd.encode("ASCII", "ignore").decode("ASCII")
    return " ".join(ascii_str.lower().split())


# ── Pre-parser ───────────────────────────────────────────────────────────────

def pre_parse(
    message: str,
    state:   ConversationState,
) -> ParsedIntent | None:
    """
    Fast path para casos determinísticos.
    Retorna None se nenhuma regra bater → chamar LLM.
    """
    normalized = normalize_text(message)

    # Emergência: prioridade máxima, qualquer estado
    if any(kw in normalized for kw in EMERGENCY_KEYWORDS):
        return ParsedIntent(
            intent=IntentType.EMERGENCIA,
            confidence=1.0,
            entities=EntitySet(),
            risk_level="high",
            needs_clarification=False,
        )

    # Pedido de humano: qualquer estado
    if any(kw in normalized for kw in HUMAN_KEYWORDS):
        return ParsedIntent(
            intent=IntentType.TRANSBORDO,
            confidence=1.0,
            entities=EntitySet(),
            risk_level="low",
            needs_clarification=False,
        )

    # Resposta de fila: APENAS quando aguardando resposta explícita
    # Tokens exatos → confiança 1.0 (sem LLM)
    # Ambíguo ("pode ser que sim dependendo do horário") → None → LLM classifica
    if state == ConversationState.AGUARDANDO_FILA:
        tokens = set(normalized.split())
        if tokens & SIM_TOKENS:
            return ParsedIntent(
                intent=IntentType.RESPOSTA_FILA,
                confidence=1.0,
                entities=EntitySet(resposta_fila="SIM"),
                risk_level="low",
                needs_clarification=False,
            )
        if tokens & NAO_TOKENS:
            return ParsedIntent(
                intent=IntentType.RESPOSTA_FILA,
                confidence=1.0,
                entities=EntitySet(resposta_fila="NAO"),
                risk_level="low",
                needs_clarification=False,
            )

    return None  # → sem match → chamar LLM
