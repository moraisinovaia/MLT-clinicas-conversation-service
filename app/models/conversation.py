from __future__ import annotations
from typing import Literal
from pydantic import BaseModel
from .state import ConversationState


class ConversationRequest(BaseModel):
    session_id:  str
    cliente_id:  str
    message:     str
    media_type:  Literal["text", "audio", "image", "document"] = "text"


class OutboundMessage(BaseModel):
    text:     str
    delay_ms: int = 800   # pausa antes de enviar (simula digitação)


class HandoffData(BaseModel):
    chatwoot_inbox_id: str
    nota_privada:      str
    labels:            list[str] = []


class ConversationResponse(BaseModel):
    messages:     list[OutboundMessage]
    action:       Literal["send", "handoff", "none"]
    handoff_data: HandoffData | None = None
    new_state:    ConversationState
    session_id:   str
    cliente_id:   str
    trace_id:     str
