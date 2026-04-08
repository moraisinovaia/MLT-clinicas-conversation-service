from __future__ import annotations
from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field


class IntentType(str, Enum):
    # Transacionais — alteram estado
    AGENDAR              = "agendar"
    REMARCAR             = "remarcar"
    CANCELAR             = "cancelar"
    CONFIRMAR            = "confirmar"
    FILA                 = "fila"
    RESPOSTA_FILA        = "resposta_fila"
    TRANSBORDO           = "transbordo"

    # Informacionais clínicos — rota rag
    DUVIDA_PREPARO          = "duvida_preparo"
    DUVIDA_ORIENTACAO       = "duvida_orientacao"
    DUVIDA_POS_PROCEDIMENTO = "duvida_pos_procedimento"

    # Informacional geral — rota sql | hybrid | clarify
    DUVIDA               = "duvida"

    # Conversacionais — rota direct
    SOCIAL               = "social"
    SAUDACAO             = "saudacao"
    AGRADECIMENTO        = "agradecimento"
    DESPEDIDA            = "despedida"
    FORA_ESCOPO          = "fora_escopo"

    # Crítico — direct imediato
    EMERGENCIA           = "emergencia"


class EntitySet(BaseModel):
    # Dados do atendimento (extraídos pelo LLM)
    medico_nome:      str | None = None
    atendimento_nome: str | None = None
    data_preferida:   str | None = None   # YYYY-MM-DD
    periodo:          Literal["manha", "tarde"] | None = None
    convenio:         str | None = None   # bruto, como veio do paciente
    convenio_canonico: str | None = None  # preenchido por alias_lookup, nunca pelo LLM
    agendamento_id:   str | None = None
    resposta_fila:    Literal["SIM", "NAO"] | None = None

    # Dados do paciente (extraídos da conversa ou de /patient-search)
    paciente_nome:    str | None = None
    paciente_celular: str | None = None
    data_nascimento:  str | None = None   # YYYY-MM-DD

    # --- Helpers usados pelo policy engine (sem I/O) ---

    def has_schedule_context(self) -> bool:
        """True se a dúvida é sobre disponibilidade/agenda → rota workflow."""
        if not self.atendimento_nome:
            return False
        schedule_keywords = {"agenda", "disponibilidade", "horario", "vaga", "data"}
        return any(kw in self.atendimento_nome.lower() for kw in schedule_keywords)

    def is_factual_only(self) -> bool:
        """True se a dúvida é puramente factual (médico ou convênio) sem procedimento.
        → rota sql; dado estruturado no banco, sem necessidade de RAG."""
        has_factual     = bool(self.medico_nome or self.convenio)
        has_explanatory = bool(self.atendimento_nome)
        return has_factual and not has_explanatory


class ParsedIntent(BaseModel):
    intent:              IntentType
    confidence:          float = Field(ge=0.0, le=1.0)
    entities:            EntitySet
    risk_level:          Literal["low", "medium", "high"]
    needs_clarification: bool
    # Mensagem pronta para WhatsApp (sem markdown), preenchida pelo executor
    mensagem_usuario:    str = ""


class ParseError(Exception):
    """LLM retornou JSON inválido ou esquema incorreto."""
    pass
