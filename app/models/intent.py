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

    # Informacional geral — rota sql | rag | workflow | clarify
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
    hora_consulta:    str | None = None   # "HH:MM" — preenchido após paciente escolher slot
    resposta_fila:    Literal["SIM", "NAO"] | None = None

    # Dados do paciente (extraídos da conversa ou de /patient-search)
    paciente_nome:    str | None = None
    paciente_celular: str | None = None
    data_nascimento:  str | None = None   # YYYY-MM-DD

    # --- Helpers usados pelo policy engine (sem I/O) ---

    def touches_live_operational_context(self, message: str = "") -> bool:
        """
        True quando a dúvida encosta em dado operacional vivo que deve vir da
        GT Inova: elegibilidade, agenda, limite, disponibilidade ou serviço ativo.

        Regra sênior: se a resposta for "sim/não/pode/tem/aceita", a fonte é a GT Inova.
        Se for "como funciona/o que levar/qual preparo", pode ir para RAG.
        """
        msg = (message or "").lower()

        schedule_keywords = {
            "agenda", "disponibilidade", "horario", "horário", "vaga", "vagas",
            "data", "dia", "dias", "turno", "manha", "manhã", "tarde",
            "limite", "quantidade", "quantos", "tem vaga", "tem horario",
        }

        # Verbos e frases de elegibilidade operacional
        eligibility_keywords = {
            "convenio", "convênio",
            "aceita", "aceito", "aceitar", "nao aceita", "não aceita",
            "atende pelo", "atende por", "trabalha com",
            "qual convenio", "quais convenios",
            "cobre", "coberto", "cobertura",
            "elegivel", "elegível", "elegibilidade",
            "plano", "planos",
        }

        # Nomes de convênios reais das clínicas (normalizados)
        # Qualquer menção = pergunta operacional → GT Inova decide
        convenio_names = {
            "hgu", "unimed", "cassi", "geap", "camed", "medclin", "medsaude",
            "cpp", "saude caixa", "saudecaixa", "mineracao caraiba", "mineracao",
            "caraiba", "medprev", "particular", "work", "dr visao", "dr. visao",
            "rosarinha", "dormentes", "sus", "sus novo afranio",
            "amil", "bradesco saude", "sulamerica", "sao francisco",
        }

        # Verbos de serviço ativo ligados a médico ou atendimento
        active_service_keywords = {
            "faz", "fazer", "realiza", "realizar", "oferece", "oferecer",
            "disponibiliza", "tem o servico", "tem servico",
            "servico", "serviço", "procedimento", "procedimentos",
            "ativo", "disponivel", "disponível",
        }

        has_schedule_question = any(kw in msg for kw in schedule_keywords)

        has_eligibility_question = (
            bool(self.convenio)
            or any(kw in msg for kw in eligibility_keywords)
            or any(name in msg for name in convenio_names)
        )

        has_active_service_question = (
            any(kw in msg for kw in active_service_keywords)
            and bool(self.medico_nome or self.atendimento_nome)
        )

        return has_schedule_question or has_eligibility_question or has_active_service_question

    def touches_doctor_profile_context(self, message: str = "") -> bool:
        """
        True quando a dúvida é sobre informação relativamente estável do médico,
        adequada para conhecimento aprovado e não para agenda viva.
        """
        msg = (message or "").lower()
        profile_keywords = {
            "idade", "idades", "crianca", "criança", "criancas", "crianças",
            "adulto", "adultos", "especialidade", "especialidades",
            "crm", "rqe", "perfil", "biografia",
        }
        return bool(self.medico_nome) and any(keyword in msg for keyword in profile_keywords)

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
    intent:               IntentType
    confidence:           float = Field(ge=0.0, le=1.0)
    entities:             EntitySet
    risk_level:           Literal["low", "medium", "high"]
    needs_clarification:  bool
    # True quando a pergunta exige dado em tempo real da API de agendamentos:
    # agenda, disponibilidade, elegibilidade de convênio, serviço ativo de médico.
    # Preenchido pelo semantic_parse via LLM — substitui keyword matching no router.
    is_operational_query: bool = False
    # Mensagem pronta para WhatsApp (sem markdown), preenchida pelo executor
    mensagem_usuario:     str = ""


class ParseError(Exception):
    """LLM retornou JSON inválido ou esquema incorreto."""
    pass
