"""
Semantic Parse — único ponto de NLU do sistema.

LLM retorna JSON estruturado → Pydantic valida → ParsedIntent.
JSON inválido → ParseError (nunca trava a conversa).
"""
from __future__ import annotations
import json
import re
from app.models.intent import IntentType, ParsedIntent, EntitySet, ParseError
from app.integrations.openrouter import call_llm


SYSTEM_PROMPT = """\
Você é o motor de intenção de um sistema de atendimento para clínicas médicas.

Analise a mensagem do paciente e retorne APENAS um JSON válido, sem texto extra, \
sem cercas de código, sem explicações.

Schema obrigatório:
{
  "intent": "<um dos intents abaixo>",
  "confidence": <float 0.0-1.0>,
  "entities": {
    "medico_nome":       "<str|null>",
    "atendimento_nome":  "<str|null>",
    "data_preferida":    "<texto livre com data ou período preferido, ex: '2025-05-15', 'próxima segunda', 'semana que vem', 'qualquer dia da próxima semana' | null>",
    "periodo":           "<manha|tarde|null>",
    "convenio":          "<str bruto|null>",
    "agendamento_id":    "<uuid|null>",
    "resposta_fila":     "<SIM|NAO|null>",
    "hora_consulta":     "<HH:MM quando paciente escolhe horário específico, ex: '09:00', '14:30' | null>",
    "slot_id":           "<id do slot se mencionado explicitamente | null>",
    "paciente_nome":     "<str|null>",
    "paciente_celular":  "<str só dígitos|null>",
    "data_nascimento":   "<YYYY-MM-DD|null>"
  },
  "risk_level":          "<low|medium|high>",
  "needs_clarification": <true|false>
}

Intents válidos:
  agendar | remarcar | cancelar | confirmar | fila | resposta_fila | transbordo
  duvida_preparo | duvida_orientacao | duvida_pos_procedimento | duvida
  social | saudacao | agradecimento | despedida | fora_escopo | emergencia

Regras de risk_level:
  high   → preparo de exame, jejum, suspensão de medicação, pós-procedimento
  medium → convênios, preços, políticas, autorização
  low    → saudação, bio de médico, endereço, FAQ geral

needs_clarification = true quando a mensagem for ambígua e faltar informação
essencial para continuar (ex: "quero marcar" sem médico nem procedimento).
"""


_INTENT_VALUES = {i.value for i in IntentType}


def _extract_json(raw: str) -> dict:
    """Remove cercas de código e extrai o primeiro objeto JSON."""
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    # Tenta encontrar o primeiro { ... }
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ParseError(f"Nenhum objeto JSON encontrado na resposta do LLM: {raw[:200]}")
    return json.loads(match.group())


def _parse_raw(raw: str) -> ParsedIntent:
    """Extrai e valida o JSON do LLM. Lança ParseError em qualquer falha."""
    try:
        data = _extract_json(raw)
    except json.JSONDecodeError as e:
        raise ParseError(f"JSON inválido: {e} | raw={raw[:200]}")

    # Normaliza intent para lowercase
    intent_raw = str(data.get("intent", "")).lower().strip()
    if intent_raw not in _INTENT_VALUES:
        intent_raw = "duvida"   # graceful fallback: desconhecido → duvida

    try:
        return ParsedIntent(
            intent=IntentType(intent_raw),
            confidence=float(data.get("confidence", 0.5)),
            entities=EntitySet(**(data.get("entities") or {})),
            risk_level=data.get("risk_level", "low"),
            needs_clarification=bool(data.get("needs_clarification", False)),
        )
    except Exception as e:
        raise ParseError(f"Erro ao montar ParsedIntent: {e} | data={data}")


async def semantic_parse(
    message:       str,
    context:       str,   # resumo/turnos anteriores montado pela session
    cliente_info:  str,   # nome e contexto da clínica (sem dados sensíveis)
) -> ParsedIntent:
    """
    Chama o LLM e retorna ParsedIntent validado.
    Em caso de falha: retorna intent DUVIDA com needs_clarification=True
    (nunca lança exceção para o caller — a conversa não para).
    """
    user_content = (
        f"Contexto da conversa:\n{context}\n\n"
        f"Clínica: {cliente_info}\n\n"
        f"Mensagem do paciente: {message}"
    )

    try:
        raw = await call_llm(
            system=SYSTEM_PROMPT,
            user=user_content,
        )
        return _parse_raw(raw)
    except ParseError:
        raise   # caller decide o que fazer com ParseError
    except Exception as e:
        # Falha de rede, timeout, etc. → clarify seguro
        raise ParseError(f"Falha ao chamar LLM: {e}")
