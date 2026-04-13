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
  "risk_level":            "<low|medium|high>",
  "needs_clarification":   <true|false>,
  "is_operational_query":  <true|false>
}

Intents válidos:
  agendar | remarcar | cancelar | confirmar | fila | resposta_fila | transbordo
  duvida_preparo | duvida_orientacao | duvida_pos_procedimento | duvida
  social | saudacao | agradecimento | despedida | fora_escopo | emergencia

Regras de risk_level:
  high   → preparo de exame, jejum, suspensão de medicação, pós-procedimento
  medium → convênios, preços, políticas, autorização
  low    → saudação, bio de médico, endereço, FAQ geral

needs_clarification = true quando a mensagem for genuinamente ambígua (ex: "sim"
sem contexto). Para intents transacionais (agendar, cancelar, remarcar, fila,
etc.), use needs_clarification = false mesmo que faltem detalhes como médico,
data ou procedimento — o sistema de agendamento coleta essas informações
interativamente. needs_clarification = false também para perguntas operacionais
claras como "tem vaga?", "tem horário disponível?" — não há ambiguidade.

is_operational_query = true quando a resposta depende de dado em tempo real da API
de agendamentos: agenda, disponibilidade de vaga, elegibilidade de convênio,
confirmação de serviço ativo de médico, ou lista de médicos/especialidades
disponíveis (ex: "Dr. X faz Y?", "aceita Unimed?", "tem vaga?", "quais médicos
atendem?", "quem atende lá?", "quais especialidades?"). False para perguntas
explicativas, de preparo, perfil biográfico ou orientação clínica.

IMPORTANTE: Extraia entidades SOMENTE do conteúdo da mensagem atual.
NÃO herde entidades de mensagens anteriores no contexto.
Se a mensagem não mencionar médico, atendimento ou convênio, deixe null.

Regras de classificação de intents informativos:

duvida_preparo → pergunta sobre COMO se preparar, o que esperar, se vai dilatar, se precisa de jejum,
  se pode dirigir depois, o que é o exame, como funciona, quanto tempo dura, precisa de acompanhante.
  Exemplos: "precisa dilatar?", "o que é fundo de olho?", "como é a OCT?", "tem preparo?",
  "vou ficar com a visão embaçada?", "precisa de acompanhante?", "quanto tempo demora?",
  "como funciona a retinografia?", "o que acontece durante o exame?"
  risk_level: high se envolver dilatação, medicação ou jejum; low para dúvidas descritivas gerais.

duvida_orientacao → pergunta sobre REGRAS, POLÍTICAS ou ORIENTAÇÕES da clínica.
  Exemplos: "quando fico sabendo do resultado?", "qual o prazo de entrega do exame?",
  "como funciona o encaixe?", "preciso levar pedido médico?", "o que levar na consulta?",
  "posso levar criança junto?", "precisa de solicitação?", "como funciona o agendamento?"
  risk_level: medium.

duvida_pos_procedimento → pergunta sobre cuidados APÓS cirurgia ou procedimento já realizado.
  Exemplos: "posso lavar o olho?", "quando posso dirigir após a cirurgia?",
  "quanto tempo de repouso?", "colírio pós-operatório", "olho ainda está vermelho após a cirurgia",
  "o que fazer depois da operação de catarata?"
  risk_level: high.

duvida com is_operational_query=true → "quem faz", "qual médico faz", "vocês fazem", "tem esse exame",
  "quais médicos atendem", "quais especialidades", "tem vaga", "tem horário disponível",
  "qual horário disponível", "aceita X convênio". Para esses casos, needs_clarification = false.

duvida com is_operational_query=false → perguntas sobre horário de FUNCIONAMENTO da clínica,
  endereço, telefone, como chegar. Esses dados vêm do banco de dados local.
  IMPORTANTE: "horário de funcionamento", "que horas abre", "que horas fecha", "endereço",
  "telefone", "onde fica" → intent = duvida (não duvida_orientacao).

agendar → "quero marcar", "quero agendar", "preciso de uma consulta".
  needs_clarification = false mesmo sem médico ou data especificados.

Quando o paciente pergunta COMO é um exame ou QUAL O PREPARO, NÃO extraia atendimento_nome.
É uma dúvida clínica (duvida_preparo), não intenção de agendar.
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
            is_operational_query=bool(data.get("is_operational_query", False)),
        )
    except Exception as e:
        raise ParseError(f"Erro ao montar ParsedIntent: {e} | data={data}")


async def semantic_parse(
    message:       str,
    context:       str,   # resumo/turnos anteriores montado pela session
    cliente_info:  str,   # nome e contexto da clínica (sem dados sensíveis)
    media_type:    str = "text",
) -> ParsedIntent:
    """
    Chama o LLM e retorna ParsedIntent validado.
    Em caso de falha: retorna intent DUVIDA com needs_clarification=True
    (nunca lança exceção para o caller — a conversa não para).
    """
    media_hint = (
        f"Tipo de mídia recebida: {media_type}. "
        "Considere isso ao interpretar a intenção.\n\n"
    ) if media_type != "text" else ""

    user_content = (
        f"{media_hint}"
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
