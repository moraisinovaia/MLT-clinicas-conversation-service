"""
Gerenciamento de sessão — load/save de ia_contexto_sessao.

Sliding window de ultimos_turnos: máximo 6 turnos.
"""
# resolve_rag_ids também exportado daqui — precisa de asyncpg (I/O)
from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from dataclasses import dataclass
import asyncpg

from app.models.state import ConversationState
from app.models.intent import EntitySet, IntentType, ParsedIntent

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def safe_json_load(value, default):
    """
    Fix 2: asyncpg pode devolver JSONB já como dict/list (não como str).
    Aceita str, dict, list ou None — nunca lança TypeError.
    """
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value          # asyncpg já desserializou
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return default
    return default


def merge_entities(existing: EntitySet, new: EntitySet) -> EntitySet:
    """
    Merge determinístico para execução/persistência de workflow.

    Nunca sobrescreve com None.
    Campos existentes são preservados; novos campos preenchem os vazios.
    Não deve ser usado como fonte implícita de verdade para roteamento da
    mensagem atual — routing usa current_turn_entities separadamente.
    """
    merged = existing.model_dump()
    for field, value in new.model_dump().items():
        if value is not None:
            merged[field] = value
    return EntitySet(**merged)


# Campos obrigatórios por intent para coletar progressivamente
_REQUIRED_FIELDS: dict[IntentType, list[str]] = {
    IntentType.AGENDAR:   ["medico_nome", "atendimento_nome", "data_preferida", "convenio"],
    IntentType.REMARCAR:  ["agendamento_id", "data_preferida"],
    IntentType.CANCELAR:  ["agendamento_id"],
    IntentType.CONFIRMAR: ["agendamento_id"],
    IntentType.FILA:      ["medico_nome", "atendimento_nome"],
}


def compute_missing_fields(intent: ParsedIntent, entities: EntitySet) -> list[str]:
    """
    Fix 4: calcula quais campos ainda faltam para executar o intent.
    Retorna lista de campos ausentes na ordem de coleta.
    """
    required = _REQUIRED_FIELDS.get(intent.intent, [])
    entities_dict = entities.model_dump()
    return [f for f in required if not entities_dict.get(f)]


# ── Modelo de contexto ───────────────────────────────────────────────────────

@dataclass
class SessionContext:
    session_id:         str
    cliente_id:         str
    estado_atual:       ConversationState
    owner_atual:        str                  # "ia" | "humano" | "sistema"
    ultimos_turnos:     list[dict]           # [{role, content, ts}] max 6
    resumo_sumarizado:  str
    entities_coletadas: EntitySet
    dados_faltantes:    list[str]
    contador_turnos:    int
    fila_id:            str | None
    nome_paciente:      str | None


# ── Load ─────────────────────────────────────────────────────────────────────

async def load_session(
    session_id: str,
    cliente_id: str,
    db:         asyncpg.Connection,
) -> SessionContext:
    """Carrega sessão existente ou inicializa nova."""
    row = await db.fetchrow(
        """
        SELECT
            estado_atual, owner_atual, ultimos_turnos, resumo_sumarizado,
            entities_coletadas, dados_faltantes, contador_turnos,
            fila_id, nome_paciente_detectado
        FROM ia_contexto_sessao
        WHERE session_id = $1 AND cliente_id = $2
        ORDER BY ultima_mensagem_em DESC
        LIMIT 1
        """,
        session_id,
        cliente_id,
    )

    if not row:
        return SessionContext(
            session_id=session_id,
            cliente_id=cliente_id,
            estado_atual=ConversationState.TRIAGEM,
            owner_atual="ia",
            ultimos_turnos=[],
            resumo_sumarizado="",
            entities_coletadas=EntitySet(),
            dados_faltantes=[],
            contador_turnos=0,
            fila_id=None,
            nome_paciente=None,
        )

    # Fix 2: usar safe_json_load em todos os campos JSONB
    return SessionContext(
        session_id=session_id,
        cliente_id=cliente_id,
        estado_atual=ConversationState(row["estado_atual"]),
        owner_atual=row["owner_atual"] or "ia",
        ultimos_turnos=safe_json_load(row["ultimos_turnos"], []),
        resumo_sumarizado=row["resumo_sumarizado"] or "",
        entities_coletadas=EntitySet(**safe_json_load(row["entities_coletadas"], {})),
        dados_faltantes=safe_json_load(row["dados_faltantes"], []),
        contador_turnos=row["contador_turnos"] or 0,
        fila_id=row["fila_id"],
        nome_paciente=row["nome_paciente_detectado"],
    )


# ── Sliding window ───────────────────────────────────────────────────────────

def push_turn(turnos: list[dict], role: str, content: str) -> list[dict]:
    """Adiciona turno e mantém sliding window de 6."""
    turnos.append({
        "role":    role,
        "content": content,
        "ts":      datetime.now(timezone.utc).isoformat(),
    })
    return turnos[-6:]


def build_context_string(ctx: SessionContext) -> str:
    """Monta string de contexto para o LLM."""
    parts = []
    if ctx.resumo_sumarizado:
        parts.append(f"[Resumo anterior]: {ctx.resumo_sumarizado}")
    for t in ctx.ultimos_turnos:
        label = "Paciente" if t["role"] == "paciente" else "Assistente"
        parts.append(f"{label}: {t['content']}")
    return "\n".join(parts) if parts else "(sem histórico)"


# ── Save ─────────────────────────────────────────────────────────────────────

async def save_session(
    ctx:         SessionContext,
    db:          asyncpg.Connection,
    novo_estado: ConversationState,
    resposta:    str,
    mensagem:    str,
) -> None:
    """Persiste o novo estado, entidades mescladas e turnos atualizados."""
    ctx.ultimos_turnos = push_turn(ctx.ultimos_turnos, "paciente", mensagem)
    ctx.ultimos_turnos = push_turn(ctx.ultimos_turnos, "assistente", resposta)
    ctx.contador_turnos += 1

    await db.execute(
        """
        INSERT INTO ia_contexto_sessao (
            id, session_id, cliente_id, whatsapp_number,
            estado_atual, owner_atual, ultimos_turnos, resumo_sumarizado,
            entities_coletadas, dados_faltantes, contador_turnos,
            fila_id, nome_paciente_detectado, ultima_mensagem_em
        ) VALUES (
            gen_random_uuid(), $1, $2, $1,
            $3, $4, $5::jsonb, $6,
            $7::jsonb, $8::jsonb, $9,
            $10, $11, NOW()
        )
        ON CONFLICT (session_id, cliente_id)
        DO UPDATE SET
            estado_atual            = EXCLUDED.estado_atual,
            owner_atual             = EXCLUDED.owner_atual,
            ultimos_turnos          = EXCLUDED.ultimos_turnos,
            resumo_sumarizado       = EXCLUDED.resumo_sumarizado,
            entities_coletadas      = EXCLUDED.entities_coletadas,
            dados_faltantes         = EXCLUDED.dados_faltantes,
            contador_turnos         = EXCLUDED.contador_turnos,
            fila_id                 = EXCLUDED.fila_id,
            nome_paciente_detectado = EXCLUDED.nome_paciente_detectado,
            ultima_mensagem_em      = NOW()
        """,
        ctx.session_id,
        ctx.cliente_id,
        novo_estado.value,
        ctx.owner_atual,
        json.dumps(ctx.ultimos_turnos, ensure_ascii=False),
        ctx.resumo_sumarizado,
        json.dumps(ctx.entities_coletadas.model_dump(), ensure_ascii=False),
        json.dumps(ctx.dados_faltantes, ensure_ascii=False),
        ctx.contador_turnos,
        ctx.fila_id,
        ctx.nome_paciente,
    )


# ── Resolução de IDs para filtros RAG ────────────────────────────────────────

async def resolve_rag_ids(
    medico_nome:      str | None,
    atendimento_nome: str | None,
    cliente_id:       str,
    db:               asyncpg.Connection,
) -> tuple[str | None, str | None]:
    """
    Resolve medico_nome → doctor_id e atendimento_nome → procedure_id
    para uso como filtros fortes no hybrid_search_v2.

    Retorna (doctor_id, procedure_id) — None quando não encontrado.
    Nunca lança exceção: falha silenciosa mantém busca sem filtro.
    """
    doctor_id    = None
    procedure_id = None

    try:
        if medico_nome:
            row = await db.fetchrow(
                """
                SELECT id FROM medicos
                WHERE cliente_id = $1
                  AND nome ILIKE $2
                LIMIT 1
                """,
                cliente_id,
                f"%{medico_nome}%",
            )
            if row and row["id"]:
                doctor_id = str(row["id"])

        if atendimento_nome:
            row = await db.fetchrow(
                """
                SELECT id FROM procedimentos_clinica
                WHERE cliente_id = $1
                  AND (nome ILIKE $2 OR $3 = ANY(aliases))
                  AND disponivel = true
                LIMIT 1
                """,
                cliente_id,
                f"%{atendimento_nome}%",
                atendimento_nome.lower(),
            )
            if row and row["id"]:
                procedure_id = str(row["id"])
    except Exception as e:
        logger.warning("resolve_rag_ids_failed cliente=%s err=%s", cliente_id, e)

    return doctor_id, procedure_id
