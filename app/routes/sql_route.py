"""
Rota sql — Fase 2.

Queries em:
  - medicos           → dados cadastrais básicos (CRM, especialidade)
  - configuracoes_clinica → endereço, telefone, horário

Fonte: SQL local = informativo. Elegibilidade de convênio nunca responde aqui —
a autoridade é a GT Inova (roteada pelo policy_engine via workflow).
"""
from __future__ import annotations
import asyncpg
from app.models.intent import ParsedIntent
from app.models.conversation import OutboundMessage

_NO_INFO = "Para informações atualizadas, entre em contato com a recepção da clínica."


async def execute_sql(
    intent:     ParsedIntent,
    cliente_id: str,
    db:         asyncpg.Connection,
) -> list[OutboundMessage]:
    e = intent.entities

    # ── Dados básicos do médico (CRM, especialidade) ──────────────────────────
    # Convenio nunca chega aqui — policy_engine garante via Regra 4/4b.
    if e.medico_nome:
        row = await db.fetchrow(
            """
            SELECT nome, crm, especialidade
            FROM medicos
            WHERE cliente_id = $1
              AND LOWER(nome) ILIKE $2
              AND ativo = true
            LIMIT 1
            """,
            cliente_id,
            f"%{e.medico_nome.lower()}%",
        )
        if row:
            parts = [row["nome"]]
            if row["crm"]:
                parts.append(f"CRM: {row['crm']}")
            if row["especialidade"]:
                parts.append(f"Especialidade: {row['especialidade']}")
            return [OutboundMessage(text="\n".join(parts))]
        return [OutboundMessage(text=_NO_INFO)]

    # ── Endereço / contato da clínica ─────────────────────────────────────────
    row = await db.fetchrow(
        """
        SELECT nome_clinica, telefone_publico
        FROM configuracoes_clinica
        WHERE id = $1
        LIMIT 1
        """,
        cliente_id,
    )
    if row:
        parts = []
        if row["nome_clinica"]:
            parts.append(row["nome_clinica"])
        if row["telefone_publico"]:
            parts.append(f"Telefone: {row['telefone_publico']}")
        return [OutboundMessage(text="\n".join(parts))]

    return [OutboundMessage(text=_NO_INFO)]
