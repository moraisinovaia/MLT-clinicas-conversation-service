"""
Rota sql — Fase 2.

Queries em:
  - convenios_medico  → convênios aceitos por médico
  - medicos           → dados cadastrais do médico
  - configuracoes_clinica → endereço, telefone, horário
  - procedimentos_clinica → exames disponíveis

Fonte: SQL local = informativo. GT Inova API = autoritativo no agendamento.
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

    # ── Convênios aceitos por médico específico ───────────────────────────────
    if e.medico_nome and not e.convenio:
        rows = await db.fetch(
            """
            SELECT m.nome_completo, array_agg(cm.convenio_nome ORDER BY cm.convenio_nome) AS convenios
            FROM medicos m
            JOIN convenios_medico cm ON cm.medico_id = m.id
            WHERE m.cliente_id = $1
              AND LOWER(m.nome_completo) ILIKE $2
            GROUP BY m.nome_completo
            LIMIT 1
            """,
            cliente_id,
            f"%{e.medico_nome.lower()}%",
        )
        if rows:
            r = rows[0]
            convs = ", ".join(r["convenios"])
            return [OutboundMessage(
                text=f"{r['nome_completo']} atende pelos convênios: {convs}."
            )]
        return [OutboundMessage(text=_NO_INFO)]

    # ── Clínica aceita determinado convênio? ──────────────────────────────────
    if e.convenio and not e.medico_nome:
        canonical = e.convenio_canonico or e.convenio
        row = await db.fetchrow(
            """
            SELECT COUNT(*) AS total
            FROM convenios_medico cm
            JOIN medicos m ON m.id = cm.medico_id
            WHERE m.cliente_id     = $1
              AND cm.convenio_nome ILIKE $2
            """,
            cliente_id,
            f"%{canonical}%",
        )
        if row and row["total"] > 0:
            return [OutboundMessage(
                text=f"Sim, a clínica aceita {canonical}. Para confirmar a disponibilidade "
                     f"no agendamento, informe o convênio no momento da marcação."
            )]
        return [OutboundMessage(
            text=f"Não encontrei {canonical} na nossa lista de convênios. "
                 f"Entre em contato com a recepção para confirmar."
        )]

    # ── Convênios aceitos por médico + convênio (dupla entidade) ─────────────
    if e.medico_nome and e.convenio:
        canonical = e.convenio_canonico or e.convenio
        row = await db.fetchrow(
            """
            SELECT m.nome_completo
            FROM medicos m
            JOIN convenios_medico cm ON cm.medico_id = m.id
            WHERE m.cliente_id     = $1
              AND LOWER(m.nome_completo) ILIKE $2
              AND cm.convenio_nome ILIKE $3
            LIMIT 1
            """,
            cliente_id,
            f"%{e.medico_nome.lower()}%",
            f"%{canonical}%",
        )
        if row:
            return [OutboundMessage(
                text=f"Sim, {row['nome_completo']} atende pelo {canonical}."
            )]
        return [OutboundMessage(
            text=f"Não encontrei {e.medico_nome} atendendo pelo {canonical}. "
                 f"Confirme com a recepção."
        )]

    # ── Endereço / contato da clínica ─────────────────────────────────────────
    row = await db.fetchrow(
        """
        SELECT nome_clinica, endereco_completo, telefone_publico, horario_funcionamento
        FROM configuracoes_clinica
        WHERE cliente_id = $1
        LIMIT 1
        """,
        cliente_id,
    )
    if row:
        parts = []
        if row["nome_clinica"]:
            parts.append(row["nome_clinica"])
        if row["endereco_completo"]:
            parts.append(f"Endereço: {row['endereco_completo']}")
        if row["telefone_publico"]:
            parts.append(f"Telefone: {row['telefone_publico']}")
        if row["horario_funcionamento"]:
            parts.append(f"Horário: {row['horario_funcionamento']}")
        return [OutboundMessage(text="\n".join(parts))]

    return [OutboundMessage(text=_NO_INFO)]
