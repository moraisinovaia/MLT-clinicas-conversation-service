"""
Testes unitários para sql_route.execute_sql.

Cobre:
  - médico encontrado → retorna nome, CRM, especialidade
  - médico não encontrado → retorna _NO_INFO
  - sem medico_nome → retorna config da clínica
  - config da clínica não encontrada → retorna _NO_INFO
  - convenio nunca chega aqui (garantido pelo policy_engine)
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from app.routes.sql_route import execute_sql
from app.models.intent import ParsedIntent, IntentType, EntitySet


def _make_parsed(
    medico_nome: str | None = None,
    convenio: str | None = None,
) -> ParsedIntent:
    return ParsedIntent(
        intent=IntentType.DUVIDA,
        confidence=0.9,
        entities=EntitySet(medico_nome=medico_nome, convenio=convenio),
        risk_level="low",
        needs_clarification=False,
    )


def _make_db(fetchrow_return=None):
    db = AsyncMock()
    db.fetchrow = AsyncMock(return_value=fetchrow_return)
    return db


CLIENTE_ID = "d7d7b7cf-4ec0-437b-8377-d7555fc5ee6a"


# ── médico encontrado ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_medico_encontrado_retorna_dados_basicos():
    row = MagicMock()
    row.__getitem__ = lambda self, k: {
        "nome": "Dr. Guilherme Lucena Moura",
        "crm": "CRM-PE 12345",
        "especialidade": "Oftalmologia geral",
    }[k]

    db = _make_db(row)
    parsed = _make_parsed(medico_nome="Guilherme")

    msgs = await execute_sql(parsed, CLIENTE_ID, db)
    assert len(msgs) == 1
    text = msgs[0].text
    assert "Dr. Guilherme Lucena Moura" in text
    assert "CRM" in text
    assert "Oftalmologia" in text


@pytest.mark.asyncio
async def test_medico_nao_encontrado_retorna_no_info():
    db = _make_db(None)
    parsed = _make_parsed(medico_nome="Dr. Inexistente")

    msgs = await execute_sql(parsed, CLIENTE_ID, db)
    assert len(msgs) == 1
    assert "recepção" in msgs[0].text.lower() or "contato" in msgs[0].text.lower()


# ── config da clínica ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_sem_medico_retorna_config_clinica():
    row = MagicMock()
    row.__getitem__ = lambda self, k: {
        "nome_clinica": "Hospital de Olhos de Petrolina",
        "endereco": "Rua Exemplo, 123",
        "telefone_publico": "(87) 99961-1057",
        "horario_funcionamento": "Seg-Sex 7h-17h",
    }[k]

    db = _make_db(row)
    parsed = _make_parsed()  # sem medico_nome nem convenio

    msgs = await execute_sql(parsed, CLIENTE_ID, db)
    assert len(msgs) == 1
    text = msgs[0].text
    assert "Hospital de Olhos" in text
    assert "Rua Exemplo" in text
    assert "(87)" in text


@pytest.mark.asyncio
async def test_config_nao_encontrada_retorna_no_info():
    db = _make_db(None)
    parsed = _make_parsed()

    msgs = await execute_sql(parsed, CLIENTE_ID, db)
    assert len(msgs) == 1
    assert "recepção" in msgs[0].text.lower() or "contato" in msgs[0].text.lower()


# ── convenio nunca deve chegar aqui ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_convenio_sem_medico_vai_para_config_nao_responde_elegibilidade():
    """
    policy_engine garante que convenio vai para workflow.
    Se por algum motivo chegar em sql_route (sem medico_nome), deve retornar
    config da clínica — NUNCA dados de elegibilidade local.
    """
    row = MagicMock()
    row.__getitem__ = lambda self, k: {
        "nome_clinica": "HOP",
        "endereco": None,
        "telefone_publico": None,
        "horario_funcionamento": None,
    }[k]

    db = _make_db(row)
    parsed = _make_parsed(convenio="Unimed")  # convenio set mas sem medico_nome

    msgs = await execute_sql(parsed, CLIENTE_ID, db)
    text = msgs[0].text
    # Não deve conter nenhuma afirmação sobre elegibilidade
    assert "aceita" not in text.lower()
    assert "atende pelo" not in text.lower()
    assert "convenio" not in text.lower()
