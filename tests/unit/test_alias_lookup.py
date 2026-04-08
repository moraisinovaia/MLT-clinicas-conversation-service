"""
Testa canonização de convênio sem banco — mock do asyncpg.
Critério 1.7: "unimed" → "UNIMED REGIONAL" para clínica correta.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from app.core.alias_lookup import lookup_convenio, normalize_text, canonicalize_entities
from app.models.intent import EntitySet


CLIENTE_ID = "00000000-0000-0000-0000-000000000001"


def make_db(canonical_name: str | None):
    """Mock de asyncpg.Connection.fetchrow."""
    db = AsyncMock()
    if canonical_name:
        db.fetchrow.return_value = {"canonical_name": canonical_name}
    else:
        db.fetchrow.return_value = None
    return db


# ── normalize_text ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("Unimed",          "unimed"),
    ("UNIMED VSF",      "unimed vsf"),
    ("unimed regional", "unimed regional"),
    ("Particular",      "particular"),
    ("HGU",             "hgu"),
])
def test_normalize_text(raw, expected):
    assert normalize_text(raw) == expected


# ── lookup_convenio ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_alias_found_returns_canonical():
    db = make_db("UNIMED REGIONAL")
    result = await lookup_convenio("Unimed", CLIENTE_ID, db)
    assert result == "UNIMED REGIONAL"
    # Confirma que a query usou o valor normalizado
    db.fetchrow.assert_awaited_once()
    call_args = db.fetchrow.call_args[0]
    assert "unimed" in call_args  # normalized_alias passado

@pytest.mark.asyncio
async def test_alias_not_found_returns_raw():
    db = make_db(None)
    result = await lookup_convenio("ConvenioDesconhecido", CLIENTE_ID, db)
    assert result == "ConvenioDesconhecido"

@pytest.mark.asyncio
async def test_unimed_vsf_maps_to_regional():
    db = make_db("UNIMED REGIONAL")
    result = await lookup_convenio("UNIMED VSF", CLIENTE_ID, db)
    assert result == "UNIMED REGIONAL"

@pytest.mark.asyncio
async def test_particular_maps_to_particular():
    db = make_db("PARTICULAR")
    result = await lookup_convenio("particular", CLIENTE_ID, db)
    assert result == "PARTICULAR"


# ── canonicalize_entities ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_canonicalize_fills_convenio_canonico():
    db = make_db("UNIMED REGIONAL")
    entities = EntitySet(convenio="unimed vsf")
    result = await canonicalize_entities(entities, CLIENTE_ID, db)
    assert result.convenio == "unimed vsf"           # bruto preservado
    assert result.convenio_canonico == "UNIMED REGIONAL"

@pytest.mark.asyncio
async def test_canonicalize_no_convenio_is_noop():
    db = make_db(None)
    entities = EntitySet()
    result = await canonicalize_entities(entities, CLIENTE_ID, db)
    assert result.convenio_canonico is None
    db.fetchrow.assert_not_awaited()
