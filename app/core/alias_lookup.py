"""
Canonização determinística de convênio via knowledge_aliases.

3 camadas (nunca LLM):
  1. normalize_text(raw)
  2. SELECT canonical_name FROM knowledge_aliases WHERE normalized_alias = $1
  3. Se não encontrado → passa valor cru para a API
     → se API retorna CONVENIO_NAO_ACEITO: usar error.message direto
"""
from __future__ import annotations
import unicodedata
import asyncpg


def normalize_text(text: str) -> str:
    """Remove acentos, lowercase, colapsa espaços."""
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_str = nfkd.encode("ASCII", "ignore").decode("ASCII")
    return " ".join(ascii_str.lower().split())


async def lookup_convenio(
    raw:        str,
    cliente_id: str,
    db:         asyncpg.Connection,
) -> str:
    """
    Retorna o canonical_name se encontrado em knowledge_aliases,
    ou o raw original se não houver alias cadastrado.

    Autoridade: GT Inova API é autoritativa — SQL é apenas normalizador.
    """
    normalized = normalize_text(raw)
    row = await db.fetchrow(
        """
        SELECT canonical_name
        FROM knowledge_aliases
        WHERE cliente_id       = $1
          AND entity_type      = 'convenio'
          AND normalized_alias = $2
        """,
        cliente_id,
        normalized,
    )
    return row["canonical_name"] if row else raw


async def canonicalize_entities(
    entities:   object,  # EntitySet
    cliente_id: str,
    db:         asyncpg.Connection,
) -> object:
    """
    Preenche entities.convenio_canonico se entities.convenio estiver presente.
    Modifica in-place e retorna o objeto.
    """
    if entities.convenio:
        entities.convenio_canonico = await lookup_convenio(
            entities.convenio, cliente_id, db
        )
    return entities
