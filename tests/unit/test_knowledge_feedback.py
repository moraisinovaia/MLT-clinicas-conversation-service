"""
Testes unitários para integração do knowledge_feedback.

Cobre:
  - execute_rag retorna (messages, feedback_id) quando há chunks
  - execute_rag retorna (messages, None) quando não há chunks
  - execute_rag retorna (messages, None) em caso de conflito
  - _write_knowledge_feedback escreve a linha correta
  - PATCH /feedback/{id} atualiza was_helpful → 204
  - PATCH /feedback/{id} retorna 404 para ID inexistente
  - PATCH /feedback/{id} com body vazio → 204 sem erro
"""
import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from app.routes.rag import execute_rag, _write_knowledge_feedback
from app.models.intent import ParsedIntent, IntentType, EntitySet
from app.core.policy_engine import RagFilters


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_parsed(risk_level="low") -> ParsedIntent:
    return ParsedIntent(
        intent=IntentType.DUVIDA_ORIENTACAO,
        confidence=0.9,
        entities=EntitySet(),
        risk_level=risk_level,
        needs_clarification=False,
        mensagem_usuario="qual o horário de funcionamento?",
    )


def _make_filters() -> RagFilters:
    return RagFilters(risk_max="low", source_types=["policy"])


def _make_row(chunk_id="aaaa-bbbb", rrf_score=0.8):
    row = MagicMock()
    row.__getitem__ = lambda self, k: {
        "chunk_id":   chunk_id,
        "rrf_score":  rrf_score,
        "source_type": "policy",
        "risk_level": "low",
        "chunk_text": "A clínica funciona de segunda a sexta, das 7h às 17h.",
    }[k]
    row.get = lambda k, d=None: row[k] if k in ["chunk_id","rrf_score","source_type","risk_level","chunk_text"] else d
    return row


# ── execute_rag retorna tupla ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_rag_returns_tuple_with_feedback_id():
    db = AsyncMock()
    db.fetch = AsyncMock(return_value=[_make_row()])
    db.execute = AsyncMock()

    with (
        patch("app.routes.rag._get_query_embedding", return_value=[0.1] * 1536),
        patch("app.routes.rag.call_llm", return_value="A clínica funciona das 7h às 17h."),
    ):
        result = await execute_rag(
            _make_parsed(), _make_filters(), "cliente-uuid", db,
            session_id="sess-1", query="qual o horário?",
        )

    assert isinstance(result, tuple)
    messages, feedback_id = result
    assert len(messages) == 1
    assert feedback_id is not None
    assert len(feedback_id) == 36   # UUID v4


@pytest.mark.asyncio
async def test_execute_rag_returns_none_feedback_when_no_chunks():
    db = AsyncMock()
    db.fetch = AsyncMock(return_value=[])
    db.execute = AsyncMock()

    with patch("app.routes.rag._get_query_embedding", return_value=[0.1] * 1536):
        messages, feedback_id = await execute_rag(
            _make_parsed(), _make_filters(), "cliente-uuid", db,
        )

    assert feedback_id is None


@pytest.mark.asyncio
async def test_execute_rag_returns_none_feedback_on_embedding_error():
    db = AsyncMock()
    db.execute = AsyncMock()

    with patch("app.routes.rag._get_query_embedding", side_effect=Exception("timeout")):
        messages, feedback_id = await execute_rag(
            _make_parsed(), _make_filters(), "cliente-uuid", db,
        )

    assert feedback_id is None
    assert "recepção" in messages[0].text.lower() or "orientações" in messages[0].text.lower()


# ── _write_knowledge_feedback ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_write_knowledge_feedback_inserts_correct_columns():
    db = AsyncMock()
    db.execute = AsyncMock()

    fid = await _write_knowledge_feedback(
        db=db,
        cliente_id="cliente-uuid",
        session_id="sess-1",
        query="qual o preparo?",
        route="rag",
        chunks_used=[{"chunk_id": "abc", "rrf_score": 0.9}],
        answer_text="Jejum de 6 horas.",
        confidence=0.9,
    )

    assert len(fid) == 36  # UUID v4
    db.execute.assert_called_once()
    call_args = db.execute.call_args[0]
    assert "INSERT INTO knowledge_feedback" in call_args[0]
    # feedback_id é o primeiro argumento posicional depois da query
    assert call_args[1] == fid
    assert call_args[2] == "cliente-uuid"
    assert call_args[4] == "qual o preparo?"
    assert call_args[5] == "rag"


@pytest.mark.asyncio
async def test_write_knowledge_feedback_swallows_db_error():
    db = AsyncMock()
    db.execute = AsyncMock(side_effect=Exception("connection lost"))

    # Não deve propagar a exceção
    fid = await _write_knowledge_feedback(
        db=db, cliente_id="c", session_id="s", query="q",
        route="rag", chunks_used=[], answer_text="resp", confidence=0.5,
    )
    assert isinstance(fid, str)


# ── PATCH /feedback/{id} ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_patch_feedback_was_helpful_returns_204():
    from httpx import AsyncClient, ASGITransport
    from app.main import app

    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(return_value="UPDATE 1")
    mock_pool = AsyncMock()
    mock_pool.acquire = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=mock_db),
        __aexit__=AsyncMock(return_value=False),
    ))

    with patch("app.api.v1.feedback.get_pool", return_value=mock_pool):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.patch(
                "/api/v1/feedback/some-uuid",
                json={"was_helpful": True},
            )

    assert resp.status_code == 204


@pytest.mark.asyncio
async def test_patch_feedback_not_found_returns_404():
    from httpx import AsyncClient, ASGITransport
    from app.main import app

    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(return_value="UPDATE 0")
    mock_pool = AsyncMock()
    mock_pool.acquire = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=mock_db),
        __aexit__=AsyncMock(return_value=False),
    ))

    with patch("app.api.v1.feedback.get_pool", return_value=mock_pool):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.patch(
                "/api/v1/feedback/nonexistent-uuid",
                json={"was_helpful": False},
            )

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_patch_feedback_empty_body_returns_204():
    from httpx import AsyncClient, ASGITransport
    from app.main import app

    mock_pool = AsyncMock()  # execute nunca deve ser chamado

    with patch("app.api.v1.feedback.get_pool", return_value=mock_pool):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.patch(
                "/api/v1/feedback/any-uuid",
                json={},
            )

    assert resp.status_code == 204
    mock_pool.acquire.assert_not_called()
