from __future__ import annotations

from copy import deepcopy
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.v1.conversation import _sanitize_entities_for_log
from app.core.session import SessionContext
from app.main import app
from app.models.conversation import OutboundMessage
from app.models.intent import EntitySet, IntentType, ParsedIntent
from app.models.state import ConversationState


CLIENTE_ID = "d7d7b7cf-4ec0-437b-8377-d7555fc5ee6a"
SESSION_ID = "sessao-teste"


class _FakeDB:
    async def execute(self, *args, **kwargs):
        return None

    async def fetchval(self, query, *args):
        if "acquire_conversation_lock" in query:
            return True
        if "release_conversation_lock" in query:
            return True
        return None

    async def fetchrow(self, query, *args):
        if "FROM configuracoes_clinica" in query:
            return {
                "transbordo_humano_ativo": False,
                "mensagem_transbordo": None,
                "mensagem_fallback_sem_humano": None,
            }
        return None


class _FakeAcquire:
    def __init__(self, db):
        self.db = db

    async def __aenter__(self):
        return self.db

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakePool:
    def __init__(self, db):
        self.db = db

    def acquire(self):
        return _FakeAcquire(self.db)


class _InMemorySessionStore:
    def __init__(self):
        self._store: dict[tuple[str, str], SessionContext] = {}

    async def load(self, session_id: str, cliente_id: str, db) -> SessionContext:
        key = (session_id, cliente_id)
        if key not in self._store:
            self._store[key] = SessionContext(
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
        return deepcopy(self._store[key])

    async def save(
        self,
        ctx: SessionContext,
        db,
        novo_estado: ConversationState,
        resposta: str,
        mensagem: str,
    ) -> None:
        persisted = deepcopy(ctx)
        persisted.estado_atual = novo_estado
        persisted.contador_turnos += 1
        key = (persisted.session_id, persisted.cliente_id)
        self._store[key] = persisted


def _make_parsed(
    *,
    message: str,
    intent: IntentType,
    entities: EntitySet | None = None,
    risk_level: str = "low",
) -> ParsedIntent:
    return ParsedIntent(
        intent=intent,
        confidence=0.95,
        entities=entities or EntitySet(),
        risk_level=risk_level,
        needs_clarification=False,
        mensagem_usuario=message,
    )


@pytest.mark.asyncio
async def test_agendamento_followup_preserva_contexto_transacional():
    store = _InMemorySessionStore()
    pool = _FakePool(_FakeDB())
    workflow_calls: list[dict] = []

    async def fake_semantic_parse(message: str, context: str, cliente_info: str, media_type: str):
        parsed_by_message = {
            "quero marcar com Dr. Guilherme": _make_parsed(
                message=message,
                intent=IntentType.AGENDAR,
                entities=EntitySet(medico_nome="Dr. Guilherme"),
            ),
            "amanhã à tarde": _make_parsed(
                message=message,
                intent=IntentType.AGENDAR,
                entities=EntitySet(data_preferida="amanhã", periodo="tarde"),
            ),
        }
        return parsed_by_message[message].model_copy(deep=True)

    async def fake_workflow(parsed, state, missing_fields, cliente_id, session_id, db, gt_inova):
        workflow_calls.append(
            {
                "message": parsed.mensagem_usuario,
                "entities": parsed.entities.model_dump(exclude_none=True),
                "missing_fields": list(missing_fields),
                "state": state,
            }
        )
        if missing_fields:
            return [OutboundMessage(text=f"faltam:{','.join(missing_fields)}")], ConversationState.COLETANDO_DADOS.value
        return [OutboundMessage(text="confirme o agendamento")], ConversationState.CONFIRMANDO.value

    async def fake_canonicalize_entities(entities, *_args):
        return entities

    with (
        patch("app.api.v1.conversation.get_pool", new=AsyncMock(return_value=pool)),
        patch("app.api.v1.conversation.load_session", new=store.load),
        patch("app.api.v1.conversation.save_session", new=store.save),
        patch("app.api.v1.conversation.pre_parse", return_value=None),
        patch("app.api.v1.conversation.semantic_parse", new=fake_semantic_parse),
        patch("app.api.v1.conversation.canonicalize_entities", new=fake_canonicalize_entities),
        patch("app.api.v1.conversation.workflow_route.execute_workflow", new=fake_workflow),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            first = await client.post(
                "/api/v1/conversation",
                json={"session_id": SESSION_ID, "cliente_id": CLIENTE_ID, "message": "quero marcar com Dr. Guilherme"},
            )
            second = await client.post(
                "/api/v1/conversation",
                json={"session_id": SESSION_ID, "cliente_id": CLIENTE_ID, "message": "amanhã à tarde"},
            )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["new_state"] == ConversationState.COLETANDO_DADOS.value
    assert second.json()["new_state"] == ConversationState.COLETANDO_DADOS.value
    assert len(workflow_calls) == 2
    assert workflow_calls[1]["entities"]["medico_nome"] == "Dr. Guilherme"
    assert workflow_calls[1]["entities"]["data_preferida"] == "amanhã"
    assert workflow_calls[1]["entities"]["periodo"] == "tarde"
    assert "medico_nome" not in workflow_calls[1]["missing_fields"]
    assert workflow_calls[1]["missing_fields"] == ["atendimento_nome", "convenio"]


@pytest.mark.asyncio
async def test_pergunta_informacional_no_meio_nao_quebra_continuidade_do_workflow():
    store = _InMemorySessionStore()
    pool = _FakePool(_FakeDB())
    workflow_calls: list[dict] = []
    sql_calls: list[dict] = []

    async def fake_semantic_parse(message: str, context: str, cliente_info: str, media_type: str):
        parsed_by_message = {
            "quero marcar com Dr. Guilherme": _make_parsed(
                message=message,
                intent=IntentType.AGENDAR,
                entities=EntitySet(medico_nome="Dr. Guilherme"),
            ),
            "qual o endereço?": _make_parsed(
                message=message,
                intent=IntentType.DUVIDA,
                entities=EntitySet(),
            ),
            "pode ser amanhã?": _make_parsed(
                message=message,
                intent=IntentType.AGENDAR,
                entities=EntitySet(data_preferida="amanhã"),
            ),
        }
        return parsed_by_message[message].model_copy(deep=True)

    async def fake_workflow(parsed, state, missing_fields, cliente_id, session_id, db, gt_inova):
        workflow_calls.append(
            {
                "message": parsed.mensagem_usuario,
                "entities": parsed.entities.model_dump(exclude_none=True),
                "missing_fields": list(missing_fields),
                "state": state,
            }
        )
        return [OutboundMessage(text=f"workflow:{','.join(missing_fields)}")], ConversationState.COLETANDO_DADOS.value

    async def fake_sql(parsed, cliente_id, db):
        sql_calls.append(
            {
                "message": parsed.mensagem_usuario,
                "entities": parsed.entities.model_dump(exclude_none=True),
            }
        )
        return [OutboundMessage(text="Clínica Olhos\nEndereço: Rua Exemplo, 123")]

    async def fake_canonicalize_entities(entities, *_args):
        return entities

    with (
        patch("app.api.v1.conversation.get_pool", new=AsyncMock(return_value=pool)),
        patch("app.api.v1.conversation.load_session", new=store.load),
        patch("app.api.v1.conversation.save_session", new=store.save),
        patch("app.api.v1.conversation.pre_parse", return_value=None),
        patch("app.api.v1.conversation.semantic_parse", new=fake_semantic_parse),
        patch("app.api.v1.conversation.canonicalize_entities", new=fake_canonicalize_entities),
        patch("app.api.v1.conversation.workflow_route.execute_workflow", new=fake_workflow),
        patch("app.api.v1.conversation.sql_route.execute_sql", new=fake_sql),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            first = await client.post(
                "/api/v1/conversation",
                json={"session_id": SESSION_ID, "cliente_id": CLIENTE_ID, "message": "quero marcar com Dr. Guilherme"},
            )
            second = await client.post(
                "/api/v1/conversation",
                json={"session_id": SESSION_ID, "cliente_id": CLIENTE_ID, "message": "qual o endereço?"},
            )
            third = await client.post(
                "/api/v1/conversation",
                json={"session_id": SESSION_ID, "cliente_id": CLIENTE_ID, "message": "pode ser amanhã?"},
            )

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 200
    assert second.json()["messages"][0]["text"].startswith("Clínica Olhos")
    assert second.json()["new_state"] == ConversationState.COLETANDO_DADOS.value
    assert len(sql_calls) == 1
    assert sql_calls[0]["entities"] == {}
    assert len(workflow_calls) == 2
    assert workflow_calls[1]["entities"]["medico_nome"] == "Dr. Guilherme"
    assert workflow_calls[1]["entities"]["data_preferida"] == "amanhã"
    assert "medico_nome" not in workflow_calls[1]["missing_fields"]


def test_sanitize_entities_for_log_redacts_sensitive_fields():
    sanitized = _sanitize_entities_for_log(
        EntitySet(
            medico_nome="Dr. Guilherme",
            atendimento_nome="Consulta",
            convenio="Unimed",
            data_preferida="amanhã",
            paciente_nome="Gabriela",
            paciente_celular="87999999999",
            agendamento_id="ag-123",
        )
    )

    assert sanitized["medico_nome"] == "Dr. Guilherme"
    assert sanitized["atendimento_nome"] == "Consulta"
    assert sanitized["convenio"] == "Unimed"
    assert sanitized["data_preferida"] == "<redacted>"
    assert sanitized["paciente_nome"] == "<redacted>"
    assert sanitized["paciente_celular"] == "<redacted>"
    assert sanitized["agendamento_id"] == "<redacted>"
