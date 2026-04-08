"""
Testa o executor de workflow sem banco e sem API real.

Estratégia:
- gt_inova=None força o caminho "API indisponível" para chamadas à API.
- Mock de GTInovaClient cobre cenários de sucesso e códigos de erro.
- Banco mockado (asyncpg.Connection) com AsyncMock.
"""
from __future__ import annotations
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.models.intent import IntentType, ParsedIntent, EntitySet
from app.models.state import ConversationState
from app.models.conversation import OutboundMessage
from app.integrations.gt_inova import GTInovaClient, GTInovaOk, GTInovaError
from app.routes.workflow_route import execute_workflow


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_parsed(intent: IntentType, **entities) -> ParsedIntent:
    return ParsedIntent(
        intent=intent,
        confidence=0.9,
        entities=EntitySet(**entities),
        risk_level="low",
        needs_clarification=False,
    )


def make_db() -> AsyncMock:
    """Retorna um mock de asyncpg.Connection com execute e fetchval."""
    db = AsyncMock()
    db.execute = AsyncMock(return_value=None)
    db.fetchval = AsyncMock(return_value=None)
    return db


def make_gt(result) -> MagicMock:
    """GTInovaClient com um único método retornando `result`."""
    gt = MagicMock(spec=GTInovaClient)
    gt.schedule         = AsyncMock(return_value=result)
    gt.reschedule       = AsyncMock(return_value=result)
    gt.cancel           = AsyncMock(return_value=result)
    gt.confirm          = AsyncMock(return_value=result)
    gt.adicionar_fila   = AsyncMock(return_value=result)
    gt.responder_fila   = AsyncMock(return_value=result)
    gt.get_availability = AsyncMock(return_value=result)
    gt.list_appointments = AsyncMock(return_value=result)
    gt.check_patient    = AsyncMock(return_value=result)
    return gt


# ── TRANSBORDO ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_transbordo_retorna_estado_transbordo():
    parsed = make_parsed(IntentType.TRANSBORDO)
    msgs, next_state = await execute_workflow(
        parsed, ConversationState.TRIAGEM, [], "cli-1", "sess-1", make_db(), None
    )
    assert next_state == ConversationState.TRANSBORDO.value
    assert msgs


# ── AGENDAR — coleta progressiva ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_agendar_sem_dados_pede_medico():
    parsed = make_parsed(IntentType.AGENDAR)  # sem nenhuma entidade
    faltantes = ["medico_nome", "atendimento_nome", "data_preferida", "convenio"]
    msgs, next_state = await execute_workflow(
        parsed, ConversationState.TRIAGEM, faltantes, "cli-1", "sess-1", make_db(), None
    )
    assert next_state == ConversationState.COLETANDO_DADOS.value
    assert "médico" in msgs[0].text.lower()


@pytest.mark.asyncio
async def test_agendar_faltando_data_pede_data():
    parsed = make_parsed(
        IntentType.AGENDAR,
        medico_nome="Dr. Marcelo",
        atendimento_nome="Consulta",
        convenio="Unimed",
    )
    faltantes = ["data_preferida"]
    msgs, next_state = await execute_workflow(
        parsed, ConversationState.COLETANDO_DADOS, faltantes, "cli-1", "sess-1", make_db(), None
    )
    assert next_state == ConversationState.COLETANDO_DADOS.value
    assert "data" in msgs[0].text.lower()


@pytest.mark.asyncio
async def test_agendar_dados_completos_pede_confirmacao():
    parsed = make_parsed(
        IntentType.AGENDAR,
        medico_nome="Dr. Marcelo",
        atendimento_nome="Consulta",
        data_preferida="2026-05-10",
        convenio="Unimed",
    )
    msgs, next_state = await execute_workflow(
        parsed, ConversationState.TRIAGEM, [], "cli-1", "sess-1", make_db(), None
    )
    assert next_state == ConversationState.CONFIRMANDO.value
    assert "confirme" in msgs[0].text.lower()
    # Sem markdown
    assert "*" not in msgs[0].text


@pytest.mark.asyncio
async def test_agendar_confirmando_sem_api_retorna_indisponivel():
    # resposta_fila=SIM sinaliza que o paciente confirmou
    parsed = make_parsed(
        IntentType.AGENDAR,
        medico_nome="Dr. Marcelo",
        atendimento_nome="Consulta",
        data_preferida="2026-05-10",
        convenio="Unimed",
        resposta_fila="SIM",
    )
    msgs, next_state = await execute_workflow(
        parsed, ConversationState.CONFIRMANDO, [], "cli-1", "sess-1", make_db(), gt_inova=None
    )
    assert next_state == ConversationState.TRIAGEM.value
    assert "indisponível" in msgs[0].text.lower() or "recep" in msgs[0].text.lower()


@pytest.mark.asyncio
async def test_agendar_confirmando_sucesso():
    ok = GTInovaOk(data={"agendamento_id": "ag-123", "message": "Agendado!"})
    gt = make_gt(ok)
    parsed = make_parsed(
        IntentType.AGENDAR,
        medico_nome="Dr. Marcelo",
        atendimento_nome="Consulta",
        data_preferida="2026-05-10",
        convenio="Unimed",
        resposta_fila="SIM",
    )
    msgs, next_state = await execute_workflow(
        parsed, ConversationState.CONFIRMANDO, [], "cli-1", "sess-1", make_db(), gt
    )
    assert next_state == ConversationState.CONCLUIDO.value
    assert "agendado" in msgs[0].text.lower()


@pytest.mark.asyncio
async def test_agendar_slot_taken_oferece_disponibilidade():
    slot_err = GTInovaError(error_code="SLOT_TAKEN", message="Horário indisponível.")
    avail_ok  = GTInovaOk(data={"message": "Vagas: segunda 9h, terça 10h"})

    gt = MagicMock(spec=GTInovaClient)
    gt.schedule         = AsyncMock(return_value=slot_err)
    gt.get_availability = AsyncMock(return_value=avail_ok)

    parsed = make_parsed(
        IntentType.AGENDAR,
        medico_nome="Dr. Marcelo",
        atendimento_nome="Consulta",
        data_preferida="2026-05-10",
        convenio="Unimed",
        resposta_fila="SIM",
    )
    msgs, next_state = await execute_workflow(
        parsed, ConversationState.CONFIRMANDO, [], "cli-1", "sess-1", make_db(), gt
    )
    assert next_state == ConversationState.COLETANDO_DADOS.value
    assert len(msgs) == 2   # erro + disponibilidade
    assert "vagas" in msgs[1].text.lower()


@pytest.mark.asyncio
async def test_agendar_duplicate_booking():
    err = GTInovaError(error_code="DUPLICATE_BOOKING", message="Já tem consulta agendada.")
    gt = make_gt(err)
    parsed = make_parsed(
        IntentType.AGENDAR,
        medico_nome="Dr. Marcelo",
        atendimento_nome="Consulta",
        data_preferida="2026-05-10",
        convenio="Unimed",
        resposta_fila="SIM",
    )
    msgs, next_state = await execute_workflow(
        parsed, ConversationState.CONFIRMANDO, [], "cli-1", "sess-1", make_db(), gt
    )
    assert next_state == ConversationState.TRIAGEM.value
    assert "agendada" in msgs[0].text.lower()


@pytest.mark.asyncio
async def test_agendar_convenio_nao_aceito():
    err = GTInovaError(error_code="CONVENIO_NAO_ACEITO", message="Convênio não aceito pelo médico.")
    gt = make_gt(err)
    parsed = make_parsed(
        IntentType.AGENDAR,
        medico_nome="Dr. Marcelo",
        atendimento_nome="Consulta",
        data_preferida="2026-05-10",
        convenio="SulAmérica",
        resposta_fila="SIM",
    )
    msgs, next_state = await execute_workflow(
        parsed, ConversationState.CONFIRMANDO, [], "cli-1", "sess-1", make_db(), gt
    )
    assert next_state == ConversationState.TRIAGEM.value
    assert "convênio" in msgs[0].text.lower()


# ── CANCELAR ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cancelar_sem_agendamento_id_pede_id():
    parsed = make_parsed(IntentType.CANCELAR)
    faltantes = ["agendamento_id"]
    msgs, next_state = await execute_workflow(
        parsed, ConversationState.TRIAGEM, faltantes, "cli-1", "sess-1", make_db(), None
    )
    assert next_state == ConversationState.COLETANDO_DADOS.value


@pytest.mark.asyncio
async def test_cancelar_confirmando_sucesso():
    ok = GTInovaOk(data={"message": "Consulta cancelada."})
    gt = make_gt(ok)
    parsed = make_parsed(IntentType.CANCELAR, agendamento_id="ag-123")
    msgs, next_state = await execute_workflow(
        parsed, ConversationState.CONFIRMANDO, [], "cli-1", "sess-1", make_db(), gt
    )
    assert next_state == ConversationState.CONCLUIDO.value


# ── REMARCAR ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_remarcar_confirmando_sucesso():
    ok = GTInovaOk(data={"message": "Remarcado!"})
    gt = make_gt(ok)
    parsed = make_parsed(IntentType.REMARCAR, agendamento_id="ag-123", data_preferida="2026-06-01")
    msgs, next_state = await execute_workflow(
        parsed, ConversationState.CONFIRMANDO, [], "cli-1", "sess-1", make_db(), gt
    )
    assert next_state == ConversationState.CONCLUIDO.value


# ── RESPOSTA_FILA ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_resposta_fila_nao_volta_triagem():
    parsed = make_parsed(IntentType.RESPOSTA_FILA, resposta_fila="NAO")
    msgs, next_state = await execute_workflow(
        parsed, ConversationState.AGUARDANDO_FILA, [], "cli-1", "sess-1", make_db(), None
    )
    assert next_state == ConversationState.TRIAGEM.value


@pytest.mark.asyncio
async def test_resposta_fila_sim_sem_fila_id():
    db = make_db()
    db.fetchval = AsyncMock(return_value=None)   # fila_id não encontrado
    parsed = make_parsed(IntentType.RESPOSTA_FILA, resposta_fila="SIM")
    msgs, next_state = await execute_workflow(
        parsed, ConversationState.AGUARDANDO_FILA, [], "cli-1", "sess-1", db, None
    )
    assert next_state == ConversationState.TRIAGEM.value
    assert "recepção" in msgs[0].text.lower() or "fila" in msgs[0].text.lower()


@pytest.mark.asyncio
async def test_resposta_fila_sim_sucesso():
    db = make_db()
    db.fetchval = AsyncMock(return_value="fila-uuid-123")
    ok = GTInovaOk(data={"agendamento_id": "ag-999", "message": "Vaga confirmada!"})
    gt = make_gt(ok)
    parsed = make_parsed(IntentType.RESPOSTA_FILA, resposta_fila="SIM")
    msgs, next_state = await execute_workflow(
        parsed, ConversationState.AGUARDANDO_FILA, [], "cli-1", "sess-1", db, gt
    )
    assert next_state == ConversationState.CONCLUIDO.value
    assert "vaga" in msgs[0].text.lower()


# ── Sem markdown nas mensagens de confirmação ─────────────────────────────────

@pytest.mark.asyncio
async def test_confirmation_sem_asteriscos():
    """Nenhuma mensagem de confirmação pode ter * (markdown)."""
    for intent, extra in [
        (IntentType.AGENDAR,  {"medico_nome": "Dr. A", "atendimento_nome": "C",
                               "data_preferida": "2026-05-10", "convenio": "U"}),
        (IntentType.REMARCAR, {"agendamento_id": "ag-1", "data_preferida": "2026-05-10"}),
        (IntentType.CANCELAR, {"agendamento_id": "ag-1"}),
    ]:
        parsed = make_parsed(intent, **extra)
        msgs, _ = await execute_workflow(
            parsed, ConversationState.TRIAGEM, [], "cli-1", "sess-1", make_db(), None
        )
        for m in msgs:
            assert "*" not in m.text, f"Markdown encontrado em {intent}: {m.text!r}"
