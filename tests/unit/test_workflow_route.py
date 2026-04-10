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
        mensagem_usuario="",
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
    gt.list_doctors     = AsyncMock(return_value=result)
    gt.doctor_schedules = AsyncMock(return_value=result)
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
    assert msgs == []  # mensagem composta pelo conversation.py, não pelo workflow


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


@pytest.mark.asyncio
async def test_duvida_operacional_convenio_consulta_gt_inova():
    gt = make_gt(
        GTInovaOk(data={
            "medicos": [{
                "nome": "Dr. Hermann Madeiro",
                "convenios_aceitos": ["PARTICULAR", "HGU"],
                "servicos": [],
            }]
        })
    )
    parsed = make_parsed(
        IntentType.DUVIDA,
        medico_nome="Dr. Hermann",
        convenio="HGU",
    )
    parsed.mensagem_usuario = "Dr. Hermann atende HGU?"

    msgs, next_state = await execute_workflow(
        parsed, ConversationState.TRIAGEM, [], "cli-1", "sess-1", make_db(), gt
    )

    gt.doctor_schedules.assert_awaited_once()
    assert next_state is None
    assert "gt inova" in msgs[0].text.lower()
    assert "hgu" in msgs[0].text.lower()


@pytest.mark.asyncio
async def test_duvida_operacional_servico_ativo_consulta_gt_inova():
    gt = make_gt(
        GTInovaOk(data={
            "medicos": [{
                "nome": "Dr. Hermann Madeiro",
                "convenios_aceitos": ["PARTICULAR"],
                "servicos": [
                    {"nome": "Gonioscopia", "dias": "Quinta", "periodos": []},
                ],
            }]
        })
    )
    parsed = make_parsed(
        IntentType.DUVIDA,
        medico_nome="Dr. Hermann",
        atendimento_nome="Gonioscopia",
    )
    parsed.mensagem_usuario = "Dr. Hermann faz gonioscopia?"

    msgs, next_state = await execute_workflow(
        parsed, ConversationState.TRIAGEM, [], "cli-1", "sess-1", make_db(), gt
    )

    gt.doctor_schedules.assert_awaited_once()
    assert next_state is None
    assert "realiza gonioscopia" in msgs[0].text.lower()


@pytest.mark.asyncio
async def test_duvida_operacional_lista_convenios_por_medico_consulta_gt_inova():
    gt = make_gt(
        GTInovaOk(data={
            "medicos": [{
                "nome": "Dr. Hermann Madeiro",
                "convenios_aceitos": ["PARTICULAR", "HGU", "CASSI"],
                "servicos": [],
            }]
        })
    )
    parsed = make_parsed(
        IntentType.DUVIDA,
        medico_nome="Dr. Hermann",
    )
    parsed.mensagem_usuario = "Quais convenios o Dr. Hermann atende?"

    msgs, next_state = await execute_workflow(
        parsed, ConversationState.TRIAGEM, [], "cli-1", "sess-1", make_db(), gt
    )

    gt.doctor_schedules.assert_awaited_once()
    assert next_state is None
    assert "segundo a gt inova agora" in msgs[0].text.lower()
    assert "cassi" in msgs[0].text.lower()


@pytest.mark.asyncio
async def test_duvida_operacional_lista_servicos_por_medico_consulta_gt_inova():
    gt = make_gt(
        GTInovaOk(data={
            "medicos": [{
                "nome": "Dr. Hermann Madeiro",
                "convenios_aceitos": ["PARTICULAR"],
                "servicos": [
                    {"nome": "Gonioscopia", "dias": "Quinta", "periodos": []},
                    {"nome": "Mapeamento de Retina", "dias": "Sexta", "periodos": []},
                ],
            }]
        })
    )
    parsed = make_parsed(
        IntentType.DUVIDA,
        medico_nome="Dr. Hermann",
    )
    parsed.mensagem_usuario = "Quais procedimentos o Dr. Hermann faz?"

    msgs, next_state = await execute_workflow(
        parsed, ConversationState.TRIAGEM, [], "cli-1", "sess-1", make_db(), gt
    )

    gt.doctor_schedules.assert_awaited_once()
    assert next_state is None
    assert "servicos ativos" in msgs[0].text.lower()
    assert "gonioscopia" in msgs[0].text.lower()


@pytest.mark.asyncio
async def test_duvida_operacional_disponibilidade_consulta_gt_inova():
    gt = make_gt(
        GTInovaOk(data={"message": "Dr. Hermann tem vagas na quinta-feira."})
    )
    parsed = make_parsed(
        IntentType.DUVIDA,
        medico_nome="Dr. Hermann",
        atendimento_nome="Consulta oftalmologica",
    )
    parsed.mensagem_usuario = "Tem vaga do Dr. Hermann para consulta?"

    msgs, next_state = await execute_workflow(
        parsed, ConversationState.TRIAGEM, [], "cli-1", "sess-1", make_db(), gt
    )

    gt.get_availability.assert_awaited_once()
    assert next_state is None
    assert "vagas" in msgs[0].text.lower()


@pytest.mark.asyncio
async def test_duvida_operacional_vaga_sem_atendimento_pede_clarificacao():
    gt = make_gt(GTInovaOk(data={"message": "nao deveria chamar"}))
    parsed = make_parsed(
        IntentType.DUVIDA,
        medico_nome="Dr. Hermann",
    )
    parsed.mensagem_usuario = "Dr. Hermann tem vaga?"

    msgs, next_state = await execute_workflow(
        parsed, ConversationState.TRIAGEM, [], "cli-1", "sess-1", make_db(), gt
    )

    gt.get_availability.assert_not_awaited()
    gt.doctor_schedules.assert_not_awaited()
    assert next_state == ConversationState.COLETANDO_DADOS.value
    assert "qual atendimento" in msgs[0].text.lower()


@pytest.mark.asyncio
async def test_duvida_operacional_limite_convenio_responde_sem_inventar_numero():
    gt = make_gt(
        GTInovaOk(data={
            "medicos": [{
                "nome": "Dr. Hermann Madeiro",
                "convenios_aceitos": ["PARTICULAR", "HGU"],
                "servicos": [],
            }]
        })
    )
    parsed = make_parsed(
        IntentType.DUVIDA,
        medico_nome="Dr. Hermann",
        convenio="HGU",
    )
    parsed.mensagem_usuario = "Qual o limite de pacientes HGU por turno do Dr. Hermann?"

    msgs, next_state = await execute_workflow(
        parsed, ConversationState.TRIAGEM, [], "cli-1", "sess-1", make_db(), gt
    )

    gt.doctor_schedules.assert_awaited_once()
    assert next_state is None
    assert "gt inova" in msgs[0].text.lower()
    assert "validado" in msgs[0].text.lower()
    assert "18" not in msgs[0].text


@pytest.mark.asyncio
async def test_duvida_operacional_sem_gt_inova_fecha_log_como_failed():
    db = make_db()
    parsed = make_parsed(
        IntentType.DUVIDA,
        medico_nome="Dr. Hermann",
        convenio="HGU",
    )
    parsed.mensagem_usuario = "Dr. Hermann atende HGU?"

    msgs, next_state = await execute_workflow(
        parsed, ConversationState.TRIAGEM, [], "cli-1", "sess-1", db, None
    )

    assert next_state is None
    assert "gt inova" in msgs[0].text.lower()
    assert db.execute.await_count == 2


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
