"""Testa validações estruturadas de combinações de agendamento."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.routes.workflow_route import (
    _check_atendimento_medico,
    _check_convenio_atendimento_medico,
    _check_convenio_medico,
    _handle_agendar,
)
from app.models.intent import EntitySet
from app.models.state import ConversationState
from app.integrations.gt_inova import GTInovaClient, GTInovaOk


def _make_gt_schedules(convenios: list[str], servicos: list[str], nome: str = "Dr. Marcelo") -> MagicMock:
    """Mock de GTInovaClient.doctor_schedules retornando um médico com convenios e serviços."""
    gt = MagicMock(spec=GTInovaClient)
    gt.doctor_schedules = AsyncMock(return_value=GTInovaOk(data={
        "medicos": [{"nome": nome, "convenios_aceitos": convenios, "servicos": servicos}]
    }))
    gt.get_availability = AsyncMock(return_value=GTInovaOk(data={
        "data": "2026-04-20", "periodos": []
    }))
    return gt


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_db(status: str | None) -> AsyncMock:
    """Cria um mock de asyncpg.Connection que retorna a linha desejada."""
    db = AsyncMock()
    if status is None:
        db.fetchrow = AsyncMock(return_value=None)
    else:
        row = MagicMock()
        row.__getitem__ = MagicMock(side_effect=lambda k: status if k == "status" else None)
        db.fetchrow = AsyncMock(return_value=row)
    return db


def _make_entities(**kwargs) -> EntitySet:
    defaults = dict(
        medico_nome="Dr. Marcelo",
        convenio="unimed",
        convenio_canonico=None,
        data_preferida="2026-04-20",
        atendimento_nome="consulta",
        paciente_nome="Maria",
        paciente_celular="5511999990000",
    )
    defaults.update(kwargs)
    return EntitySet(**defaults)


def _make_row(**values) -> MagicMock:
    row = MagicMock()
    row.__getitem__ = MagicMock(side_effect=lambda key: values.get(key))
    return row


# ── _check_convenio_medico ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_atende_returns_none():
    """Médico atende o convênio → sem bloqueio."""
    db = _make_db("atende")
    result = await _check_convenio_medico(_make_entities(), "cli-1", db)
    assert result is None


@pytest.mark.asyncio
async def test_nao_atende_returns_error_message():
    """Médico não atende o convênio → mensagem de erro."""
    db = _make_db("nao_atende")
    result = await _check_convenio_medico(_make_entities(), "cli-1", db)
    assert result is not None
    assert "unimed" in result.lower()


@pytest.mark.asyncio
async def test_sem_registro_returns_none():
    """Sem registro na tabela → None (GT Inova decide)."""
    db = _make_db(None)
    result = await _check_convenio_medico(_make_entities(), "cli-1", db)
    assert result is None


@pytest.mark.asyncio
async def test_sem_medico_skips_query():
    """Sem medico_nome → None sem consultar o DB."""
    db = AsyncMock()
    entities = _make_entities(medico_nome=None)
    result = await _check_convenio_medico(entities, "cli-1", db)
    assert result is None
    db.fetchrow.assert_not_called()


@pytest.mark.asyncio
async def test_sem_convenio_skips_query():
    """Sem convenio nem convenio_canonico → None sem consultar o DB."""
    db = AsyncMock()
    entities = _make_entities(convenio=None, convenio_canonico=None)
    result = await _check_convenio_medico(entities, "cli-1", db)
    assert result is None
    db.fetchrow.assert_not_called()


@pytest.mark.asyncio
async def test_usa_convenio_canonico_se_disponivel():
    """convenio_canonico tem prioridade sobre convenio bruto."""
    db = _make_db("nao_atende")
    entities = _make_entities(convenio="uni med", convenio_canonico="unimed")
    await _check_convenio_medico(entities, "cli-1", db)
    # O terceiro argumento passado ao fetchrow deve ser "unimed" (canônico)
    call_args = db.fetchrow.call_args
    assert call_args.args[3] == "unimed"


# ── integração com _handle_agendar ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_handle_agendar_blocks_when_nao_atende():
    """_handle_agendar bloqueia via GT Inova precheck quando convênio não é aceito."""
    db = _make_db(None)
    entities = _make_entities()  # convenio="unimed"
    # GT Inova diz que Dr. Marcelo só aceita CASSI e PARTICULAR
    gt = _make_gt_schedules(convenios=["CASSI", "PARTICULAR"], servicos=["consulta"])
    messages, next_state = await _handle_agendar(
        entities=entities,
        estado_atual=ConversationState.TRIAGEM,
        dados_faltantes=[],
        cliente_id="cli-1",
        session_id="ses-1",
        db=db,
        gt_inova=gt,
    )
    assert next_state == ConversationState.TRIAGEM.value
    assert len(messages) == 1
    assert "unimed" in messages[0].text.lower()


@pytest.mark.asyncio
async def test_handle_agendar_proceeds_when_atende():
    """_handle_agendar não bloqueia quando convênio é aceito."""
    db = _make_db("atende")
    # gt_inova=None → _handle_offer_availability vai pular para confirmação
    entities = _make_entities()
    messages, next_state = await _handle_agendar(
        entities=entities,
        estado_atual=ConversationState.TRIAGEM,
        dados_faltantes=[],
        cliente_id="cli-1",
        session_id="ses-1",
        db=db,
        gt_inova=None,
    )
    # Com gt_inova=None, cai em _handle_offer_availability → retorna confirmação
    assert next_state == ConversationState.CONFIRMANDO.value


@pytest.mark.asyncio
async def test_handle_agendar_not_validated_when_confirmando():
    """Em estado CONFIRMANDO (dados já validados), a validação não é re-executada."""
    db = AsyncMock()
    # Mesmo que o DB retornasse nao_atende, não deve ser consultado em CONFIRMANDO
    db.fetchrow = AsyncMock(return_value=None)

    entities = _make_entities(resposta_fila="SIM")
    messages, _ = await _handle_agendar(
        entities=entities,
        estado_atual=ConversationState.CONFIRMANDO,
        dados_faltantes=[],
        cliente_id="cli-1",
        session_id="ses-1",
        db=db,
        gt_inova=None,
    )
    # Deve ter chamado _execute_schedule, não _check_convenio_medico
    db.fetchrow.assert_not_called()


# ── _check_atendimento_medico ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_atendimento_medico_skips_when_missing_fields():
    db = AsyncMock()
    entities = _make_entities(atendimento_nome=None)
    assert await _check_atendimento_medico(entities, "cli-1", db) is None
    db.fetchrow.assert_not_called()


@pytest.mark.asyncio
async def test_atendimento_medico_returns_none_when_unresolved_ids():
    db = AsyncMock()
    with patch("app.routes.workflow_route.resolve_rag_ids", new=AsyncMock(return_value=(None, None))):
        assert await _check_atendimento_medico(_make_entities(), "cli-1", db) is None
    db.fetchrow.assert_not_called()


@pytest.mark.asyncio
async def test_atendimento_medico_blocks_when_inactive():
    db = AsyncMock()
    db.fetchrow = AsyncMock(return_value=_make_row(
        rule_action="deny",
        mensagem_bloqueio="Esse medico nao realiza esse atendimento.",
        id="rule-1",
        notes="bloqueio teste",
    ))
    with patch("app.routes.workflow_route.resolve_rag_ids", new=AsyncMock(return_value=("med-1", "proc-1"))):
        result = await _check_atendimento_medico(_make_entities(), "cli-1", db)
    assert result == "Esse medico nao realiza esse atendimento."


@pytest.mark.asyncio
async def test_atendimento_medico_allows_when_active():
    db = AsyncMock()
    db.fetchrow = AsyncMock(return_value=_make_row(
        rule_action="allow",
        mensagem_bloqueio=None,
        id="rule-1",
        notes=None,
    ))
    with patch("app.routes.workflow_route.resolve_rag_ids", new=AsyncMock(return_value=("med-1", "proc-1"))):
        result = await _check_atendimento_medico(_make_entities(), "cli-1", db)
    assert result is None


# ── _check_convenio_atendimento_medico ───────────────────────────────────────

@pytest.mark.asyncio
async def test_tripla_regra_skips_when_missing_fields():
    entities = _make_entities(convenio_canonico="HGU", atendimento_nome=None)
    db = AsyncMock()
    assert await _check_convenio_atendimento_medico(entities, "cli-1", db) is None
    db.fetchrow.assert_not_called()


@pytest.mark.asyncio
async def test_tripla_regra_skips_when_not_applicable():
    entities = _make_entities(
        medico_nome="Dr. Guilherme Lucena Moura",
        convenio_canonico="Unimed",
        atendimento_nome="consulta eletiva",
    )
    db = AsyncMock()
    with patch("app.routes.workflow_route.resolve_rag_ids", new=AsyncMock(return_value=("med-1", "proc-1"))):
        db.fetchrow = AsyncMock(return_value=None)
        assert await _check_convenio_atendimento_medico(entities, "cli-1", db) is None


@pytest.mark.asyncio
async def test_tripla_regra_allows_hgu_with_catarata():
    entities = _make_entities(
        medico_nome="Dr. Guilherme Lucena Moura",
        convenio_canonico="HGU",
        atendimento_nome="cirurgia de catarata",
    )
    db = AsyncMock()
    with patch("app.routes.workflow_route.resolve_rag_ids", new=AsyncMock(return_value=(None, None))):
        assert await _check_convenio_atendimento_medico(entities, "cli-1", db) is None


@pytest.mark.asyncio
async def test_tripla_regra_allows_hgu_with_retina():
    entities = _make_entities(
        medico_nome="Dr. Guilherme Moura",
        convenio_canonico="HGU",
        atendimento_nome="tratamento em retina com laser",
    )
    db = AsyncMock()
    with patch("app.routes.workflow_route.resolve_rag_ids", new=AsyncMock(return_value=(None, None))):
        assert await _check_convenio_atendimento_medico(entities, "cli-1", db) is None


@pytest.mark.asyncio
async def test_tripla_regra_sem_ids_nao_bloqueia():
    """Sem IDs (médico/procedimento não cadastrado) → sem regra → GT Inova decide."""
    entities = _make_entities(
        medico_nome="Dr. Guilherme Lucena Moura",
        convenio_canonico="HGU",
        atendimento_nome="consulta eletiva",
    )
    db = AsyncMock()
    with patch("app.routes.workflow_route.resolve_rag_ids", new=AsyncMock(return_value=(None, None))):
        result = await _check_convenio_atendimento_medico(entities, "cli-1", db)
    assert result is None  # action="none" → GT Inova decide


@pytest.mark.asyncio
async def test_tripla_regra_uses_global_table_when_seeded():
    entities = _make_entities(
        medico_nome="Dr. Guilherme Lucena Moura",
        convenio_canonico="HGU",
        atendimento_nome="consulta eletiva",
    )
    db = AsyncMock()
    db.fetchrow = AsyncMock(return_value=_make_row(
        rule_action="deny",
        mensagem_bloqueio="Bloqueio vindo da matriz global.",
        id="rule-2",
        notes="bloqueio seed",
    ))
    with patch("app.routes.workflow_route.resolve_rag_ids", new=AsyncMock(return_value=("med-1", "proc-1"))):
        result = await _check_convenio_atendimento_medico(entities, "cli-1", db)
    assert result == "Bloqueio vindo da matriz global."


@pytest.mark.asyncio
async def test_handle_agendar_bloqueia_hgu_via_tabela_global():
    """Regra cadastrada (deny) na tabela global bloqueia antes do GT Inova."""
    from unittest.mock import AsyncMock as AM

    db = _make_db("atende")
    db.fetchrow = AM(return_value=_make_row(
        rule_action="deny",
        mensagem_bloqueio="HGU nao cobre consulta com Dr. Guilherme.",
        id="rule-block",
        notes="deny via tabela",
    ))
    entities = _make_entities(
        medico_nome="Dr. Guilherme Lucena Moura",
        convenio_canonico="HGU",
        atendimento_nome="consulta eletiva",
    )
    with patch("app.routes.workflow_route.resolve_rag_ids", new=AM(return_value=("med-1", "proc-1"))):
        messages, next_state = await _handle_agendar(
            entities=entities,
            estado_atual=ConversationState.TRIAGEM,
            dados_faltantes=[],
            cliente_id="cli-1",
            session_id="ses-1",
            db=db,
            gt_inova=None,
        )
    assert next_state == ConversationState.TRIAGEM.value
    assert len(messages) == 1
    assert "hgu" in messages[0].text.lower()


@pytest.mark.asyncio
async def test_handle_agendar_allows_hgu_when_allowed_attendance():
    db = _make_db("atende")
    entities = _make_entities(
        medico_nome="Dr. Guilherme Lucena Moura",
        convenio_canonico="HGU",
        atendimento_nome="cirurgia de retina",
    )
    with patch("app.routes.workflow_route.resolve_rag_ids", new=AsyncMock(return_value=(None, None))):
        messages, next_state = await _handle_agendar(
            entities=entities,
            estado_atual=ConversationState.TRIAGEM,
            dados_faltantes=[],
            cliente_id="cli-1",
            session_id="ses-1",
            db=db,
            gt_inova=None,
        )
    assert next_state == ConversationState.CONFIRMANDO.value
    assert "confirme" in messages[-1].text.lower()


@pytest.mark.asyncio
async def test_handle_agendar_blocks_when_atendimento_medico_inativo():
    """_handle_agendar bloqueia via GT Inova precheck quando serviço não está ativo."""
    db = _make_db(None)
    entities = _make_entities(
        medico_nome="Dr. Outro",
        convenio_canonico="Unimed",
        atendimento_nome="cirurgia",
    )
    # GT Inova diz que Dr. Outro aceita Unimed, mas só faz "consulta" (não "cirurgia")
    gt = _make_gt_schedules(
        convenios=["Unimed", "PARTICULAR"],
        servicos=["consulta"],
        nome="Dr. Outro",
    )
    messages, next_state = await _handle_agendar(
        entities=entities,
        estado_atual=ConversationState.TRIAGEM,
        dados_faltantes=[],
        cliente_id="cli-1",
        session_id="ses-1",
        db=db,
        gt_inova=gt,
    )
    assert next_state == ConversationState.TRIAGEM.value
    assert "cirurgia" in messages[0].text.lower()


@pytest.mark.asyncio
async def test_handle_agendar_specific_allow_overrides_generic_checks():
    db = AsyncMock()
    db.fetchrow = AsyncMock(return_value=_make_row(
        rule_action="allow",
        mensagem_bloqueio=None,
        id="rule-4",
        notes="allow especifico",
    ))
    entities = _make_entities(
        medico_nome="Dr. Guilherme Lucena Moura",
        convenio_canonico="HGU",
        atendimento_nome="cirurgia de retina",
    )
    with patch("app.routes.workflow_route.resolve_rag_ids", new=AsyncMock(return_value=("med-1", "proc-1"))):
        messages, next_state = await _handle_agendar(
            entities=entities,
            estado_atual=ConversationState.TRIAGEM,
            dados_faltantes=[],
            cliente_id="cli-1",
            session_id="ses-1",
            db=db,
            gt_inova=None,
        )
    assert next_state == ConversationState.CONFIRMANDO.value
    assert "confirme" in messages[-1].text.lower()
    assert db.fetchrow.call_count == 1
