"""
Cliente GT Inova API.

- Retry exponencial: 3 tentativas, backoff 1→4s
- Circuit breaker: 3 falhas em 60s → abre o circuito
- error.message da API usado diretamente, sem reformatar
"""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass
from typing import Any
import httpx
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type, before_sleep_log,
)

logger = logging.getLogger(__name__)


# ── Tipos de resposta ─────────────────────────────────────────────────────────

@dataclass
class GTInovaError:
    error_code: str
    message:    str     # já formatado para WhatsApp — usar diretamente


@dataclass
class GTInovaOk:
    data: dict[str, Any]


GTInovaResult = GTInovaOk | GTInovaError


# ── Circuit breaker (simples, em memória) ─────────────────────────────────────
# Estado compartilhado no processo — suficiente para 1 worker uvicorn.
# Em multi-worker: usar Redis. Por ora: correto para a VPS atual (1 worker).

_cb_failures:       int   = 0
_cb_opened_at:      float = 0.0
_CB_THRESHOLD:      int   = 3    # falhas para abrir
_CB_TIMEOUT_SEC:    int   = 60   # segundos antes de tentar novamente


def _cb_is_open() -> bool:
    global _cb_failures, _cb_opened_at
    if _cb_failures < _CB_THRESHOLD:
        return False
    if time.monotonic() - _cb_opened_at > _CB_TIMEOUT_SEC:
        # Half-open: deixa 1 tentativa passar
        _cb_failures = _CB_THRESHOLD - 1
        return False
    return True


def _cb_record_failure() -> None:
    global _cb_failures, _cb_opened_at
    _cb_failures += 1
    if _cb_failures >= _CB_THRESHOLD:
        _cb_opened_at = time.monotonic()
        logger.error("circuit_breaker_open failures=%d", _cb_failures)


def _cb_record_success() -> None:
    global _cb_failures
    _cb_failures = 0


# ── Cliente ───────────────────────────────────────────────────────────────────

class GTInovaClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.headers  = {
            "x-api-key":    api_key,
            "Content-Type": "application/json",
        }

    # ── Helper de requisição ──────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _request(self, method: str, path: str, payload: dict) -> GTInovaResult:
        if _cb_is_open():
            return GTInovaError(
                error_code="CIRCUIT_OPEN",
                message="Sistema de agendamento temporariamente indisponível. Tente novamente em alguns minutos.",
            )
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.request(
                    method,
                    f"{self.base_url}{path}",
                    headers=self.headers,
                    json=payload,
                )
                _cb_record_success()

                if resp.status_code >= 400:
                    body = resp.json()
                    return GTInovaError(
                        error_code=body.get("error_code", "API_ERROR"),
                        message=body.get("message", "Erro no sistema de agendamento."),
                    )
                return GTInovaOk(data=resp.json())

        except (httpx.TimeoutException, httpx.NetworkError) as e:
            _cb_record_failure()
            raise   # tenacity faz o retry

        except Exception as e:
            _cb_record_failure()
            logger.error("gt_inova_unexpected_error path=%s err=%s", path, e)
            return GTInovaError(
                error_code="UNEXPECTED_ERROR",
                message="Não foi possível completar a solicitação. Tente novamente.",
            )

    # ── Endpoints ─────────────────────────────────────────────────────────────

    async def patient_search(self, phone: str, cliente_id: str) -> GTInovaResult:
        """Busca paciente pelo telefone — chamado na 1ª mensagem da sessão."""
        return await self._request("POST", "/patient-search", {
            "phone":      phone,
            "cliente_id": cliente_id,
        })

    async def get_availability(
        self,
        medico_nome:      str,
        atendimento_nome: str,
        cliente_id:       str,
        periodo:          str | None = None,
    ) -> GTInovaResult:
        return await self._request("POST", "/availability", {
            "medico_nome":      medico_nome,
            "atendimento_nome": atendimento_nome,
            "cliente_id":       cliente_id,
            "periodo":          periodo,
        })

    async def list_doctors(self, cliente_id: str) -> GTInovaResult:
        return await self._request("POST", "/list-doctors", {
            "cliente_id": cliente_id,
        })

    async def doctor_schedules(
        self,
        cliente_id: str,
        medico_nome: str | None = None,
    ) -> GTInovaResult:
        payload: dict[str, str] = {"cliente_id": cliente_id}
        if medico_nome:
            payload["medico_nome"] = medico_nome
        return await self._request("POST", "/doctor-schedules", payload)

    async def schedule(
        self,
        medico_nome:      str,
        atendimento_nome: str,
        data_preferida:   str,
        convenio:         str,
        cliente_id:       str,
        paciente_nome:    str | None = None,
        paciente_celular: str | None = None,
        data_nascimento:  str | None = None,
        periodo:          str | None = None,
    ) -> GTInovaResult:
        return await self._request("POST", "/schedule", {
            "medico_nome":      medico_nome,
            "atendimento_nome": atendimento_nome,
            "data_preferida":   data_preferida,
            "convenio":         convenio,
            "cliente_id":       cliente_id,
            "paciente_nome":    paciente_nome,
            "paciente_celular": paciente_celular,
            "data_nascimento":  data_nascimento,
            "periodo":          periodo,
        })

    async def reschedule(self, agendamento_id: str, data_preferida: str, cliente_id: str) -> GTInovaResult:
        return await self._request("POST", "/reschedule", {
            "agendamento_id": agendamento_id,
            "data_preferida": data_preferida,
            "cliente_id":     cliente_id,
        })

    async def cancel(self, agendamento_id: str, cliente_id: str) -> GTInovaResult:
        return await self._request("POST", "/cancel", {
            "agendamento_id": agendamento_id,
            "cliente_id":     cliente_id,
        })

    async def confirm(self, agendamento_id: str, cliente_id: str) -> GTInovaResult:
        return await self._request("POST", "/confirm", {
            "agendamento_id": agendamento_id,
            "cliente_id":     cliente_id,
        })

    async def adicionar_fila(
        self,
        medico_nome:      str,
        atendimento_nome: str,
        cliente_id:       str,
        convenio:         str | None = None,
    ) -> GTInovaResult:
        return await self._request("POST", "/adicionar-fila", {
            "medico_nome":      medico_nome,
            "atendimento_nome": atendimento_nome,
            "cliente_id":       cliente_id,
            "convenio":         convenio,
        })

    async def responder_fila(
        self,
        fila_id:    str,
        resposta:   str,           # "SIM" | "NAO"
        cliente_id: str,
    ) -> GTInovaResult:
        return await self._request("POST", "/responder-fila", {
            "fila_id":    fila_id,
            "resposta":   resposta,
            "cliente_id": cliente_id,
        })

    async def list_appointments(self, paciente_celular: str, cliente_id: str) -> GTInovaResult:
        return await self._request("POST", "/list-appointments", {
            "paciente_celular": paciente_celular,
            "cliente_id":       cliente_id,
        })

    async def check_patient(self, paciente_celular: str, cliente_id: str) -> GTInovaResult:
        return await self._request("POST", "/check-patient", {
            "paciente_celular": paciente_celular,
            "cliente_id":       cliente_id,
        })
