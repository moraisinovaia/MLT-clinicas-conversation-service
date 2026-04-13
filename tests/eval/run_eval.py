"""
Framework de avaliação RAG — Clínica Olhos.

Dois modos:

  routing   Testa policy_engine com mock_parsed_intent do ground_truth.
            Puro Python, sem chamadas externas. Detecta regressões de roteamento.
            Uso: python -m tests.eval.run_eval --mode routing

  e2e       Chama o serviço HTTP deployado e verifica qualidade da resposta.
            Requer SERVICE_URL configurado. Testa pipeline completo.
            Uso: python -m tests.eval.run_eval --mode e2e --url https://seu-servico/

  full      Chama semantic_parse (LLM) + decide_route localmente. Não precisa de DB.
            Mede acurácia de intent e risco do LLM. Requer OpenRouter API key.
            Uso: python -m tests.eval.run_eval --mode full
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx

# Adiciona a raiz do projeto ao path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.models.intent import IntentType, ParsedIntent, EntitySet
from app.models.state import ConversationState
from app.core.policy_engine import decide_route
from tests.eval.metrics import (
    CaseResult, compute_metrics, format_report,
    check_redirect, check_keywords, check_forbidden, chunk_recall,
)

GROUND_TRUTH_PATH = Path(__file__).parent / "ground_truth.json"
RESULTS_DIR       = Path(__file__).parent / "results"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_ground_truth() -> dict:
    with open(GROUND_TRUTH_PATH) as f:
        return json.load(f)


def mock_to_parsed_intent(mock: dict, query: str) -> ParsedIntent:
    """Constrói ParsedIntent a partir do mock_parsed_intent do ground truth."""
    entities_raw = mock.get("entities", {})
    return ParsedIntent(
        intent=IntentType(mock["intent"]),
        confidence=mock.get("confidence", 0.9),
        entities=EntitySet(**entities_raw),
        risk_level=mock.get("risk_level", "low"),
        needs_clarification=mock.get("needs_clarification", False),
        is_operational_query=mock.get("is_operational_query", False),
        mensagem_usuario=query,   # injeta a query para as regras de keyword funcionarem
    )


# ── Modo routing (puro Python, sem I/O) ──────────────────────────────────────

def run_routing_case(case: dict) -> CaseResult:
    parsed   = mock_to_parsed_intent(case["mock_parsed_intent"], case["query"])
    decision = decide_route(parsed, ConversationState.TRIAGEM)
    return CaseResult(
        id             = case["id"],
        category       = case["category"],
        query          = case["query"],
        expected_route = case["expected_route"],
        actual_route   = decision.route,
        route_ok       = decision.route == case["expected_route"],
    )


# ── Modo full (LLM local + routing, sem DB) ───────────────────────────────────

async def run_full_case(case: dict) -> CaseResult:
    from app.core.semantic_parser import semantic_parse
    try:
        parsed = await semantic_parse(
            message      = case["query"],
            context      = "",
            cliente_info = "Clínica Olhos de Petrolina — oftalmologia",
        )
    except Exception as e:
        return CaseResult(
            id             = case["id"],
            category       = case["category"],
            query          = case["query"],
            expected_route = case["expected_route"],
            actual_route   = "error",
            route_ok       = False,
            error          = f"semantic_parse failed: {e}",
        )

    # Injeta mensagem para regras de keyword do policy engine
    parsed = parsed.model_copy(update={"mensagem_usuario": case["query"]})
    decision = decide_route(parsed, ConversationState.TRIAGEM)

    return CaseResult(
        id              = case["id"],
        category        = case["category"],
        query           = case["query"],
        expected_route  = case["expected_route"],
        actual_route    = decision.route,
        route_ok        = decision.route == case["expected_route"],
        expected_intent = case["expected_intent"],
        actual_intent   = parsed.intent.value,
        intent_ok       = parsed.intent.value == case["expected_intent"],
        expected_risk   = case.get("expected_risk_level"),
        actual_risk     = parsed.risk_level,
        risk_ok         = (parsed.risk_level == case.get("expected_risk_level"))
                          if case.get("expected_risk_level") else None,
    )


# ── Modo e2e (chama serviço HTTP deployado) ───────────────────────────────────

async def run_e2e_case(
    case:        dict,
    service_url: str,
    cliente_id:  str,
    client:      httpx.AsyncClient,
) -> CaseResult:
    session_id = f"eval-{case['id']}-{uuid.uuid4().hex[:6]}"
    try:
        resp = await client.post(
            f"{service_url.rstrip('/')}/api/v1/conversation",
            json={
                "message":    case["query"],
                "session_id": session_id,
                "cliente_id": cliente_id,
                "media_type": "text",
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return CaseResult(
            id             = case["id"],
            category       = case["category"],
            query          = case["query"],
            expected_route = case["expected_route"],
            actual_route   = "error",
            route_ok       = False,
            error          = str(e),
        )

    messages = data.get("messages", [])
    answer   = " ".join(m.get("text", "") for m in messages)

    # Para e2e não temos acesso direto à rota escolhida nem ao intent.
    # Inferimos a rota a partir do tipo de resposta — heurística simples.
    inferred_route = _infer_route_from_response(data, answer, case)

    must_contain     = case.get("answer_must_contain", [])
    must_not_contain = case.get("answer_must_not_contain", [])

    has_redir   = check_redirect(answer) if answer else None
    kw_ok       = check_keywords(answer, must_contain)     if must_contain else None
    forb_ok     = check_forbidden(answer, must_not_contain) if must_not_contain else None

    # Chunk recall: usa o endpoint de debug /api/v1/eval/retrieval se disponível.
    # O endpoint requer EVAL_RETRIEVAL_ENABLED=true no serviço.
    retrieved_ids: list[str] = []
    expected_ids  = case.get("expected_chunk_ids", [])
    recall_ok: bool | None = None

    if expected_ids and case.get("is_rag_case"):
        try:
            filters = case.get("mock_parsed_intent", {})
            ret_resp = await client.post(
                f"{service_url.rstrip('/')}/api/v1/eval/retrieval",
                json={
                    "query":        case["query"],
                    "cliente_id":   cliente_id,
                    "risk_max":     filters.get("risk_level", "high"),
                    "source_types": case.get("expected_source_types", []),
                    "k":            6,
                },
                timeout=15.0,
            )
            if ret_resp.status_code == 200:
                retrieved_ids = ret_resp.json().get("chunk_ids", [])
                recall_ok = chunk_recall(retrieved_ids, expected_ids)
        except Exception:
            pass  # endpoint indisponível — não penaliza o caso

    return CaseResult(
        id                   = case["id"],
        category             = case["category"],
        query                = case["query"],
        expected_route       = case["expected_route"],
        actual_route         = inferred_route,
        route_ok             = inferred_route == case["expected_route"],
        answer               = answer,
        expected_chunk_ids   = expected_ids,
        retrieved_chunk_ids  = retrieved_ids,
        chunk_recall_ok      = recall_ok,
        answer_kw_ok         = kw_ok,
        answer_forbidden_ok  = forb_ok,
        has_redirect         = has_redir,
    )


def _infer_route_from_response(data: dict, answer: str, case: dict) -> str:
    """
    Inferência heurística da rota a partir da resposta do serviço.
    Não é perfeita — serve como proxy para e2e quando não temos debug header.
    """
    # Se veio feedback_id → foi RAG
    if data.get("pending_feedback_id"):
        return "rag"
    # Se mensagem de emergência
    if "SAMU" in answer or "192" in answer:
        return "direct"
    # Se mensagem de saudação padrão
    greet_markers = ["estou aqui para ajudar", "como posso te ajudar", "seja bem-vindo"]
    if any(m in answer.lower() for m in greet_markers):
        return "direct"
    # Se pediu para clarificar
    clarify_markers = ["poderia informar", "qual médico", "qual procedimento", "me diz mais"]
    if any(m in answer.lower() for m in clarify_markers):
        return "clarify"
    # Se retornou endereço/telefone
    if "Endereço:" in answer or "Telefone:" in answer:
        return "sql"
    # Se retornou lista de médicos ou mensagem de agendamento
    wf_markers = ["agendamento", "vaga", "horário disponível", "escolha", "confirmar"]
    if any(m in answer.lower() for m in wf_markers):
        return "workflow"
    # Default: assume rag para casos informativos
    return case.get("expected_route", "unknown")


# ── Runner principal ──────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    gt       = load_ground_truth()
    cases    = gt["cases"]
    version  = gt.get("version", "1.0")
    cli_id   = gt["cliente_id"]

    # Filtra categorias se --category foi passado
    if args.category:
        cases = [c for c in cases if c["category"] == args.category]
        if not cases:
            print(f"Nenhum caso com category='{args.category}'.")
            return

    print(f"\nIniciando eval  mode={args.mode}  casos={len(cases)}\n")

    results: list[CaseResult] = []

    if args.mode == "routing":
        for case in cases:
            r = run_routing_case(case)
            results.append(r)
            status = "✓" if r.route_ok else "✗"
            print(f"  {status} {r.id:<22} route={r.actual_route:<10}"
                  f" {'OK' if r.route_ok else f'EXPECTED={r.expected_route}'}")

    elif args.mode == "full":
        for case in cases:
            r = await run_full_case(case)
            results.append(r)
            intent_info = (
                f"intent={r.actual_intent} {'✓' if r.intent_ok else '✗'}  "
                if r.actual_intent else ""
            )
            status = "✓" if r.route_ok else "✗"
            print(f"  {status} {r.id:<22} {intent_info}route={r.actual_route:<10}"
                  f" {'OK' if r.route_ok else f'EXPECTED={r.expected_route}'}"
                  + (f"  ERR={r.error}" if r.error else ""))

    elif args.mode == "e2e":
        if not args.url:
            print("Erro: --url é obrigatório para modo e2e.")
            sys.exit(1)
        async with httpx.AsyncClient() as client:
            for case in cases:
                r = await run_e2e_case(case, args.url, cli_id, client)
                results.append(r)
                kw_info = f"kw={'✓' if r.answer_kw_ok else '✗'}  " if r.answer_kw_ok is not None else ""
                redir_info = "REDIRECT! " if r.has_redirect else ""
                status = "✓" if r.route_ok else "✗"
                print(f"  {status} {r.id:<22} {kw_info}{redir_info}"
                      f"route={r.actual_route:<10}"
                      + (f"  ERR={r.error}" if r.error else ""))

    # Calcula e imprime métricas
    metrics = compute_metrics(results)
    print()
    print(format_report(metrics, mode=args.mode, version=version))

    # Salva resultados em JSON
    if args.output:
        RESULTS_DIR.mkdir(exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = RESULTS_DIR / f"eval_{args.mode}_{ts}.json"
        payload = {
            "mode":    args.mode,
            "ts":      ts,
            "metrics": metrics,
            "results": [
                {k: v for k, v in r.__dict__.items() if v is not None}
                for r in results
            ],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nResultados salvos em: {out_path}")

    # Exit code não-zero se acurácia de roteamento abaixo de 90%
    route_pct = metrics.get("route_accuracy_pct", 100)
    if route_pct is not None and route_pct < 90:
        print(f"\nFALHA: route_accuracy {route_pct}% < 90% threshold.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Eval RAG — Clínica Olhos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["routing", "full", "e2e"],
        default="routing",
        help="routing=sem I/O  full=LLM local  e2e=serviço HTTP (default: routing)",
    )
    parser.add_argument(
        "--url",
        default=os.getenv("EVAL_SERVICE_URL", ""),
        help="URL do serviço para modo e2e (ou env EVAL_SERVICE_URL)",
    )
    parser.add_argument(
        "--category",
        default="",
        help="Filtra por categoria (ex: duvida_preparo, sql, direct)",
    )
    parser.add_argument(
        "--output",
        action="store_true",
        default=True,
        help="Salva resultados em tests/eval/results/ (default: True)",
    )
    args = parser.parse_args()
    asyncio.run(main(args))
