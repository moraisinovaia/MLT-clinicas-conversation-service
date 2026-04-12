"""
Métricas para o framework de avaliação RAG.

Nenhum I/O aqui — só cálculos sobre listas de CaseResult.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ── Resultado de um único caso ────────────────────────────────────────────────

@dataclass
class CaseResult:
    id:               str
    category:         str
    query:            str

    # Roteamento
    expected_route:   str
    actual_route:     str
    route_ok:         bool

    # Intent + risco (só no modo full — None no modo routing)
    expected_intent:  Optional[str]  = None
    actual_intent:    Optional[str]  = None
    intent_ok:        Optional[bool] = None
    expected_risk:    Optional[str]  = None
    actual_risk:      Optional[str]  = None
    risk_ok:          Optional[bool] = None

    # Qualidade da resposta RAG (só para casos is_rag_case em modo e2e)
    answer:               Optional[str]  = None
    expected_chunk_ids:   list[str]      = field(default_factory=list)
    retrieved_chunk_ids:  list[str]      = field(default_factory=list)
    chunk_recall_ok:      Optional[bool] = None   # ≥1 expected chunk no top-k
    answer_kw_ok:         Optional[bool] = None   # answer_must_contain presentes
    answer_forbidden_ok:  Optional[bool] = None   # answer_must_not_contain ausentes
    has_redirect:         Optional[bool] = None   # contém frase de redirect desnecessário

    error: Optional[str] = None


# ── Frases de redirect que nunca devem aparecer quando a info está no chunk ───

REDIRECT_PHRASES = [
    "entre em contato com a recepção",
    "ligar para a recepção",
    "ligue para a recepção",
    "contate a recepção",
    "fale com a recepção",
    "vá à recepção",
    "presencialmente",
]


def check_redirect(answer: str) -> bool:
    """True se a resposta contém frase de redirect desnecessário."""
    low = answer.lower()
    return any(phrase in low for phrase in REDIRECT_PHRASES)


def check_keywords(answer: str, must_contain: list[str]) -> bool:
    """True se TODOS os termos de must_contain aparecem na resposta."""
    low = answer.lower()
    return all(kw.lower() in low for kw in must_contain)


def check_forbidden(answer: str, must_not_contain: list[str]) -> bool:
    """True se NENHUM termo proibido aparece na resposta (resultado limpo)."""
    low = answer.lower()
    return not any(kw.lower() in low for kw in must_not_contain)


def chunk_recall(retrieved: list[str], expected: list[str]) -> bool:
    """True se pelo menos 1 chunk esperado está nos retrieved."""
    if not expected:
        return True   # sem expectativa definida → passa
    return bool(set(expected) & set(retrieved))


# ── Agregação de métricas ────────────────────────────────────────────────────

def compute_metrics(results: list[CaseResult]) -> dict:
    total = len(results)
    if not total:
        return {}

    # Roteamento
    route_ok    = [r for r in results if r.route_ok]
    route_wrong = [r for r in results if not r.route_ok]

    # Intent (modo full)
    intent_tested = [r for r in results if r.intent_ok is not None]
    intent_ok     = [r for r in intent_tested if r.intent_ok]

    # Risk level (modo full)
    risk_tested = [r for r in results if r.risk_ok is not None]
    risk_ok     = [r for r in risk_tested if r.risk_ok]

    # RAG quality
    rag_with_answer  = [r for r in results if r.answer is not None]
    no_redirect      = [r for r in rag_with_answer if not r.has_redirect]
    kw_ok_cases      = [r for r in rag_with_answer if r.answer_kw_ok is True]
    forbidden_ok     = [r for r in rag_with_answer if r.answer_forbidden_ok is True]
    chunk_recall_ok  = [r for r in rag_with_answer if r.chunk_recall_ok is True]
    chunk_recall_tested = [r for r in rag_with_answer if r.chunk_recall_ok is not None]

    def pct(num, den):
        return round(num / den * 100, 1) if den else None

    return {
        "total_cases":            total,
        # Routing
        "route_accuracy_pct":     pct(len(route_ok), total),
        "route_correct":          len(route_ok),
        "route_wrong":            len(route_wrong),
        "route_wrong_cases":      [{"id": r.id, "query": r.query[:60],
                                    "expected": r.expected_route, "got": r.actual_route}
                                   for r in route_wrong],
        # Intent (full mode)
        "intent_accuracy_pct":    pct(len(intent_ok), len(intent_tested)),
        "intent_correct":         len(intent_ok),
        "intent_tested":          len(intent_tested),
        "intent_wrong_cases":     [{"id": r.id, "query": r.query[:60],
                                    "expected": r.expected_intent, "got": r.actual_intent}
                                   for r in intent_tested if not r.intent_ok],
        # Risk (full mode)
        "risk_accuracy_pct":      pct(len(risk_ok), len(risk_tested)),
        "risk_correct":           len(risk_ok),
        "risk_tested":            len(risk_tested),
        # RAG quality
        "rag_cases_with_answer":  len(rag_with_answer),
        "no_redirect_rate_pct":   pct(len(no_redirect), len(rag_with_answer)),
        "answer_kw_hit_rate_pct": pct(len(kw_ok_cases),
                                      len([r for r in rag_with_answer
                                           if r.answer_kw_ok is not None])),
        "answer_forbidden_ok_pct": pct(len(forbidden_ok),
                                       len([r for r in rag_with_answer
                                            if r.answer_forbidden_ok is not None])),
        "chunk_recall_at_k_pct":  pct(len(chunk_recall_ok), len(chunk_recall_tested)),
        "chunk_recall_tested":    len(chunk_recall_tested),
        "redirect_cases":         [{"id": r.id, "query": r.query[:60],
                                    "answer_snippet": (r.answer or "")[:120]}
                                   for r in rag_with_answer if r.has_redirect],
    }


def format_report(metrics: dict, mode: str, version: str = "1.0") -> str:
    lines = [
        "═" * 60,
        f"  RAG EVAL REPORT — Clínica Olhos  |  v{version}  |  mode={mode}",
        "═" * 60,
        "",
        f"ROTEAMENTO  ({metrics['total_cases']} casos)",
        f"  Acurácia:  {metrics['route_accuracy_pct']}%"
        f"  ({metrics['route_correct']}/{metrics['total_cases']})",
    ]

    if metrics["route_wrong_cases"]:
        lines.append("  Erros:")
        for e in metrics["route_wrong_cases"]:
            lines.append(f"    {e['id']:<18} \"{e['query']}\"")
            lines.append(f"    {'':>18} expected={e['expected']}  got={e['got']}")

    if metrics["intent_tested"]:
        lines += [
            "",
            f"INTENT (LLM)  ({metrics['intent_tested']} testados)",
            f"  Acurácia:  {metrics['intent_accuracy_pct']}%"
            f"  ({metrics['intent_correct']}/{metrics['intent_tested']})",
        ]
        if metrics["intent_wrong_cases"]:
            lines.append("  Erros:")
            for e in metrics["intent_wrong_cases"]:
                lines.append(f"    {e['id']:<18} \"{e['query']}\"")
                lines.append(f"    {'':>18} expected={e['expected']}  got={e['got']}")

    if metrics["risk_tested"]:
        lines += [
            "",
            f"RISK LEVEL  ({metrics['risk_tested']} testados)",
            f"  Acurácia:  {metrics['risk_accuracy_pct']}%"
            f"  ({metrics['risk_correct']}/{metrics['risk_tested']})",
        ]

    if metrics["rag_cases_with_answer"]:
        k = metrics.get("_k", 6)
        lines += [
            "",
            f"QUALIDADE RAG  ({metrics['rag_cases_with_answer']} casos com resposta)",
            f"  Sem redirect:      {metrics['no_redirect_rate_pct']}%",
            f"  KW hit:            {metrics['answer_kw_hit_rate_pct']}%",
            f"  Sem proibidos:     {metrics['answer_forbidden_ok_pct']}%",
        ]
        if metrics["chunk_recall_tested"]:
            lines.append(
                f"  Chunk recall @{k}:  {metrics['chunk_recall_at_k_pct']}%"
                f"  ({metrics['chunk_recall_tested']} testados)"
            )
        if metrics["redirect_cases"]:
            lines.append("  Casos com redirect:")
            for r in metrics["redirect_cases"]:
                lines.append(f"    {r['id']:<18} \"{r['query']}\"")
                lines.append(f"    {'':>18} snippet: {r['answer_snippet']}")

    lines += ["", "═" * 60]
    return "\n".join(lines)
