"""
Cliente OpenRouter — Gemini 2.5 Flash como primário, Gemini Flash 1.5 como fallback.
Timeout de 20s. Retry 2x com backoff exponencial.
"""
from __future__ import annotations
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
)
async def call_llm(system: str, user: str) -> str:
    """
    Chama o LLM via OpenRouter e retorna o texto da resposta.
    Tenta o modelo primário; se falhar, usa o fallback.
    """
    from app.core.config import settings  # lazy — evita falha de import em testes sem .env
    for model in [settings.openrouter_model_primary, settings.openrouter_model_fallback]:
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {settings.openrouter_api_key}",
                        "Content-Type":  "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system",  "content": system},
                            {"role": "user",    "content": user},
                        ],
                        "temperature": 0.1,   # determinístico para parse
                        "max_tokens":  512,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            if model == settings.openrouter_model_fallback:
                raise
            continue  # tenta o fallback

    raise RuntimeError("Ambos os modelos LLM falharam")
