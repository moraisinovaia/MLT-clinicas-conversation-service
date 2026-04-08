"""
Pipeline de embeddings — Fase 2, Tarefa 2.2.

Busca todos os knowledge_chunks sem embedding, gera via OpenAI
text-embedding-3-small (1536-dim) e atualiza o banco.

Uso:
  DATABASE_URL=... OPENAI_API_KEY=... python scripts/generate_embeddings.py

Custo estimado: 77 docs ≈ $0.01 com text-embedding-3-small.
Idempotente: só processa chunks com embedding IS NULL.
"""
from __future__ import annotations
import asyncio
import os
import sys
import time
import asyncpg
import httpx

DATABASE_URL  = os.environ["DATABASE_URL"].replace("postgresql+asyncpg://", "postgresql://")
OPENAI_KEY    = os.environ["OPENAI_API_KEY"]
BATCH_SIZE    = 20    # chunks por requisição à OpenAI
MODEL         = "text-embedding-3-small"
DIMENSIONS    = 1536


async def embed_batch(texts: list[str], client: httpx.AsyncClient) -> list[list[float]]:
    resp = await client.post(
        "https://api.openai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {OPENAI_KEY}"},
        json={"model": MODEL, "input": texts, "dimensions": DIMENSIONS},
        timeout=60.0,
    )
    resp.raise_for_status()
    data = resp.json()
    # Garante que a ordem dos embeddings corresponde à ordem dos inputs
    sorted_items = sorted(data["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in sorted_items]


async def run():
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=3)

    # Total sem embedding
    total = await pool.fetchval(
        "SELECT COUNT(*) FROM knowledge_chunks WHERE embedding IS NULL AND chunk_text IS NOT NULL"
    )
    print(f"Chunks sem embedding: {total}")
    if total == 0:
        print("Nada a fazer.")
        return

    processed = 0
    errors = 0

    async with httpx.AsyncClient() as http:
        while True:
            rows = await pool.fetch(
                """
                SELECT id, chunk_text
                FROM knowledge_chunks
                WHERE embedding IS NULL AND chunk_text IS NOT NULL
                LIMIT $1
                """,
                BATCH_SIZE,
            )
            if not rows:
                break

            ids   = [r["id"]         for r in rows]
            texts = [r["chunk_text"] for r in rows]

            try:
                embeddings = await embed_batch(texts, http)
            except Exception as e:
                print(f"  Erro no batch (ids={ids[:2]}...): {e}")
                errors += 1
                await asyncio.sleep(5)
                continue

            # Atualiza um por um dentro de uma transação
            async with pool.acquire() as conn:
                async with conn.transaction():
                    for chunk_id, emb in zip(ids, embeddings):
                        # pgvector via asyncpg exige string no formato '[f1,f2,...]'
                        emb_str = "[" + ",".join(str(v) for v in emb) + "]"
                        await conn.execute(
                            "UPDATE knowledge_chunks SET embedding = $1 WHERE id = $2",
                            emb_str,
                            chunk_id,
                        )

            processed += len(rows)
            pct = processed / total * 100
            print(f"  {processed}/{total} ({pct:.1f}%) — batch ok", flush=True)

            # Rate limit gentil: 1 req/s para não estourar quota
            time.sleep(1)

    remaining = await pool.fetchval(
        "SELECT COUNT(*) FROM knowledge_chunks WHERE embedding IS NULL"
    )
    await pool.close()

    print(f"\nConcluído. Processados: {processed} | Erros: {errors} | Restantes: {remaining}")
    if remaining > 0:
        print("⚠  Restam chunks sem embedding — rodar novamente para completar.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run())
