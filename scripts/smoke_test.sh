#!/usr/bin/env bash
# =============================================================================
# smoke_test.sh — Valida o conversation-service após o deploy
#
# Pré-requisitos (rodar nessa ordem antes deste script):
#   1. Preencher .env
#   2. Rodar migrations em ordem: 001 → 003 → 004 → 005 no Supabase SQL Editor
#   3. Rodar pipeline de embeddings:
#        python scripts/generate_embeddings.py
#      Verificar: SELECT COUNT(*) FROM knowledge_chunks WHERE embedding IS NULL
#      Esperado: 0. Sem isso, o teste de RAG sobe verde mas sem validar qualidade real.
#   4. Subir o serviço no Coolify
#
# Uso:
#   export BASE_URL=https://seu-dominio.coolify.io
#   export CLIENTE_ID=<uuid da clínica no Supabase>
#   bash scripts/smoke_test.sh
#
# O script para no primeiro erro e mostra qual teste falhou.
# =============================================================================

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
CLIENTE_ID="${CLIENTE_ID:-00000000-0000-0000-0000-000000000001}"
SESSION_PREFIX="smoke-$(date +%s)"

GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[1;33m"
NC="\033[0m"

pass() { echo -e "${GREEN}✓ $1${NC}"; }
fail() { echo -e "${RED}✗ $1${NC}"; exit 1; }
info() { echo -e "${YELLOW}» $1${NC}"; }

post() {
    local label="$1"
    local session="$2"
    local body="$3"
    info "$label"
    echo "  payload: $body"
    resp=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/api/v1/conversation" \
        -H "Content-Type: application/json" \
        -d "$body")
    code=$(echo "$resp" | tail -1)
    body_resp=$(echo "$resp" | head -n -1)
    echo "  status: $code"
    echo "  response: $body_resp" | head -c 400
    echo
    if [[ "$code" != "200" ]]; then
        fail "$label → HTTP $code"
    fi
    echo "$body_resp"
}

# ─────────────────────────────────────────────────────────────────────────────
info "=== 0. Health check ==="
resp=$(curl -s -w "\n%{http_code}" "$BASE_URL/health")
code=$(echo "$resp" | tail -1)
body=$(echo "$resp" | head -n -1)
echo "  $body"
if [[ "$code" != "200" ]]; then
    fail "Health → HTTP $code"
fi
if echo "$body" | grep -q '"status":"ok"'; then
    pass "Health OK (banco conectado)"
else
    fail "Health degraded — checar DATABASE_URL e conexão com Supabase"
fi

echo
# ─────────────────────────────────────────────────────────────────────────────
info "=== 1. Saudação (rota direct) ==="
r=$(post "Saudação" "${SESSION_PREFIX}-1" \
    "{\"session_id\":\"${SESSION_PREFIX}-1\",\"cliente_id\":\"$CLIENTE_ID\",\"message\":\"Boa tarde\"}")
echo "$r" | grep -q '"new_state"' && pass "Saudação retornou resposta" || fail "Saudação sem new_state"

echo
# ─────────────────────────────────────────────────────────────────────────────
info "=== 2. Intent de agendamento — sem dados (rota clarify) ==="
r=$(post "Agendar sem dados" "${SESSION_PREFIX}-2" \
    "{\"session_id\":\"${SESSION_PREFIX}-2\",\"cliente_id\":\"$CLIENTE_ID\",\"message\":\"quero marcar uma consulta\"}")
echo "$r" | grep -q '"new_state"' && pass "Agendar → pediu dados" || fail "Sem new_state"

echo
# ─────────────────────────────────────────────────────────────────────────────
info "=== 3. Dúvida clínica (rota rag) ==="
r=$(post "Dúvida RAG" "${SESSION_PREFIX}-3" \
    "{\"session_id\":\"${SESSION_PREFIX}-3\",\"cliente_id\":\"$CLIENTE_ID\",\"message\":\"qual é o preparo para ultrassom abdominal?\"}")
echo "$r" | grep -q '"messages"' && pass "Dúvida → resposta gerada" || fail "Sem messages"

echo
# ─────────────────────────────────────────────────────────────────────────────
info "=== 4. Pedido de atendente (rota workflow → TRANSBORDO) ==="
r=$(post "Transbordo" "${SESSION_PREFIX}-4" \
    "{\"session_id\":\"${SESSION_PREFIX}-4\",\"cliente_id\":\"$CLIENTE_ID\",\"message\":\"quero falar com um atendente\"}")
echo "$r" | grep -qi '"transbordo"' && pass "Transbordo → new_state=transbordo" || fail "Estado de transbordo não retornado"

echo
# ─────────────────────────────────────────────────────────────────────────────
info "=== 5. Agendar com médico + data (coleta parcial) ==="
r=$(post "Agendar parcial" "${SESSION_PREFIX}-5" \
    "{\"session_id\":\"${SESSION_PREFIX}-5\",\"cliente_id\":\"$CLIENTE_ID\",\"message\":\"quero marcar com Dr. Marcelo na terça\"}")
echo "$r" | grep -q '"new_state"' && pass "Agendar parcial → solicitou mais dados" || fail "Sem new_state"

echo
# ─────────────────────────────────────────────────────────────────────────────
echo -e "${GREEN}=== Todos os smoke tests passaram ===${NC}"
echo
echo "Próximo passo: testar um fluxo completo de agendamento end-to-end"
echo "com GT_INOVA_BASE_URL e GT_INOVA_API_KEY preenchidos no .env."
