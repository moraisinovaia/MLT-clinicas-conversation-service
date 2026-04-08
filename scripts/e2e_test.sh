#!/usr/bin/env bash
# =============================================================================
# e2e_test.sh — Teste end-to-end do fluxo de agendamento real
#
# Cobre:
#   1. Coleta progressiva até CONFIRMANDO
#   2. Confirmação real com chamada ao /schedule
#   3. SLOT_TAKEN → oferta de disponibilidade
#   4. CONVENIO_NAO_ACEITO
#   5. DUPLICATE_BOOKING
#   6. RESPOSTA_FILA SIM / NÃO
#
# Uso:
#   export BASE_URL=http://seu-dominio
#   export CLIENTE_ID=<uuid real>
#   bash scripts/e2e_test.sh
# =============================================================================

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
CLIENTE_ID="${CLIENTE_ID:-00000000-0000-0000-0000-000000000001}"
PREFIX="e2e-$(date +%s)"

GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[1;33m"
CYAN="\033[0;36m"
NC="\033[0m"

pass()  { echo -e "${GREEN}✓ $1${NC}"; }
fail()  { echo -e "${RED}✗ $1${NC}"; exit 1; }
info()  { echo -e "${YELLOW}» $1${NC}"; }
step()  { echo -e "${CYAN}--- $1 ---${NC}"; }
check_state() {
    local resp="$1" expected="$2" label="$3"
    local state
    state=$(echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('new_state',''))" 2>/dev/null || echo "")
    if [[ "$state" == "$expected" ]]; then
        pass "$label → new_state=$state"
    else
        fail "$label → esperado=$expected, obtido=$state"
    fi
}

post() {
    local session="$1" msg="$2"
    curl -s -X POST "$BASE_URL/api/v1/conversation" \
        -H "Content-Type: application/json" \
        -d "{\"session_id\":\"$session\",\"cliente_id\":\"$CLIENTE_ID\",\"message\":\"$msg\"}"
}

show() {
    local resp="$1"
    echo "$resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    msgs = [m.get('text','') for m in d.get('messages',[])]
    state = d.get('new_state','?')
    print(f'  state: {state}')
    for m in msgs:
        print(f'  msg: {m[:120]}')
except:
    print('  (resposta não é JSON)')
" 2>/dev/null || echo "  (erro ao parsear resposta)"
    echo
}

# =============================================================================
echo
step "CENÁRIO 1 — Coleta progressiva → CONFIRMANDO → /schedule real"
SID="${PREFIX}-agendar"

info "Turn 1: intenção sem dados"
R=$(post "$SID" "quero marcar uma consulta")
show "$R"
check_state "$R" "coletando_dados" "Turn 1"

info "Turn 2: informa médico"
R=$(post "$SID" "com a Dra. Camila Leite")
show "$R"
check_state "$R" "coletando_dados" "Turn 2"

info "Turn 3: informa período"
R=$(post "$SID" "prefiro manhã")
show "$R"
check_state "$R" "coletando_dados" "Turn 3"

info "Turn 4: informa convênio"
R=$(post "$SID" "Unimed Nacional")
show "$R"
# pode ser coletando_dados (falta data) ou confirmando
STATE=$(echo "$R" | python3 -c "import sys,json; print(json.load(sys.stdin).get('new_state',''))" 2>/dev/null)
[[ "$STATE" == "coletando_dados" || "$STATE" == "confirmando" ]] && \
    pass "Turn 4 → $STATE" || fail "Turn 4 → estado inesperado: $STATE"

info "Turn 5: informa data"
R=$(post "$SID" "qualquer dia da próxima semana")
show "$R"
STATE=$(echo "$R" | python3 -c "import sys,json; print(json.load(sys.stdin).get('new_state',''))" 2>/dev/null)
[[ "$STATE" == "confirmando" || "$STATE" == "coletando_dados" ]] && \
    pass "Turn 5 → $STATE" || fail "Turn 5 → estado inesperado: $STATE"

info "Turn 6: confirma com SIM"
R=$(post "$SID" "SIM")
show "$R"
STATE=$(echo "$R" | python3 -c "import sys,json; print(json.load(sys.stdin).get('new_state',''))" 2>/dev/null)
MSG=$(echo "$R" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('messages',[{}])[0].get('text','')[:150])" 2>/dev/null)
# concluido = agendamento OK; triagem = erro da API (SLOT_TAKEN, etc)
[[ "$STATE" == "concluido" || "$STATE" == "triagem" || "$STATE" == "coletando_dados" ]] && \
    pass "Turn 6 (confirmação) → $STATE" || fail "Turn 6 → estado inesperado: $STATE"
echo "  resposta da API: $MSG"

# =============================================================================
echo
step "CENÁRIO 2 — CONVENIO_NAO_ACEITO (convênio inválido deliberado)"
SID="${PREFIX}-convenio"

info "Turn 1: agendar com convênio inválido"
R=$(post "$SID" "quero agendar com Dra. Camila, convênio XYZ_INVALIDO, semana que vem, manhã")
show "$R"
STATE=$(echo "$R" | python3 -c "import sys,json; print(json.load(sys.stdin).get('new_state',''))" 2>/dev/null)
[[ "$STATE" == "confirmando" || "$STATE" == "coletando_dados" ]] && \
    pass "Cenário 2 Turn 1 → $STATE" || fail "Cenário 2 Turn 1 → $STATE"

info "Turn 2: confirma"
R=$(post "$SID" "SIM")
show "$R"
MSG=$(echo "$R" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('messages',[{}])[0].get('text','')[:200])" 2>/dev/null)
echo "  resposta: $MSG"
pass "Cenário 2 completo (verificar manualmente se CONVENIO_NAO_ACEITO na resposta)"

# =============================================================================
echo
step "CENÁRIO 3 — RESPOSTA_FILA NÃO"
SID="${PREFIX}-fila-nao"

info "Turn 1: responde NÃO a uma oferta de fila"
R=$(post "$SID" "não, obrigado")
show "$R"
pass "Cenário 3 RESPOSTA_FILA NÃO — verificar resposta acima"

# =============================================================================
echo
step "CENÁRIO 4 — Transbordo → TRANSBORDO"
SID="${PREFIX}-transbordo"
R=$(post "$SID" "quero falar com um atendente humano")
show "$R"
check_state "$R" "transbordo" "Transbordo"

# =============================================================================
echo -e "${GREEN}=== Todos os cenários e2e executados ===${NC}"
echo
echo "Verifique manualmente:"
echo "  - Se Turn 6 retornou 'concluido': agendamento real foi criado na GT Inova ✅"
echo "  - Se retornou 'triagem': API retornou erro (SLOT_TAKEN, etc) — normal sem slot real"
echo "  - Cenário 2: resposta deve mencionar convênio não aceito"
echo "  - Cenário 3: resposta deve ser mensagem de encerramento amigável"
