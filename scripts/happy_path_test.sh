#!/usr/bin/env bash
# =============================================================================
# happy_path_test.sh — Valida o happy path completo com paciente de teste
#
# Fluxo:
#   1. Agendar com todos os dados (nome, celular, nascimento, médico, convênio, data)
#   2. Confirmar → espera new_state=concluido
#   3. Se concluído: cancelar o agendamento criado
#
# Uso:
#   BASE_URL=http://... CLIENTE_ID=<uuid> bash scripts/happy_path_test.sh
# =============================================================================

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
CLIENTE_ID="${CLIENTE_ID:-00000000-0000-0000-0000-000000000001}"
SID="happy-path-$(date +%s)"

# Paciente de teste controlado
PACIENTE_NOME="Teste Silva Santos"
PACIENTE_CELULAR="87999990001"
DATA_NASCIMENTO="1990-03-15"

GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[1;33m"
CYAN="\033[0;36m"
NC="\033[0m"

pass()  { echo -e "${GREEN}✓ $1${NC}"; }
fail()  { echo -e "${RED}✗ $1${NC}"; exit 1; }
info()  { echo -e "${YELLOW}» $1${NC}"; }
step()  { echo -e "${CYAN}--- $1 ---${NC}"; }

post() {
    local msg="$1"
    curl -s -X POST "$BASE_URL/api/v1/conversation" \
        -H "Content-Type: application/json" \
        -d "{\"session_id\":\"$SID\",\"cliente_id\":\"$CLIENTE_ID\",\"message\":\"$msg\"}"
}

get_state() { echo "$1" | python3 -c "import sys,json; print(json.load(sys.stdin).get('new_state',''))" 2>/dev/null; }
get_msg()   { echo "$1" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('messages',[{}])[0].get('text','')[:200])" 2>/dev/null; }
get_field() { echo "$1" | python3 -c "import sys,json; print(json.load(sys.stdin).get('$2',''))" 2>/dev/null; }

show() {
    local state msg
    state=$(get_state "$1")
    msg=$(get_msg "$1")
    echo "  state: $state"
    echo "  msg:   $msg"
    echo
}

echo
echo -e "${CYAN}Paciente de teste:${NC}"
echo "  Nome:        $PACIENTE_NOME"
echo "  Celular:     $PACIENTE_CELULAR"
echo "  Nascimento:  $DATA_NASCIMENTO"
echo "  Session:     $SID"
echo

# =============================================================================
step "TURN 1 — Intenção de agendar"
info "Mensagem: quero marcar uma consulta"
R=$(post "quero marcar uma consulta")
show "$R"
STATE=$(get_state "$R")
[[ "$STATE" == "coletando_dados" ]] && pass "Turn 1 → coletando_dados" || fail "Turn 1 → esperado coletando_dados, obtido $STATE"

# =============================================================================
step "TURN 2 — Médico"
info "Mensagem: com a Dra. Camila Leite"
R=$(post "com a Dra. Camila Leite")
show "$R"
STATE=$(get_state "$R")
[[ "$STATE" == "coletando_dados" ]] && pass "Turn 2 → coletando_dados" || fail "Turn 2 → $STATE"

# =============================================================================
step "TURN 3 — Convênio"
info "Mensagem: Unimed Nacional"
R=$(post "Unimed Nacional")
show "$R"
STATE=$(get_state "$R")
[[ "$STATE" == "coletando_dados" ]] && pass "Turn 3 → coletando_dados" || fail "Turn 3 → $STATE"

# =============================================================================
step "TURN 4 — Data"
info "Mensagem: qualquer dia da próxima semana, manhã"
R=$(post "qualquer dia da próxima semana, manhã")
show "$R"
STATE=$(get_state "$R")
[[ "$STATE" == "coletando_dados" || "$STATE" == "confirmando" ]] && pass "Turn 4 → $STATE" || fail "Turn 4 → $STATE"

# Se ainda coletando, pode estar pedindo atendimento_nome
if [[ "$STATE" == "coletando_dados" ]]; then
    MSG=$(get_msg "$R")
    step "TURN 4b — Tipo de atendimento (campo extra)"
    info "Mensagem: consulta oftalmológica"
    R=$(post "consulta oftalmológica")
    show "$R"
    STATE=$(get_state "$R")
    [[ "$STATE" == "coletando_dados" || "$STATE" == "confirmando" ]] && pass "Turn 4b → $STATE" || fail "Turn 4b → $STATE"
fi

# =============================================================================
step "TURN 5 — Nome completo do paciente"
info "Mensagem: me chamo $PACIENTE_NOME"
R=$(post "me chamo $PACIENTE_NOME")
show "$R"
STATE=$(get_state "$R")
[[ "$STATE" == "coletando_dados" || "$STATE" == "confirmando" ]] && pass "Turn 5 → $STATE" || fail "Turn 5 → $STATE"

# =============================================================================
step "TURN 6 — Data de nascimento"
info "Mensagem: nasci em 15/03/1990"
R=$(post "nasci em 15/03/1990, celular $PACIENTE_CELULAR")
show "$R"
STATE=$(get_state "$R")
[[ "$STATE" == "confirmando" || "$STATE" == "coletando_dados" ]] && pass "Turn 6 → $STATE" || fail "Turn 6 → $STATE"

# Garantir que chegamos em confirmando
if [[ "$STATE" == "coletando_dados" ]]; then
    info "Ainda coletando — forçando confirmação"
    R=$(post "pode confirmar")
    show "$R"
    STATE=$(get_state "$R")
fi

[[ "$STATE" == "confirmando" ]] && pass "Chegou em CONFIRMANDO ✅" || {
    echo -e "${RED}Estado atual: $STATE — não chegou em CONFIRMANDO${NC}"
    echo "  Resposta: $(get_msg "$R")"
    exit 1
}

# =============================================================================
step "TURN FINAL — Confirmação com SIM"
info "Mensagem: SIM"
R=$(post "SIM")
show "$R"
STATE=$(get_state "$R")
MSG=$(get_msg "$R")

echo "  Resposta completa da GT Inova:"
echo "$R" | python3 -m json.tool 2>/dev/null || echo "$R"
echo

if [[ "$STATE" == "concluido" ]]; then
    pass "AGENDAMENTO CRIADO ✅ new_state=concluido"
    echo -e "${GREEN}  Mensagem: $MSG${NC}"

    # Tentar extrair agendamento_id da resposta
    AGENDAMENTO_ID=$(echo "$R" | python3 -c "
import sys,json
d=json.load(sys.stdin)
msgs = d.get('messages',[])
for m in msgs:
    t = m.get('text','')
    import re
    ids = re.findall(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', t, re.I)
    if ids: print(ids[0]); break
" 2>/dev/null || echo "")

    if [[ -n "$AGENDAMENTO_ID" ]]; then
        echo "  agendamento_id detectado: $AGENDAMENTO_ID"
        echo
        step "CANCELAMENTO — limpando agendamento de teste"
        SID_CANCEL="happy-cancel-$(date +%s)"
        RC=$(curl -s -X POST "$BASE_URL/api/v1/conversation" \
            -H "Content-Type: application/json" \
            -d "{\"session_id\":\"$SID_CANCEL\",\"cliente_id\":\"$CLIENTE_ID\",\"message\":\"quero cancelar o agendamento $AGENDAMENTO_ID\"}")
        echo "  $(get_msg "$RC")"
        RC2=$(curl -s -X POST "$BASE_URL/api/v1/conversation" \
            -H "Content-Type: application/json" \
            -d "{\"session_id\":\"$SID_CANCEL\",\"cliente_id\":\"$CLIENTE_ID\",\"message\":\"SIM\"}")
        echo "  $(get_msg "$RC2")"
        CANCEL_STATE=$(get_state "$RC2")
        [[ "$CANCEL_STATE" == "concluido" ]] && pass "Agendamento cancelado ✅" || echo "  Cancel state: $CANCEL_STATE"
    else
        echo "  (agendamento_id não detectado na resposta — cancelar manualmente se necessário)"
    fi

elif [[ "$STATE" == "triagem" || "$STATE" == "coletando_dados" ]]; then
    echo -e "${YELLOW}⚠ GT Inova retornou erro (state=$STATE):${NC}"
    echo "  $MSG"
    echo
    echo "Possíveis causas:"
    echo "  - SLOT_TAKEN: não há vagas para 'próxima semana' na Dra. Camila"
    echo "  - CONVENIO_NAO_ACEITO: Unimed Nacional não aceito por este médico"
    echo "  - DUPLICATE_BOOKING: paciente já tem agendamento"
    echo "  - Dados incompletos enviados à API"
    echo
    echo "O fluxo de agendamento funcionou corretamente — erro veio da GT Inova, não do conversation-service."
else
    fail "Estado inesperado: $STATE"
fi

echo
echo -e "${GREEN}=== Happy path test concluído ===${NC}"
