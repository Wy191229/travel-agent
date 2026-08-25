#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${AGENT_API_BASE:-http://127.0.0.1:8001}"

if [ -f ".env" ]; then
  set -a
  . ./.env
  set +a
fi

API_KEY="${AGENT_API_KEY:-${API_KEY:-}}"

if [ -z "$API_KEY" ]; then
  echo "ERROR: API key not found. Set AGENT_API_KEY or API_KEY in .env"
  exit 1
fi

SESSION_ID="docker-smoke-$(date +%s)"

echo "== Smoke Test =="
echo "BASE_URL=$BASE_URL"
echo "SESSION_ID=$SESSION_ID"

echo
echo "== 1. health =="
HEALTH_RESPONSE=$(curl -fsS "$BASE_URL/health")
echo "$HEALTH_RESPONSE"
echo "$HEALTH_RESPONSE" | grep -q '"status":"ok"'

echo
echo "== 2. clear_session before chat =="
curl -fsS -X POST "$BASE_URL/clear_session" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d "{\"session_id\":\"$SESSION_ID\"}"
echo

echo
echo "== 3. chat =="
CHAT_RESPONSE=$(curl -fsS -X POST "$BASE_URL/chat" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d "{\"session_id\":\"$SESSION_ID\",\"message\":\"上海下雨适合去哪玩？\"}")

echo "$CHAT_RESPONSE"
echo "$CHAT_RESPONSE" | grep -Eq "东方明珠|上海博物馆|外滩"

echo
echo "== 4. admin stats =="
STATS_RESPONSE=$(curl -fsS "$BASE_URL/admin/stats" \
  -H "X-API-Key: $API_KEY")

echo "$STATS_RESPONSE"
echo "$STATS_RESPONSE" | grep -q "request_count"

echo
echo "== 5. clear_session after chat =="
curl -fsS -X POST "$BASE_URL/clear_session" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d "{\"session_id\":\"$SESSION_ID\"}"
echo

echo
echo "SMOKE TEST PASSED"
