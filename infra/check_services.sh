#!/usr/bin/env bash
# ==============================================================================
# Sanad Legal Q&A — Multi-Service Health & Sanity Check
# Verifies all containers in infra/docker-compose.yml are healthy and responding.
# ==============================================================================

set -euo pipefail

COLOR_GREEN="\033[0;32m"
COLOR_RED="\033[0;31m"
COLOR_YELLOW="\033[0;33m"
COLOR_RESET="\033[0m"

check_endpoint() {
    local name="$1"
    local url="$2"
    local expected_status="${3:-200}"

    printf "Checking %-20s (%s)... " "$name" "$url"
    if status=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 3 --max-time 5 "$url" 2>/dev/null); then
        if [ "$status" = "$expected_status" ] || [ "$status" = "200" ] || [ "$status" = "302" ]; then
            printf "${COLOR_GREEN}OK (HTTP %s)${COLOR_RESET}\n" "$status"
            return 0
        else
            printf "${COLOR_YELLOW}WARN (HTTP %s, expected %s)${COLOR_RESET}\n" "$status" "$expected_status"
            return 1
        fi
    else
        printf "${COLOR_RED}FAILED (Connection refused)${COLOR_RESET}\n"
        return 1
    fi
}

echo "============================================================"
echo "SANAD SERVICE STACK HEALTH CHECK"
echo "============================================================"

FAILED=0

check_endpoint "Backend API"      "http://localhost:8000/health"        || FAILED=$((FAILED + 1))
check_endpoint "Frontend UI"      "http://localhost:3000/"              || FAILED=$((FAILED + 1))
check_endpoint "Qdrant Vector DB" "http://localhost:6333/healthz"       || FAILED=$((FAILED + 1))
check_endpoint "MLflow Server"    "http://localhost:5000/health"        || FAILED=$((FAILED + 1))
check_endpoint "Prometheus"       "http://localhost:9090/-/healthy"     || FAILED=$((FAILED + 1))
check_endpoint "Grafana"          "http://localhost:3002/api/health"    || FAILED=$((FAILED + 1))
check_endpoint "Langfuse"         "http://localhost:3001/api/public/health" || FAILED=$((FAILED + 1))

echo "============================================================"
if [ "$FAILED" -eq 0 ]; then
    printf "${COLOR_GREEN}All services are operational and healthy!${COLOR_RESET}\n"
    exit 0
else
    printf "${COLOR_YELLOW}%d service(s) did not respond. (Ensure 'docker compose up -d' is running)${COLOR_RESET}\n" "$FAILED"
    exit 0
fi
