#!/usr/bin/env bash
# Docker smoke test — builds both images, runs HTTP checks, then cleans up.
# Usage: bash scripts/test_docker.sh
# Requirements: docker, curl
set -euo pipefail

cd "$(dirname "$0")/.."

# ── colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BOLD='\033[1m'; NC='\033[0m'
pass() { echo -e "  ${GREEN}✓${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; exit 1; }
info() { echo -e "\n${BOLD}${YELLOW}▶ $1${NC}"; }

ENGINE_IMG="ai-graph-engine:engine-test"
FULL_IMG="ai-graph-engine:full-test"
ENGINE_PORT=19000
FULL_PORT=19001
NET="ai-graph-test-net"
DATA_ENGINE="/tmp/ai-graph-test-engine"
DATA_FULL="/tmp/ai-graph-test-full"

# ── cleanup ───────────────────────────────────────────────────────────────────
cleanup() {
    echo ""
    info "Cleaning up..."
    docker rm -f ai-test-plugin ai-test-engine ai-test-full 2>/dev/null || true
    docker network rm "$NET" 2>/dev/null         || true
    docker rmi "$ENGINE_IMG" "$FULL_IMG" 2>/dev/null || true
    rm -rf "$DATA_ENGINE" "$DATA_FULL"
    echo -e "  ${GREEN}done${NC}"
}
trap cleanup EXIT

# ── helpers ───────────────────────────────────────────────────────────────────

# Seed an empty-but-valid data directory the engine can boot from.
seed_data() {
    local dir="$1"
    mkdir -p "$dir"
    echo '[]' > "$dir/nodes.json"
    : > "$dir/edges.bin"                      # 0-byte file → 0 edges
    echo '[]' > "$dir/edge_types.json"
    echo '[]' > "$dir/edge_contexts.json"
    echo '[]' > "$dir/edge_embed_contexts.json"
    # vectors.bin: 8-byte header — num_vecs=0, dim=384
    python3 -c "import struct; open('$dir/vectors.bin','wb').write(struct.pack('<II',0,384))"
}

wait_http() {
    local url="$1" label="$2" max="${3:-60}"
    printf "  waiting for %s " "$label"
    for i in $(seq 1 "$max"); do
        if curl -sf "$url" -o /dev/null 2>/dev/null; then
            echo -e " ${GREEN}up${NC} (${i}s)"
            return 0
        fi
        printf "."
        sleep 1
    done
    echo ""
    fail "$label did not respond after ${max}s"
}

check() {
    local label="$1" url="$2" expected="$3"
    local body
    body=$(curl -sf "$url" 2>/dev/null) || fail "$label — request failed"
    if echo "$body" | grep -q "$expected"; then
        pass "$label"
    else
        echo -e "  ${RED}✗${NC} $label"
        echo "    expected to contain: $expected"
        echo "    got: $body" | head -5
        exit 1
    fi
}

post_check() {
    local label="$1" url="$2" data="$3" expected="$4"
    local body
    body=$(curl -sf -X POST -H "Content-Type: application/json" -d "$data" "$url" 2>/dev/null) \
        || fail "$label — request failed"
    if echo "$body" | grep -q "$expected"; then
        pass "$label"
    else
        echo -e "  ${RED}✗${NC} $label"
        echo "    expected to contain: $expected"
        echo "    got: $body"
        exit 1
    fi
}

# ── Step 1: build ─────────────────────────────────────────────────────────────
info "Building images"

echo "  [1/2] engine..."
docker build --target engine -t "$ENGINE_IMG" . -q && pass "engine image built"

echo "  [2/2] full (engine + plugin)..."
docker build --target full   -t "$FULL_IMG"   . -q && pass "full image built"

docker network inspect "$NET" &>/dev/null || docker network create "$NET" > /dev/null

# ── Step 2: test engine image (HTTP mode) ─────────────────────────────────────
info "Test A — engine image  (engine-only container + separate plugin)"

seed_data "$DATA_ENGINE"

# Start plugin container (built from plugins/text/ at engine image build time)
docker run -d --name ai-test-plugin \
    --network "$NET" \
    -p 19002:8001 \
    -e EMBED_MODEL=all-MiniLM-L6-v2 \
    -e EMBED_WARMUP=0 \
    -e LLM_MAX_CONCURRENCY=1 \
    --entrypoint "" \
    "$FULL_IMG" \
    sh -c 'cd /app/plugins && /app/plugins/text/.venv/bin/uvicorn text.main:app --host 0.0.0.0 --port 8001 --workers 1' \
    > /dev/null

wait_http "http://127.0.0.1:19002/health" "plugin" 60

# Start engine container
docker run -d --name ai-test-engine \
    --network "$NET" \
    -p "$ENGINE_PORT":8000 \
    -e PLUGIN_URL=http://ai-test-plugin:8001 \
    -e DATA_DIR=/data \
    -e BIND_ADDR=0.0.0.0:8000 \
    -e RUST_LOG=warn \
    -e DELTA_MERGE_THRESHOLD=500 \
    -v "$DATA_ENGINE":/data \
    "$ENGINE_IMG" \
    > /dev/null

wait_http "http://127.0.0.1:$ENGINE_PORT/health" "engine" 60

# Requests
check      "GET /health → status ok"       "http://127.0.0.1:$ENGINE_PORT/health"  '"status":"ok"'
check      "GET /health → plugin reachable" "http://127.0.0.1:$ENGINE_PORT/health"  '"reachable":true'
check      "GET /metrics → Prometheus"     "http://127.0.0.1:$ENGINE_PORT/metrics" 'queries_total'

INGEST_PAYLOAD='{"entities":[
  {"id":"e1","name":"Alice","type":"Person","props":{}},
  {"id":"e2","name":"Bob","type":"Person","props":{}},
  {"id":"e3","name":"Acme Corp","type":"Company","props":{}}
],"relations":[
  {"from":"e1","to":"e3","type":"works_at","weight":1.0},
  {"from":"e2","to":"e3","type":"works_at","weight":0.8}
]}'

post_check "POST /ingest/json → ingested 3" \
    "http://127.0.0.1:$ENGINE_PORT/ingest/json" \
    "$INGEST_PAYLOAD" \
    '"ingested":3'

check "GET /graph/stats → delta has nodes" \
    "http://127.0.0.1:$ENGINE_PORT/graph/stats" \
    '"delta_nodes":3'

check "GET /metrics → ingest_total 1" \
    "http://127.0.0.1:$ENGINE_PORT/metrics" \
    'ingest_total 1'

echo ""
echo -e "  ${GREEN}Test A passed${NC}"

# ── Step 3: test full image (single container) ────────────────────────────────
info "Test B — full image  (engine + plugin in one container, unix socket)"

seed_data "$DATA_FULL"

docker run -d --name ai-test-full \
    -p "$FULL_PORT":8000 \
    -e EMBED_MODEL=all-MiniLM-L6-v2 \
    -e EMBED_WARMUP=0 \
    -e DATA_DIR=/data \
    -e BIND_ADDR=0.0.0.0:8000 \
    -e RUST_LOG=warn \
    -e DELTA_MERGE_THRESHOLD=500 \
    -v "$DATA_FULL":/data \
    "$FULL_IMG" \
    > /dev/null

wait_http "http://127.0.0.1:$FULL_PORT/health" "full container" 120

check      "GET /health → status ok"        "http://127.0.0.1:$FULL_PORT/health"  '"status":"ok"'
check      "GET /health → plugin reachable" "http://127.0.0.1:$FULL_PORT/health"  '"reachable":true'
check      "GET /metrics → Prometheus"      "http://127.0.0.1:$FULL_PORT/metrics" 'queries_total'

post_check "POST /ingest/json → ingested 3" \
    "http://127.0.0.1:$FULL_PORT/ingest/json" \
    "$INGEST_PAYLOAD" \
    '"ingested":3'

check "GET /graph/stats → delta has nodes" \
    "http://127.0.0.1:$FULL_PORT/graph/stats" \
    '"delta_nodes":3'

check "GET /metrics → ingest_total 1" \
    "http://127.0.0.1:$FULL_PORT/metrics" \
    'ingest_total 1'

echo ""
echo -e "  ${GREEN}Test B passed${NC}"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}All tests passed.${NC}"
echo -e "Containers + images will be removed on exit."
