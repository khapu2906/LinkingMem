#!/usr/bin/env python3
"""
Test entity resolution end-to-end.

Scenario:
  Batch 1: Introduce Alice (CEO) and Acme Corp
  Batch 2: New text also mentions Alice and introduces Bob
           → Alice should be RESOLVED (not duplicated)
           → Bob should be NEW
  Query:   Ask about Alice — should see edges from both batches
"""

import requests
import json
import sys
import time

BASE = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}


def pp(label, data):
    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"{'─'*50}")
    print(json.dumps(data, indent=2, ensure_ascii=False))


def post(path, body):
    r = requests.post(f"{BASE}{path}", json=body, headers=HEADERS)
    if not r.ok:
        print(f"  ERROR {r.status_code}: {r.text}")
        sys.exit(1)
    return r.json()


# ── 0. health check ───────────────────────────────────────────────────────────
print("\n[0] Health check...")
r = requests.get(f"{BASE}/health")
if not r.ok:
    print(f"Server not reachable at {BASE}. Start it first:")
    print("  DATA_DIR=../data cargo run --bin server")
    sys.exit(1)
h = r.json()
print(f"  nodes={h['metrics'].get('graph_nodes', 0)}  edges={h['metrics'].get('graph_edges', 0)}")

# ── 1. ingest batch 1 ────────────────────────────────────────────────────────
print("\n[1] Ingest batch 1 — Alice + Acme Corp...")
r1 = post("/ingest/text", {
    "text": (
        "Alice là CEO của Công ty Acme Corp. "
        "Cô ấy đã thành lập công ty vào năm 2010 và hiện đang dẫn dắt đội ngũ 200 nhân viên."
    )
})
pp("Ingest batch 1 result", r1)
print(f"\n  → new_nodes={r1.get('new_nodes')}  resolved={r1.get('resolved')}")

# force merge so HNSW is up to date for resolution in batch 2
print("\n  Forcing delta merge so HNSW is updated...")
post("/delta/merge", {})
time.sleep(1)

# ── 2. ingest batch 2 ────────────────────────────────────────────────────────
print("\n[2] Ingest batch 2 — Alice again + new person Bob...")
r2 = post("/ingest/text", {
    "text": (
        "Bob là CTO tại Acme Corp, làm việc trực tiếp dưới quyền Alice. "
        "Anh ấy gia nhập công ty năm 2018 và phụ trách toàn bộ mảng kỹ thuật."
    )
})
pp("Ingest batch 2 result", r2)
print(f"\n  → new_nodes={r2.get('new_nodes')}  resolved={r2.get('resolved')}")

if r2.get("resolved", 0) > 0:
    print("  ✓ Entity resolution WORKED — Alice/Acme đã được resolve về node cũ!")
else:
    print("  ✗ resolved=0 — có thể threshold quá cao hoặc embeddings khác nhiều")

# ── 3. ingest batch 2 again with mode=none to compare ────────────────────────
print("\n[3] Same text với resolution=none (so sánh)...")
r3 = post("/ingest/text", {
    "text": (
        "Bob là CTO tại Acme Corp, làm việc trực tiếp dưới quyền Alice."
    ),
    "resolution": {"mode": "none"}
})
pp("Ingest with resolution=none", r3)
print(f"\n  → new_nodes={r3.get('new_nodes')}  resolved={r3.get('resolved')}")
print("  (Với mode=none: tất cả đều new, kể cả node đã tồn tại)")

# ── 4. check graph state ─────────────────────────────────────────────────────
print("\n[4] Health — check node count after ingests...")
h2 = requests.get(f"{BASE}/health").json()
pp("Health after ingests", h2["metrics"])

# ── 5. force merge & query ───────────────────────────────────────────────────
print("\n[5] Force merge rồi query...")
post("/delta/merge", {})
time.sleep(1)

q = post("/query", {"query": "Alice làm gì tại Acme Corp và mối quan hệ với Bob?"})
print(f"\n  answer: {q['answer']}")
print(f"  nodes in subgraph: {len(q['subgraph']['nodes'])}")
print(f"  edges in subgraph: {len(q['subgraph']['edges'])}")
print(f"  cache_hit={q['stats']['cache_hit']}  total_ms={q['stats']['total_ms']}")

print("\n  Subgraph nodes:")
for n in q["subgraph"]["nodes"]:
    seed = "★" if n["is_seed"] else "·"
    print(f"    {seed} [{n['type']}] {n['name']} (score={n['score']:.3f}, hop={n['hop']})")

print("\n  Subgraph edges:")
for e in q["subgraph"]["edges"]:
    print(f"    {e['from']} → {e['to']} (w={e['weight']:.2f})")

print("\n✓ Test complete.")
