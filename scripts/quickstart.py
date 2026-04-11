#!/usr/bin/env python3
"""
Quick-start script: generate sample data + ingest + test query.
Run: python scripts/quickstart.py
"""

import json, subprocess, sys, os, time, urllib.request

SAMPLE_DATA = {
    "entities": [
        {"id": "e1",  "name": "Nguyễn Văn A",   "type": "Person",  "props": {"role": "CEO"}},
        {"id": "e2",  "name": "Công ty ABC",      "type": "Company", "props": {"industry": "AI"}},
        {"id": "e3",  "name": "Machine Learning", "type": "Concept", "props": {}},
        {"id": "e4",  "name": "Trần Thị B",       "type": "Person",  "props": {"role": "CTO"}},
        {"id": "e5",  "name": "Graph Database",   "type": "Concept", "props": {}},
        {"id": "e6",  "name": "Startup XYZ",      "type": "Company", "props": {"industry": "Fintech"}},
        {"id": "e7",  "name": "Lê Văn C",         "type": "Person",  "props": {"role": "Engineer"}},
        {"id": "e8",  "name": "Vector Search",    "type": "Concept", "props": {}},
        {"id": "e9",  "name": "Knowledge Graph",  "type": "Concept", "props": {}},
        {"id": "e10", "name": "Python",           "type": "Technology", "props": {}},
        {"id": "e11", "name": "Rust",             "type": "Technology", "props": {}},
    ],
    "relations": [
        {"from": "e1",  "to": "e2",  "type": "leads",      "weight": 1.0},
        {"from": "e4",  "to": "e2",  "type": "works_at",   "weight": 1.0},
        {"from": "e7",  "to": "e2",  "type": "works_at",   "weight": 0.8},
        {"from": "e7",  "to": "e6",  "type": "works_at",   "weight": 0.5},
        {"from": "e2",  "to": "e3",  "type": "uses",       "weight": 1.0},
        {"from": "e2",  "to": "e9",  "type": "builds",     "weight": 1.0},
        {"from": "e3",  "to": "e8",  "type": "related_to", "weight": 0.9},
        {"from": "e9",  "to": "e5",  "type": "uses",       "weight": 1.0},
        {"from": "e9",  "to": "e8",  "type": "uses",       "weight": 1.0},
        {"from": "e11", "to": "e5",  "type": "implements", "weight": 1.0},
        {"from": "e10", "to": "e3",  "type": "used_for",   "weight": 0.9},
        {"from": "e1",  "to": "e4",  "type": "collaborates","weight": 0.7},
    ]
}

def run(cmd, **kwargs):
    print(f"\n$ {cmd}")
    return subprocess.run(cmd, shell=True, check=True, **kwargs)

def wait_for(url, retries=10):
    for i in range(retries):
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except:
            print(f"  waiting for {url} ({i+1}/{retries})...")
            time.sleep(2)
    return False

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    # 1. write sample input
    with open("data/input.json", "w") as f:
        json.dump(SAMPLE_DATA, f, ensure_ascii=False, indent=2)
    print("wrote data/input.json")

    # 2. start Python plugin server in background (if not already running)
    print("\nstarting Python plugin server...")
    print("  cd plugins && uvicorn server:app --port 8001 &")
    print("  (or: docker-compose up plugins -d)")

    if not wait_for("http://localhost:8001/health"):
        print("\nPlugin server not running. Start it first:")
        print("  cd plugins")
        print("  uv sync")
        print("  GEMINI_API_KEY=your-key uv run uvicorn server:app --port 8001")
        sys.exit(1)

    # 3. ingest
    run("cd core && cargo run --bin ingest -- --input ../data/input.json --data-dir ../data")

    # 4. start Rust engine
    print("\nstart Rust engine:")
    print("  cd core && DATA_DIR=../data cargo run --bin server")
    print("\nor with docker-compose:")
    print("  docker-compose up")

    # 5. sample query
    print("\nsample query (once engine is running):")
    print("""  curl -X POST http://localhost:8000/query \\
    -H 'Content-Type: application/json' \\
    -d '{"query": "Ai làm việc tại Công ty ABC và họ dùng công nghệ gì?", "mode": "relationship"}'""")
