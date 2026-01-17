#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import os
import re
import sys
import time
import socket
import threading
import http.server
import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from functools import partial

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError


# ----------------------------
# CONFIGURATION (your settings)
# ----------------------------
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "PASSWORD"
NEO4J_DATABASE = "neo4j"

DBMS_DIR = Path(
    "DBMS_DIR/dbmss/"
    "dbms-UUID"
)

RESULTS_CSV = Path("neo4j_load_benchmark_results.csv")

DELETE_BATCH_SIZE = 10000
TZ = ZoneInfo("Europe/Warsaw")


# ----------------------------
# HTTP SERVER (for apoc.import.csv via localhost)
# ----------------------------
HTTP_SERVE_ROOT = Path(__file__).resolve().parent / "datasets"
HTTP_BIND_HOST = "::"          # IPv6 dual-stack (usually also supports IPv4)
HTTP_DEFAULT_PORT = 8000       # used when it cannot be detected from files


class DualStackThreadingHTTPServer(http.server.ThreadingHTTPServer):
    """
    HTTP server with a best-effort dual-stack (IPv6 + IPv4) setup.
    On many Linux systems, binding to "::" also accepts IPv4 connections.
    """
    address_family = socket.AF_INET6


class HttpServerController:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.httpd = None
        self.thread = None
        self.port = None

    def start(self, port: int) -> None:
        if self.httpd is not None:
            raise RuntimeError("HTTP server already running")

        handler_cls = partial(http.server.SimpleHTTPRequestHandler, directory=str(self.root_dir))

        # Try dual-stack
        try:
            self.httpd = DualStackThreadingHTTPServer((HTTP_BIND_HOST, port), handler_cls)
        except OSError:
            # Fallback to IPv4-only
            self.httpd = http.server.ThreadingHTTPServer(("0.0.0.0", port), handler_cls)

        self.port = port
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if self.httpd is None:
            return
        try:
            self.httpd.shutdown()
            self.httpd.server_close()
        finally:
            self.httpd = None
            self.thread = None
            self.port = None

    @property
    def base_url(self) -> str:
        if self.port is None:
            return ""
        return f"http://localhost:{self.port}"


def detect_port_from_load_file(load_file: Path) -> int | None:
    """
    Searches the file for occurrences of:
      http://localhost:PORT/...
      http://127.0.0.1:PORT/...
    and returns PORT if found.
    """
    text = load_file.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"http://(?:localhost|127\.0\.0\.1):(\d+)\b", text)
    if not m:
        return None
    return int(m.group(1))


# ----------------------------
# MODELS
# ----------------------------
@dataclass(frozen=True)
class Experiment:
    dataset: str
    variant: str
    load_file: Path


@dataclass(frozen=True)
class Workload:
    name: str
    cypher: str


# ----------------------------
# HELPERS: directory size
# ----------------------------
def directory_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = Path(root) / f
            try:
                total += fp.stat().st_size
            except (FileNotFoundError, PermissionError):
                pass
    return total


def human_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n)
    for u in units:
        if x < 1024.0:
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{x:.2f} PiB"


# ----------------------------
# PARSING .cypher FILES
# ----------------------------
def split_cypher_statements(text: str) -> list[str]:
    statements = []
    buf = []

    in_sq = False
    in_dq = False
    in_bt = False
    in_line_comment = False
    in_block_comment = False

    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
                buf.append(ch)
            i += 1
            continue

        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
            else:
                i += 1
            continue

        if not in_sq and not in_dq and not in_bt:
            if ch == "/" and nxt == "/":
                in_line_comment = True
                i += 2
                continue
            if ch == "/" and nxt == "*":
                in_block_comment = True
                i += 2
                continue

        if not in_dq and not in_bt and ch == "'":
            if in_sq and nxt == "'":
                buf.append("''")
                i += 2
                continue
            in_sq = not in_sq
            buf.append(ch)
            i += 1
            continue

        if not in_sq and not in_bt and ch == '"':
            in_dq = not in_dq
            buf.append(ch)
            i += 1
            continue

        if not in_sq and not in_dq and ch == "`":
            in_bt = not in_bt
            buf.append(ch)
            i += 1
            continue

        if not in_sq and not in_dq and not in_bt and ch == ";":
            stmt = "".join(buf).strip()
            if stmt:
                statements.append(stmt)
            buf = []
            i += 1
            continue

        buf.append(ch)
        i += 1

    tail = "".join(buf).strip()
    if tail:
        statements.append(tail)

    cleaned = []
    for s in statements:
        ss = s.strip()
        if not ss:
            continue
        if ss.startswith(":"):  # Neo4j Browser directives
            continue
        cleaned.append(ss)

    return cleaned


def load_cypher_file_statements(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return split_cypher_statements(text)


# ----------------------------
# NEO4J OPERATIONS
# ----------------------------
def neo4j_versions(session) -> tuple[str, str]:
    neo4j_v = "unknown"
    apoc_v = "unknown"

    try:
        rec = session.run("CALL dbms.components() YIELD name, versions RETURN name, versions").single()
        if rec and rec["versions"]:
            neo4j_v = rec["versions"][0]
    except Exception:
        pass

    try:
        rec2 = session.run("CALL apoc.version() YIELD version RETURN version").single()
        if rec2:
            apoc_v = rec2["version"]
    except Exception:
        pass

    return neo4j_v, apoc_v


def drop_all_schema(session) -> None:
    try:
        cons = session.run("SHOW CONSTRAINTS YIELD name RETURN name").values()
        for (name,) in cons:
            session.run(f"DROP CONSTRAINT `{name}` IF EXISTS").consume()
    except Neo4jError as e:
        print(f"[WARN] Failed to drop all constraints: {e}")

    try:
        idxs = session.run("SHOW INDEXES YIELD name, type WHERE type <> 'LOOKUP' RETURN name").values()
        for (name,) in idxs:
            session.run(f"DROP INDEX `{name}` IF EXISTS").consume()
    except Neo4jError as e:
        print(f"[WARN] Failed to drop all indexes: {e}")


def clear_database(session) -> None:
    print("  - DROP schema (constraints & indexes)")
    drop_all_schema(session)

    print("  - DELETE all nodes (batched)")
    session.run(
        f"""
        MATCH (n)
        CALL (n) {{
          DETACH DELETE n
        }} IN TRANSACTIONS OF {DELETE_BATCH_SIZE} ROWS
        """
    ).consume()

    try:
        session.run("CALL db.checkpoint()").consume()
    except Exception:
        pass


def run_statements(session, statements: list[str]) -> None:
    for stmt in statements:
        s = stmt.strip()
        if not s:
            continue
        session.run(s).consume()


def count_graph(session) -> tuple[int, int]:
    nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
    rels = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
    return int(nodes), int(rels)


# ----------------------------
# WORKLOAD QUERIES (query-time overhead)
# ----------------------------
WQ_A1 = Workload("WQ_A1", """
MATCH (n:Entity)
WHERE "Person" IN n.k_noval
RETURN count(n) AS persons;
""".strip())

WQ_A2_BASE = Workload("WQ_A2", """
MATCH (p:Entity {id:$pid})-[r:REL]->(q:Entity)
WHERE "KNOWS" IN r.k_noval
RETURN count(q) AS deg;
""".strip())

WQ_A2_REIF = Workload("WQ_A2", """
MATCH (p:Entity {id:$pid})<-[:SRC]-(f:EdgeFact)-[:TGT]->(q:Entity)
WHERE "KNOWS" IN f.k_noval
RETURN count(q) AS deg;
""".strip())

WQ_A3_BASE = Workload("WQ_A3", """
MATCH (c:Entity)
WHERE "City" IN c.k_noval
  AND any(x IN c.population WHERE x > $pop)
RETURN count(c) AS cities;
""".strip())

WQ_A3_OCC = Workload("WQ_A3", """
MATCH (c:Entity)<-[:SRC]-(p:PropEdge)-[:TGT]->(v:Value)
WHERE "City" IN c.k_noval
  AND p.pkey = "population"
  AND v.val > $pop
RETURN count(DISTINCT c) AS cities;
""".strip())

WQ_B1_BASE = Workload("WQ_B1", """
MATCH (:Entity)-[r:REL]->(:Entity)
WHERE "KNOWS" IN r.k_noval AND r.m_since >= $since
RETURN count(r) AS facts;
""".strip())

WQ_B1_REIF = Workload("WQ_B1", """
MATCH (f:EdgeFact)-[:SRC]->(a:Entity), (f)-[:TGT]->(b:Entity)
WHERE "KNOWS" IN f.k_noval AND f.m_since >= $since
RETURN count(f) AS facts;
""".strip())

WQ_B2_BASE = Workload("WQ_B2", """
MATCH ()-[r:REL]->()
WHERE "KNOWS" IN r.k_noval AND r.m_since >= $since
MATCH (s:Source {id:r.m_sourceId})
RETURN s.id, s.name, count(r) AS facts
ORDER BY facts DESC;
""".strip())

WQ_B2_REIF = Workload("WQ_B2", """
MATCH (f:EdgeFact)-[:HAS_SOURCE]->(s:Source)
WHERE "KNOWS" IN f.k_noval AND f.m_since >= $since
RETURN s.id, s.name, count(f) AS facts
ORDER BY facts DESC;
""".strip())

WQ_C1 = Workload("WQ_C1", """
MATCH (c:Entity {id:$cid})<-[:SRC]-(p:PropEdge)-[:TGT]->(v:Value)
WHERE p.pkey = "population"
  AND p.m_point_in_time = $year
  AND p.m_confidence >= $conf
RETURN v.val AS population, p.m_point_in_time AS year, p.m_confidence AS conf;
""".strip())

WQ_C2 = Workload("WQ_C2", """
MATCH (c:Entity {id:$cid})<-[:SRC]-(p:PropEdge)-[:TGT]->(v:Value)
WHERE p.pkey = "population"
RETURN v.val AS population, p.m_point_in_time AS year, p.m_confidence AS conf
ORDER BY conf DESC
LIMIT $k;
""".strip())

WQ_C3 = Workload("WQ_C3", """
MATCH (c:Entity)
WHERE "City" IN c.k_noval AND EXISTS {
  MATCH (c)<-[:SRC]-(p:PropEdge)-[:TGT]->(v:Value)
  WHERE p.pkey = "population"
    AND v.val > $pop
    AND p.m_confidence >= $conf
}
RETURN count(c) AS cities;
""".strip())

WQ_D1 = Workload("WQ_D1", """
MATCH (p:Entity)
WHERE "Person" IN p.k_noval AND EXISTS {
  MATCH (p)<-[:SRC]-(pe:PropEdge)-[:TGT]->(root:Value)
  WHERE pe.pkey = "address"
  MATCH (ce:CompEdge)-[:SRC]->(root), (ce)-[:TGT]->(z:Value)
  WHERE ce.selKind = "field" AND ce.selVal = "zip"
    AND ce.m_curator = $curator
}
RETURN count(p) AS persons;
""".strip())

WQ_D2 = Workload("WQ_D2", """
MATCH (p:Entity)
WHERE "Person" IN p.k_noval AND EXISTS {
  MATCH (p)<-[:SRC]-(pe:PropEdge)-[:TGT]->(root:Value)
  WHERE pe.pkey = "address"
  MATCH (ce:CompEdge)-[:SRC]->(root)
  WHERE ce.selKind = "field" AND ce.selVal = $field
    AND ce.m_lastUpdated >= $dateISO
}
RETURN count(p) AS persons;
""".strip())

WQ_D3 = Workload("WQ_D3", """
MATCH (p:Entity {id:$pid})<-[:SRC]-(pe:PropEdge)-[:TGT]->(root:Value)
WHERE pe.pkey = "address"
MATCH (ce:CompEdge)-[:SRC]->(root), (ce)-[:TGT]->(v:Value)
WHERE ce.selKind = "field" AND ce.selVal IN ["zip","city"]
RETURN root.val AS address_json,
       ce.selVal AS field, v.val AS field_value,
       ce.m_curator AS curator, ce.m_lastUpdated AS lastUpdated
ORDER BY field;
""".strip())


def workloads_for_variant(variant: str) -> list[Workload]:
    """
    Choose workloads depending on the encoding:
    - gpg:  :REL + direct properties on the node
    - gpge: EdgeFact (relationship reification), but population still on the node
    - gpgp/gpgc: PropEdge/Value for population (occurrences), plus C
    - gpgc: additionally CompEdge (component-level), plus D
    """
    wqs: list[Workload] = []
    wqs.append(WQ_A1)

    if variant == "gpg":
        wqs.extend([WQ_A2_BASE, WQ_A3_BASE, WQ_B1_BASE, WQ_B2_BASE])
        return wqs

    # reified: gpge/gpgp/gpgc
    wqs.append(WQ_A2_REIF)

    if variant == "gpge":
        wqs.append(WQ_A3_BASE)
    else:
        wqs.append(WQ_A3_OCC)

    wqs.extend([WQ_B1_REIF, WQ_B2_REIF])

    if variant in ("gpgp", "gpgc"):
        wqs.extend([WQ_C1, WQ_C2, WQ_C3])

    if variant == "gpgc":
        wqs.extend([WQ_D1, WQ_D2, WQ_D3])

    return wqs


def pick_one_id(session, kind_label: str, default: str = "0") -> str:
    """
    Fetches an example id for an Entity having kind_label in k_noval, e.g., Person/City.
    """
    rec = session.run(
        """
        MATCH (x:Entity)
        WHERE $lbl IN x.k_noval AND x.id IS NOT NULL
        RETURN x.id AS id
        LIMIT 1
        """,
        {"lbl": kind_label},
    ).single()
    if rec and rec.get("id") is not None:
        return str(rec["id"])
    return default


def pick_one_curator(session, default: str = "Editor42") -> str:
    rec = session.run(
        """
        MATCH (ce:CompEdge)
        WHERE ce.m_curator IS NOT NULL
        RETURN ce.m_curator AS c
        LIMIT 1
        """
    ).single()
    if rec and rec.get("c") is not None:
        return str(rec["c"])
    return default


def query_params_for_benchmark(session) -> dict:
    """
    Parameters for workloads:
    - pid/cid are picked from the graph, the rest are fixed (sensible and repeatable).
    """
    pid = pick_one_id(session, "Person", default="0")
    cid = pick_one_id(session, "City", default="0")
    curator = pick_one_curator(session, default="Editor42")

    return {
        "pid": pid,
        "cid": cid,
        "pop": 100000,
        "since": 2010,
        "year": 2005,
        "conf": 0.5,
        "k": 10,
        "curator": curator,
        "field": "zip",
        "dateISO": "2024-01-01",
    }


def consume_fully(result) -> None:
    """
    For fair timing: force the server to fully finish the query.
    """
    result.consume()


def run_query_benchmark(session, variant: str, qreps: int, qwarmup: int) -> dict[str, float]:
    """
    Returns a dict with timings:
      - per-workload times: q_WQ_..._s (mean over qreps)
      - q_total_ab_time_s: sum of WQ_A1..WQ_B2 (common part across variants)
      - q_total_time_s: sum of all workloads executed for this variant
    """
    params = query_params_for_benchmark(session)
    wqs = workloads_for_variant(variant)

    times: dict[str, float] = {}

    for wq in wqs:
        # warmup (no measurement)
        for _ in range(max(0, qwarmup)):
            consume_fully(session.run(wq.cypher, params))

        # measurement (averaging)
        durs = []
        for _ in range(max(1, qreps)):
            t0 = time.perf_counter()
            consume_fully(session.run(wq.cypher, params))
            t1 = time.perf_counter()
            durs.append(t1 - t0)

        avg = statistics.mean(durs)
        times[f"q_{wq.name}_s"] = avg

    # 1) total sum (only what was actually executed as workload)
    wq_sum = sum(times.values())

    # 2) common sum A+B (comparable for all variants)
    ab_keys = ["q_WQ_A1_s", "q_WQ_A2_s", "q_WQ_A3_s", "q_WQ_B1_s", "q_WQ_B2_s"]
    ab_sum = 0.0
    for k in ab_keys:
        if k in times:
            ab_sum += times[k]

    # store aggregates
    times["q_total_ab_time_s"] = ab_sum
    times["q_total_time_s"] = wq_sum
    return times


# ----------------------------
# CSV RESULTS
# ----------------------------
RESULT_COLUMNS = [
    "timestamp",
    "dataset",
    "variant",
    "load_script",
    "http_base_url",
    "load_time_s",
    "nodes",
    "rels",
    "db_store_bytes",
    "db_tx_bytes",
    "db_total_bytes",
    "db_total_human",
    "neo4j_version",
    "apoc_version",

    # --- query-time overhead (absolute times) ---
    "q_total_time_s",
    "q_total_ab_time_s",
    "q_WQ_A1_s",
    "q_WQ_A2_s",
    "q_WQ_A3_s",
    "q_WQ_B1_s",
    "q_WQ_B2_s",
    "q_WQ_C1_s",
    "q_WQ_C2_s",
    "q_WQ_C3_s",
    "q_WQ_D1_s",
    "q_WQ_D2_s",
    "q_WQ_D3_s",

    # --- overhead vs baseline (gpg), computed for the common A+B part ---
    "q_total_ab_vs_gpg_ratio",
    "q_total_ab_vs_gpg_pct",
]


def ensure_results_csv(path: Path) -> None:
    """
    Safe CSV creation:
    - if the file does not exist -> create it with the current header,
    - if it exists but the header does not match -> backup and create a new one.
    """
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
            w.writeheader()
        return

    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, [])
        if header != RESULT_COLUMNS:
            backup = path.with_name(
                path.stem + f".bak_{datetime.now(TZ).strftime('%Y%m%d_%H%M%S')}" + path.suffix
            )
            path.rename(backup)
            with path.open("w", newline="", encoding="utf-8") as f2:
                w = csv.DictWriter(f2, fieldnames=RESULT_COLUMNS)
                w.writeheader()
            print(f"[WARN] CSV schema changed -> old file moved to: {backup}")
    except Exception:
        # Do not block the benchmark in case of an unusual file state
        pass


def append_result(path: Path, row: dict) -> None:
    ensure_results_csv(path)
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        w.writerow(row)


# ----------------------------
# BENCHMARK
# ----------------------------
def db_paths(dbms_dir: Path, db_name: str) -> tuple[Path, Path]:
    store = dbms_dir / "data" / "databases" / db_name
    tx = dbms_dir / "data" / "transactions" / db_name
    return store, tx


def experiments_from_files(base_dir: Path) -> list[Experiment]:
    datasets = ["D1-S", "D2-M", "D3-L"]
    variants = ["gpg", "gpge", "gpgp", "gpgc"]

    exps = []
    for ds in datasets:
        for var in variants:
            f = base_dir / f"load_{ds}_{var}.cypher"
            exps.append(Experiment(dataset=ds, variant=var, load_file=f))
    return exps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Neo4j load benchmark (clear -> load -> time -> disk size) + query-time overhead + overhead vs baseline (gpg)."
    )
    parser.add_argument(
        "-r", "--reps",
        type=int,
        default=5,
        help="Number of repetitions for each (dataset, variant). Default: 5."
    )
    parser.add_argument(
        "--qreps",
        type=int,
        default=3,
        help="Number of repetitions for each workload query (for averaging). Default: 3."
    )
    parser.add_argument(
        "--qwarmup",
        type=int,
        default=1,
        help="How many warmup runs per query (not measured). Default: 1."
    )
    parser.add_argument(
        "--noqueries",
        action="store_true",
        help="If set, skip workload queries (only load + size)."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reps = args.reps
    qreps = args.qreps
    qwarmup = args.qwarmup
    noqueries = args.noqueries

    if reps <= 0:
        print("[ERROR] --reps must be a positive integer (>= 1).")
        return 2
    if qreps <= 0:
        print("[ERROR] --qreps must be a positive integer (>= 1).")
        return 2
    if qwarmup < 0:
        print("[ERROR] --qwarmup cannot be negative.")
        return 2

    base_dir = Path(__file__).resolve().parent
    exps = experiments_from_files(base_dir)

    missing = [e.load_file for e in exps if not e.load_file.exists()]
    if missing:
        print("Missing load_*.cypher files:")
        for m in missing:
            print(f"  - {m}")
        return 2

    store_path, tx_path = db_paths(DBMS_DIR, NEO4J_DATABASE)

    print("==== Neo4j Load Benchmark ====")
    print(f"URI:          {NEO4J_URI}")
    print(f"Database:     {NEO4J_DATABASE}")
    print(f"DBMS dir:     {DBMS_DIR}")
    print(f"Store path:   {store_path}")
    print(f"Tx path:      {tx_path}")
    print(f"Results CSV:  {RESULTS_CSV.resolve()}")
    print(f"Repetitions:  {reps}")
    print(f"Query reps:   {qreps} (+ warmup={qwarmup})")
    print(f"Queries:      {'OFF' if noqueries else 'ON'}")
    print("================================\n")

    # Sanity check: datasets/ must exist, because we serve it over HTTP
    if not HTTP_SERVE_ROOT.exists():
        print(f"[ERROR] Could not find datasets directory to serve: {HTTP_SERVE_ROOT}")
        return 2

    http_ctrl = HttpServerController(HTTP_SERVE_ROOT)
    current_port = None

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    # --- baseline cache (per-dataset), computed from gpg over reps ---
    baseline_ab_runs: dict[str, list[float]] = {}
    baseline_ab_mean: dict[str, float] = {}

    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            neo4j_v, apoc_v = neo4j_versions(session)
            print(f"Detected Neo4j version: {neo4j_v}")
            print(f"Detected APOC  version: {apoc_v}")
            print()

        for e in exps:
            desired_port = detect_port_from_load_file(e.load_file) or HTTP_DEFAULT_PORT

            # Start/Restart HTTP server if needed
            if current_port != desired_port:
                if current_port is not None:
                    print(f"Stopping HTTP server on port {current_port} ...")
                    http_ctrl.stop()
                print(f"Starting HTTP server on port {desired_port}, serving: {HTTP_SERVE_ROOT}")
                http_ctrl.start(desired_port)
                current_port = desired_port
                print(f"HTTP base URL: {http_ctrl.base_url}\n")

            # REPETITIONS
            for rep in range(1, reps + 1):
                print(f"==> RUN: dataset={e.dataset} variant={e.variant} file={e.load_file.name}  [repeat {rep}/{reps}]")

                with driver.session(database=NEO4J_DATABASE) as session:
                    # 1) clearing
                    t0_clear = time.perf_counter()
                    clear_database(session)
                    t1_clear = time.perf_counter()
                    print(f"  - Clear done in {t1_clear - t0_clear:.3f}s")

                    # 2) load
                    statements = load_cypher_file_statements(e.load_file)

                    t0 = time.perf_counter()
                    run_statements(session, statements)
                    t1 = time.perf_counter()

                    load_time = t1 - t0
                    print(f"  - Load done in {load_time:.3f}s")

                    try:
                        session.run("CALL db.checkpoint()").consume()
                    except Exception:
                        pass

                    nodes, rels = count_graph(session)
                    print(f"  - Graph counts: nodes={nodes}, rels={rels}")

                    store_bytes = directory_size_bytes(store_path)
                    tx_bytes = directory_size_bytes(tx_path)
                    total_bytes = store_bytes + tx_bytes

                    print(f"  - Store size: {human_bytes(store_bytes)}")
                    print(f"  - Tx size:    {human_bytes(tx_bytes)}")
                    print(f"  - Total:      {human_bytes(total_bytes)}")

                    # 3) query-time overhead (workload queries)
                    query_times: dict[str, float] = {}
                    if not noqueries:
                        print("  - Running workload queries (query-time overhead)...")
                        query_times = run_query_benchmark(session, e.variant, qreps=qreps, qwarmup=qwarmup)
                        print(f"    * q_total_time_s    = {query_times.get('q_total_time_s', 0.0):.6f}s")
                        print(f"    * q_total_ab_time_s = {query_times.get('q_total_ab_time_s', 0.0):.6f}s")

                        # collect baseline (gpg) to later compute the mean per dataset
                        if e.variant == "gpg":
                            ab_val = query_times.get("q_total_ab_time_s", None)
                            if isinstance(ab_val, (float, int)):
                                baseline_ab_runs.setdefault(e.dataset, []).append(float(ab_val))

                    # Timestamp with milliseconds (easier to distinguish repetitions)
                    timestamp = datetime.now(TZ).isoformat(timespec="milliseconds")

                    # result row
                    row = {
                        "timestamp": timestamp,
                        "dataset": e.dataset,
                        "variant": e.variant,
                        "load_script": e.load_file.name,
                        "http_base_url": http_ctrl.base_url,
                        "load_time_s": f"{load_time:.3f}",
                        "nodes": nodes,
                        "rels": rels,
                        "db_store_bytes": store_bytes,
                        "db_tx_bytes": tx_bytes,
                        "db_total_bytes": total_bytes,
                        "db_total_human": human_bytes(total_bytes),
                        "neo4j_version": neo4j_v,
                        "apoc_version": apoc_v,
                    }

                    # initialize query fields as empty (fixed CSV schema)
                    for col in [
                        "q_total_time_s",
                        "q_total_ab_time_s",
                        "q_WQ_A1_s", "q_WQ_A2_s", "q_WQ_A3_s",
                        "q_WQ_B1_s", "q_WQ_B2_s",
                        "q_WQ_C1_s", "q_WQ_C2_s", "q_WQ_C3_s",
                        "q_WQ_D1_s", "q_WQ_D2_s", "q_WQ_D3_s",
                        "q_total_ab_vs_gpg_ratio",
                        "q_total_ab_vs_gpg_pct",
                    ]:
                        row[col] = ""

                    # write query times
                    for k, v in query_times.items():
                        row[k] = f"{v:.6f}"

                    # --- AUTOMATIC OVERHEAD COMPUTATION VS BASELINE (gpg) ---
                    # We compare ONLY q_total_ab_time_s (the common A+B part),
                    # because q_total_time_s may include extra workloads (C, D) and is not directly “fair”.
                    if (not noqueries) and ("q_total_ab_time_s" in query_times):
                        this_ab = query_times.get("q_total_ab_time_s", None)

                        if isinstance(this_ab, (float, int)):
                            this_ab = float(this_ab)

                            if e.variant == "gpg":
                                # baseline vs baseline -> always 1 and 0%
                                row["q_total_ab_vs_gpg_ratio"] = f"{1.0:.6f}"
                                row["q_total_ab_vs_gpg_pct"] = f"{0.0:.6f}"
                            else:
                                base = baseline_ab_mean.get(e.dataset, None)
                                if isinstance(base, (float, int)) and float(base) > 0.0:
                                    base = float(base)
                                    ratio = this_ab / base
                                    pct = (ratio - 1.0) * 100.0
                                    row["q_total_ab_vs_gpg_ratio"] = f"{ratio:.6f}"
                                    row["q_total_ab_vs_gpg_pct"] = f"{pct:.6f}"
                                else:
                                    # baseline not known yet (should not happen if order is gpg->...)
                                    row["q_total_ab_vs_gpg_ratio"] = ""
                                    row["q_total_ab_vs_gpg_pct"] = ""

                    append_result(RESULTS_CSV, row)

                print("  - Result appended.\n")

            # After finishing the gpg series for a given dataset:
            # compute baseline mean and store it for subsequent variants.
            if (not noqueries) and (e.variant == "gpg"):
                vals = baseline_ab_runs.get(e.dataset, [])
                if vals:
                    baseline_ab_mean[e.dataset] = statistics.mean(vals)
                    print(f"[BASELINE] dataset={e.dataset} gpg mean q_total_ab_time_s = {baseline_ab_mean[e.dataset]:.6f}s\n")
                else:
                    print(f"[WARN] No baseline values for dataset={e.dataset} (gpg), overheads will remain empty.\n")

        print("DONE. Results saved to:")
        print(f"  {RESULTS_CSV.resolve()}")
        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C).")
        return 130
    except Neo4jError as e:
        print("\n[ERROR] Neo4jError:")
        print(str(e))
        return 1
    except Exception as e:
        print("\n[ERROR] Exception:")
        print(str(e))
        return 1
    finally:
        try:
            http_ctrl.stop()
        except Exception:
            pass
        driver.close()


if __name__ == "__main__":
    raise SystemExit(main())

