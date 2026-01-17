# Neo4j Experimental Study — Synthetic GPG Datasets + Load & Query Benchmarks

This repository contains a **reproducible experimental pipeline** for evaluating the storage, loading, and query-time overhead of increasingly expressive metadata encodings in a Neo4j backend.

It provides:

- a **deterministic synthetic dataset generator** producing three datasets (*D1-S*, *D2-M*, *D3-L*),
- four **encoding variants** (*gpg*, *gpge*, *gpgp*, *gpgc*),
- a **benchmark runner** that repeatedly:
  1) clears the database,  
  2) loads CSV data via `apoc.import.csv`,  
  3) measures load time and on-disk sizes,  
  4) optionally runs a **workload suite** (WQ-A–WQ-D) to measure query-time overhead,  
  5) appends results to a uniform CSV file.

A consolidated listing of the Cypher workloads and the loading workflow is provided in **`online_appendix.md`**.

---

## Repository layout

```

├── benchmark_loads.py
├── datasets
│   ├── D1-S
│   ├── D2-M
│   └── D3-L
├── gpg_dataset_generator.py
├── load_D1-S_gpgc.cypher
├── load_D1-S_gpg.cypher
├── load_D1-S_gpge.cypher
├── load_D1-S_gpgp.cypher
├── load_D2-M_gpgc.cypher
├── load_D2-M_gpg.cypher
├── load_D2-M_gpge.cypher
├── load_D2-M_gpgp.cypher
├── load_D3-L_gpgc.cypher
├── load_D3-L_gpg.cypher
├── load_D3-L_gpge.cypher
├── load_D3-L_gpgp.cypher
└── online_appendix.md

```

### What each file does

- **`gpg_dataset_generator.py`**  
  Generates `datasets/<DATASET>/<VARIANT>/*.csv` plus an `import.apoc.cypher` helper script.
  The generator is deterministic given `--seed`.

- **`load_*.cypher`**  
  Ready-to-run Neo4j Browser scripts containing `CALL apoc.import.csv(...)` with **hardcoded URLs**
  (assuming files are served from `http://localhost:8000/`).

- **`benchmark_loads.py`**  
  Automated benchmark runner:
  - starts/stops a local HTTP server serving `./datasets`,
  - clears Neo4j,
  - executes `load_*.cypher` statements,
  - measures load time and disk size,
  - optionally executes workload queries (WQ-A–WQ-D),
  - writes results to `neo4j_load_benchmark_results.csv`.

- **`online_appendix.md`**  
  Human-readable reference: workload definitions (WQ-A–WQ-D), loading conventions, APOC import settings,
  and recommended indexes/constraints.

---

## Encodings (variants)

Each dataset can be instantiated in four encodings:

- **`gpg` (baseline)**  
  Logical relationship types are carried by membership in `k_noval`, while relationships are stored as native `:REL`.
  City populations are stored as a multi-valued property (value-set) on the city node.

- **`gpge` (edge-fact reification)**  
  Each logical relationship fact is represented by an `:EdgeFact` node connected to endpoints via `:SRC` and `:TGT`.
  Provenance sources are connected via `:HAS_SOURCE`. City populations remain a value-set on the city node.

- **`gpgp` (property-occurrence reification)**  
  In addition to edge-facts, population values are stored as explicit **occurrences** via:
  `(:Entity)<-[:SRC]-(:PropEdge)-[:TGT]->(:Value)`  
  where `PropEdge` carries metadata such as `m_point_in_time`, `m_confidence`.

- **`gpgc` (component-level metadata)**  
  Extends `gpgp` with **structured values** and **field-level metadata** using:
  `(:CompEdge)-[:SRC]->(root:Value)` and `(:CompEdge)-[:TGT]->(fieldValue:Value)`  
  enabling metadata directly on sub-values (e.g., ZIP field curator / last-updated).

---

## Datasets

Three synthetic datasets are generated (scale increases from D1 to D3).  
Each dataset contains:

- two entity kinds: `Person` and `City`,
- logical relationship facts: `LIVES_IN` and `KNOWS`,
- (depending on encoding) multi-valued population properties and structured addresses.

The dataset generator uses a deterministic construction with no global RNG state, enabling reproducible regeneration
given `--seed`.

---

## Requirements

### Software requirements

- **Python 3.10+** (recommended)
- Python packages:
  - `neo4j` (official Neo4j Python driver)

Install dependencies:

```bash
pip install neo4j
````

### Neo4j requirements (tested setup)

* **Neo4j 4.4.42** (community or enterprise)
* **APOC 4.4.x** installed and enabled

> The benchmark runner uses APOC procedures (notably `apoc.import.csv` and `apoc.version()`).

### Local HTTP access requirement

Imports are performed via URLs such as:

```
http://localhost:8000/D1-S/gpg/nodes_entities.csv
```

Therefore, the dataset directory must be served via an HTTP server.
`benchmark_loads.py` automatically starts such a server, but you may also run one manually.

---

## Quick start

### 1) Generate datasets

Generate all datasets and all encodings into `./datasets`:

```bash
python3 gpg_dataset_generator.py --out ./datasets --seed 1 --values-mode split
```

Common options:

* `--seed <int>`: controls deterministic generation (same seed → identical output)
* `--values-mode split|single`:

  * `split` (default): creates numeric values in `nodes_values_num.csv` and string values in `nodes_values_str.csv`
  * `single`: stores all values as strings in `nodes_values.csv`

### 2) Start Neo4j

Ensure Neo4j is running and that Bolt is reachable at:

```
bolt://localhost:7687
```

and that APOC is enabled.

### 3) Run the benchmark suite

Run the benchmark with default settings:

```bash
python3 benchmark_loads.py
```

Useful flags:

```bash
python3 benchmark_loads.py --reps 5 --qreps 3 --qwarmup 1
```

To measure only **load + disk size** (skip queries):

```bash
python3 benchmark_loads.py --noqueries
```

---

## Benchmark configuration (important)

`benchmark_loads.py` contains a configuration block near the top:

```python
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "domeldomel"
NEO4J_DATABASE = "neo4j"

DBMS_DIR = Path("/home/domel/.config/Neo4j Desktop/Application/relate-data/dbmss/dbms-...")

RESULTS_CSV = Path("neo4j_load_benchmark_results.csv")
```

You must adapt at least:

* `NEO4J_PASSWORD` (and possibly user/database name)
* `DBMS_DIR` so the script can measure on-disk sizes:

  * store size is computed from:
    `DBMS_DIR/data/databases/<db_name>`
  * transaction size is computed from:
    `DBMS_DIR/data/transactions/<db_name>`

> If `DBMS_DIR` is incorrect, the benchmark will still run, but disk-size measurements will be wrong or zero.

---

## How loading works

### HTTP server

`benchmark_loads.py` serves the `./datasets` directory via a built-in HTTP server:

* bind host: `::` (dual stack IPv6 + IPv4 if possible)
* default port: `8000`

The script also attempts to **detect the correct port** by scanning each `load_*.cypher` file for:

* `http://localhost:<PORT>/...`
* `http://127.0.0.1:<PORT>/...`

If a port is detected, the HTTP server is restarted on that port automatically.

### Execution of `.cypher` scripts

Each `load_*.cypher` file is parsed into individual statements and executed sequentially.
Neo4j Browser directives (lines beginning with `:`) are ignored.

---

## Workloads (query-time overhead)

The benchmark can run a query suite intended to reflect increasing metadata demands:

* **WQ-A**: baseline graph access patterns (label-as-property, 1-hop neighborhood, city selection)
* **WQ-B**: fact-level metadata and provenance
* **WQ-C**: property-occurrence metadata (population occurrences hooking metadata on `:PropEdge`)
* **WQ-D**: component-level metadata over structured values (only for `gpgc`)

### Workload selection by variant

The workload suite is automatically adjusted:

* `gpg`: runs A + B using native relationships and node properties
* `gpge`: runs A + B using reified `EdgeFact`
* `gpgp`: runs A + B + C
* `gpgc`: runs A + B + C + D

This matters for aggregation:

* `q_total_time_s` sums **all** executed workloads for a given encoding
* `q_total_ab_time_s` sums only the **common subset** (A + B), used for fair cross-variant comparison

Full Cypher listings and conventions are documented in **`online_appendix.md`**.

---

## Output: results CSV

Benchmark results are stored in:

* **`neo4j_load_benchmark_results.csv`**

The CSV schema is stable and includes:

### General load measurements

* `timestamp` — ISO timestamp (ms resolution)
* `dataset` — `D1-S`, `D2-M`, `D3-L`
* `variant` — `gpg`, `gpge`, `gpgp`, `gpgc`
* `load_script` — executed file (e.g., `load_D1-S_gpg.cypher`)
* `http_base_url` — e.g. `http://localhost:8000`
* `load_time_s` — total load execution time (seconds)
* `nodes`, `rels` — resulting graph counts
* `db_store_bytes`, `db_tx_bytes`, `db_total_bytes`, `db_total_human` — disk usage breakdown
* `neo4j_version`, `apoc_version`

### Query-time overhead measurements

* `q_WQ_*_s` — average time per workload query
* `q_total_time_s` — sum of all executed workload averages
* `q_total_ab_time_s` — sum of A + B workloads only (common subset)

### Overhead vs baseline (per dataset)

To compare variants fairly, the script computes overhead relative to `gpg` **for A + B only**:

* `q_total_ab_vs_gpg_ratio`
* `q_total_ab_vs_gpg_pct`

These are computed after establishing the baseline mean:

* baseline = mean(`q_total_ab_time_s`) for `gpg` repeated runs within the same dataset

---

## Reproducibility notes

* The dataset generator is deterministic given `--seed`.
* The benchmark runner performs:

  * constraint/index removal,
  * batched deletion of all nodes,
  * a `db.checkpoint()` attempt (best-effort),
  * then import and measurement.

To reduce noise:

* avoid other workloads during benchmarking,
* pin Neo4j memory settings for repeatability (heap, page cache),
* ensure datasets are served locally and filesystem caches behave consistently.

---

## Troubleshooting

### 1) `Neo.DatabaseError.Statement.ExecutionFailed: Java heap space`

This indicates insufficient heap for the import or queries.

Typical mitigations:

* increase `server.memory.heap.initial_size` and `server.memory.heap.max_size`,
* ensure `server.memory.pagecache.size` is reasonable for the dataset size,
* reduce dataset scale (e.g., test with `D1-S` first),
* run fewer repetitions (`--reps 1`) during debugging.

### 2) APOC procedures not found

Verify:

* APOC plugin is installed,
* APOC is enabled in configuration,
* the Neo4j instance is restarted after installation.

### 3) Import cannot fetch CSV via HTTP

Verify that the HTTP server is running and accessible:

* open a dataset file in the browser, e.g.
  `http://localhost:8000/D1-S/gpg/nodes_entities.csv`

Also verify that the port in `load_*.cypher` matches the server port.

### 4) Disk size is reported as zero

This usually means `DBMS_DIR` is incorrect for your Neo4j installation.
Set it to the correct Neo4j DBMS directory (Neo4j Desktop stores it under `relate-data/dbmss/...`).



