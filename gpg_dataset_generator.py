#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic dataset generator for the Experimental Study in "Metadata in Property Graphs"
(D1-S, D2-M, D3-L) and four encodings: gpg, gpge, gpgp, gpgc.

What it generates (per dataset + variant):
- nodes_entities.csv  (label: Entity)
- nodes_sources.csv   (label: Source)  -- fixed catalog of nS=100 sources
- gpg:
    - rels_rel.csv     (type: REL)      -- logical relationship types are carried in r.k_noval
- gpge / gpgp / gpgc:
    - nodes_edgefacts.csv (label: EdgeFact)
    - rels_src.csv        (type: SRC)
    - rels_tgt.csv        (type: TGT)
    - rels_has_source.csv (type: HAS_SOURCE)
- gpgp / gpgc:
    - nodes_propedges.csv (label: PropEdge)
    - nodes_values*.csv   (label: Value)  -- see --values-mode
- gpgc:
    - nodes_compedges.csv (label: CompEdge)

Values mode:
- split  (default): creates
    - nodes_values_num.csv (val:long) for numeric population values
    - nodes_values_str.csv (val:string) for JSON / address-field strings
  and an import.apoc.cypher with both files included.
- single: creates
    - nodes_values.csv (val defaults to string in APOC import header format)
  This is simplest, but then population values are strings; numeric comparisons in Cypher
  must cast (e.g., toInteger(v.val)) or use padded strings.

The generator is deterministic given --seed.

Tested assumptions:
- Neo4j 4.4.42 + APOC 4.4.x
- Loading via `CALL apoc.import.csv(...)` from localhost HTTP server.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

# ----------------------------
# Dataset configuration (Table 1)
# ----------------------------

@dataclass(frozen=True)
class DatasetCfg:
    name: str
    nV: int        # entity vertices (persons + cities)
    nE: int        # logical relationship facts (LIVES_IN + KNOWS)
    m: int         # population occurrences per city
    q: int         # metadata attributes per annotated item
    c: int         # address fields (components)
    faddr: float   # fraction of persons with address (structured value)

DATASETS: Dict[str, DatasetCfg] = {
    "D1-S": DatasetCfg("D1-S", nV=10_000,  nE=40_000,    m=3,  q=2, c=2, faddr=0.60),
    "D2-M": DatasetCfg("D2-M", nV=50_000,  nE=250_000,   m=10, q=3, c=4, faddr=0.80),
    "D3-L": DatasetCfg("D3-L", nV=200_000, nE=1_200_000, m=30, q=4, c=6, faddr=0.90),
}

VARIANTS = ("gpg", "gpge", "gpgp", "gpgc")

N_SOURCES = 100

# ----------------------------
# Deterministic pseudo-randomness (no global RNG state)
# ----------------------------

MASK64 = (1 << 64) - 1

def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    z = x
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & MASK64
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & MASK64
    z = z ^ (z >> 31)
    return z & MASK64

def h64(seed: int, *vals: int) -> int:
    x = seed & MASK64
    for v in vals:
        x = _splitmix64(x ^ (v & MASK64))
    return x

def u32(seed: int, *vals: int) -> int:
    return h64(seed, *vals) & 0xFFFFFFFF

def choose(seed: int, n: int, *vals: int) -> int:
    # n must be > 0
    return u32(seed, *vals) % n

def rand01(seed: int, *vals: int) -> float:
    return u32(seed, *vals) / 2**32

# ----------------------------
# ID helpers
# ----------------------------

def pid(i: int) -> str:  # Person ID
    return f"P{i:07d}"

def cid(i: int) -> str:  # City ID
    return f"C{i:07d}"

def sid(i: int) -> str:  # Source ID
    return f"S{i:03d}"

def efid(i: int) -> str:  # EdgeFact ID
    return f"EF{i:09d}"

def peid(i: int) -> str:  # PropEdge ID
    return f"PE{i:09d}"

def ceid(i: int) -> str:  # CompEdge ID
    return f"CE{i:09d}"

def vid(i: int) -> str:  # Value ID
    return f"V{i:09d}"

# ----------------------------
# CSV writing
# ----------------------------

def write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        w.writerow(list(header))
        for r in rows:
            w.writerow(list(r))

# ----------------------------
# Logical generator primitives
# ----------------------------

def split_person_city(nV: int) -> Tuple[int, int]:
    nP = (7 * nV) // 10      # 70%
    nC = nV - nP             # 30%
    return nP, nC

def person_has_address(cfg: DatasetCfg, seed: int, person_idx: int) -> bool:
    # Deterministic Bernoulli, no stored sets.
    return rand01(seed, 700_001, person_idx) < cfg.faddr

def assigned_city(cfg: DatasetCfg, seed: int, person_idx: int, nC: int) -> int:
    # Deterministic city assignment for LIVES_IN
    return choose(seed, nC, 700_101, person_idx)

def knows_edges(cfg: DatasetCfg, seed: int, nP: int) -> Iterator[Tuple[int, int, int]]:
    """
    Yields KNOWS edges as (fact_idx, src_person_idx, tgt_person_idx),
    with fact_idx ranging over [nP, cfg.nE).
    Construction avoids self-loops and (practically) duplicates without storing a set:
    we generate edges in passes; each pass uses a distinct stride.
    """
    n_total = cfg.nE - nP
    if n_total <= 0:
        return
        yield  # pragma: no cover

    # number of passes over all persons needed
    passes = (n_total + nP - 1) // nP
    # distinct strides in [1, nP-1]
    used = set()
    strides: List[int] = []
    for p in range(passes):
        # pick a new stride deterministically; re-pick if collision
        s = 1 + choose(seed, nP - 1, 700_201, p)
        while s in used:
            s = 1 + ((s + 1) % (nP - 1))
        used.add(s)
        strides.append(s)

    out = 0
    base_fact = nP
    for p, stride in enumerate(strides):
        for u in range(nP):
            if out >= n_total:
                return
            v = (u + stride) % nP
            fact_idx = base_fact + out
            yield (fact_idx, u, v)
            out += 1

# ----------------------------
# Metadata generators (deterministic)
# ----------------------------

def edge_since_year(seed: int, fact_idx: int) -> int:
    return 1990 + choose(seed, 36, 710_001, fact_idx)  # 1990..2025

def edge_source_idx(seed: int, fact_idx: int) -> int:
    return choose(seed, N_SOURCES, 710_101, fact_idx)

def pop_point_in_time(seed: int, city_idx: int, occ_idx: int) -> int:
    return 1990 + choose(seed, 36, 720_001, city_idx, occ_idx)

def pop_confidence(seed: int, city_idx: int, occ_idx: int) -> float:
    # 0.50 .. 1.00
    return 0.50 + (choose(seed, 501, 720_101, city_idx, occ_idx) / 1000.0)

def extra_int(seed: int, tag: int, *vals: int) -> int:
    # generic "extra" metadata field
    return choose(seed, 1_000_000, tag, *vals)

def curator(seed: int, person_idx: int, field_idx: int) -> str:
    return f"curator_{choose(seed, 500, 730_001, person_idx, field_idx):03d}"

def last_updated(seed: int, person_idx: int, field_idx: int) -> date:
    base = date(2018, 1, 1)
    # up to ~8 years range
    return base + timedelta(days=choose(seed, 8 * 365, 730_101, person_idx, field_idx))

# ----------------------------
# Address generation
# ----------------------------

ADDRESS_FIELDS = ["zip", "city", "street", "number", "country", "region"]
COUNTRIES = ["PL", "DE", "FR", "ES", "IT", "NL", "SE", "NO"]
REGIONS = ["north", "south", "east", "west", "central"]

def address_record(cfg: DatasetCfg, seed: int, person_idx: int, city_idx: int) -> Dict[str, str]:
    fields = ADDRESS_FIELDS[: cfg.c]
    rec: Dict[str, str] = {}
    for fi, f in enumerate(fields):
        if f == "zip":
            # keep canonical 5 digits, optionally with dash
            z = 10_000 + choose(seed, 90_000, 740_001, person_idx, fi)
            rec[f] = f"{z:05d}"
        elif f == "city":
            rec[f] = f"City_{city_idx:07d}"
        elif f == "street":
            rec[f] = f"Street_{choose(seed, 50_000, 740_101, person_idx, fi):05d}"
        elif f == "number":
            rec[f] = str(1 + choose(seed, 300, 740_201, person_idx, fi))
        elif f == "country":
            rec[f] = COUNTRIES[choose(seed, len(COUNTRIES), 740_301, person_idx, fi)]
        elif f == "region":
            rec[f] = REGIONS[choose(seed, len(REGIONS), 740_401, person_idx, fi)]
        else:
            rec[f] = f"v_{choose(seed, 1_000_000, 740_501, person_idx, fi)}"
    return rec

# ----------------------------
# Variant generators
# ----------------------------

def gen_entities_rows_gpg_like(cfg: DatasetCfg, seed: int, variant: str) -> Iterator[List[object]]:
    """
    gpg / gpge: entities contain population as a value-set property on City nodes.
    For other variants, City population occurrences are reified, so we omit population on City nodes.
    """
    nP, nC = split_person_city(cfg.nV)

    with_population = variant in ("gpg", "gpge")
    with_address_prop = variant in ("gpg", "gpge", "gpgp")  # in gpgc address is reified

    # persons
    for i in range(nP):
        row: List[object] = [pid(i), "Person"]
        if with_population:
            row.append("")  # population only on cities
        if with_address_prop:
            if person_has_address(cfg, seed, i):
                city_idx = assigned_city(cfg, seed, i, nC)
                rec = address_record(cfg, seed, i, city_idx)
                row.append(json.dumps(rec, separators=(",", ":"), ensure_ascii=False))
            else:
                row.append("")
        yield row

    # cities
    for j in range(nC):
        row = [cid(j), "City"]
        if with_population:
            pops = []
            base = 50_000 + choose(seed, 5_000_000, 750_001, j)
            for occ in range(cfg.m):
                delta = choose(seed, 200_000, 750_101, j, occ)
                pops.append(base + delta)
            # array delimiter is ';' (as in the APOC config)
            row.append(";".join(str(x) for x in pops))
        if with_address_prop:
            row.append("")  # address only on persons
        yield row

def entities_header(variant: str) -> List[str]:
    # id:ID is required by APOC import header format
    if variant in ("gpg", "gpge"):
        return ["id:ID", "k_noval:string[]", "population:long[]", "address:string"]
    if variant == "gpgp":
        return ["id:ID", "k_noval:string[]", "address:string"]
    # gpgc
    return ["id:ID", "k_noval:string[]"]

def gen_sources_rows() -> Iterator[List[object]]:
    for i in range(N_SOURCES):
        s = sid(i)
        yield [s, f"Source_{i:03d}", f"https://example.org/source/{s}"]

def sources_header() -> List[str]:
    return ["id:ID", "name:string", "url:string"]

def gen_rels_gpg(cfg: DatasetCfg, seed: int) -> Iterator[List[object]]:
    """
    Baseline relationships as :REL, logical types carried in r.k_noval.
    Each person has exactly one LIVES_IN, the remainder are KNOWS.
    Metadata on KNOWS: m_since + m_sourceId (+ extras to match q).
    """
    nP, nC = split_person_city(cfg.nV)

    extra_cols = max(0, cfg.q - 2)

    # LIVES_IN facts: exactly nP (one per person), no metadata
    for i in range(nP):
        j = assigned_city(cfg, seed, i, nC)
        yield [pid(i), cid(j), "LIVES_IN", "", "", *["" for _ in range(extra_cols)]]

    # KNOWS facts: cfg.nE - nP
    for fact_idx, u, v in knows_edges(cfg, seed, nP):
        since = edge_since_year(seed, fact_idx)
        src = sid(edge_source_idx(seed, fact_idx))
        extras = [extra_int(seed, 760_001 + k, fact_idx) for k in range(extra_cols)]
        yield [pid(u), pid(v), "KNOWS", since, src, *extras]

def rels_gpg_header(cfg: DatasetCfg) -> List[str]:
    # k_noval is an array, but we store singletons; arrayDelimiter controls parsing
    extra_cols = max(0, cfg.q - 2)
    header = [":START_ID", ":END_ID", "k_noval:string[]", "m_since:int", "m_sourceId:string"]
    for k in range(extra_cols):
        header.append(f"m_extra{k+1}:int")
    return header

def gen_edgefacts_rows(cfg: DatasetCfg, seed: int) -> Iterator[List[object]]:
    """
    EdgeFact nodes for all logical relationship facts:
    - first nP facts are LIVES_IN (one per person)
    - remaining are KNOWS
    Metadata on KNOWS: m_since (+ extras to match q); provenance is via HAS_SOURCE.
    """
    nP, nC = split_person_city(cfg.nV)
    extra_cols = max(0, cfg.q - 2)

    # LIVES_IN facts: fact_idx = 0..nP-1
    for i in range(nP):
        yield [efid(i), "LIVES_IN", "", *["" for _ in range(extra_cols)]]

    # KNOWS facts: fact_idx = nP..cfg.nE-1
    for fact_idx, _, _ in knows_edges(cfg, seed, nP):
        since = edge_since_year(seed, fact_idx)
        extras = [extra_int(seed, 770_001 + k, fact_idx) for k in range(extra_cols)]
        yield [efid(fact_idx), "KNOWS", since, *extras]

def edgefacts_header(cfg: DatasetCfg) -> List[str]:
    extra_cols = max(0, cfg.q - 2)
    header = ["id:ID", "k_noval:string[]", "m_since:int"]
    for k in range(extra_cols):
        header.append(f"m_extra{k+1}:int")
    return header

def gen_rels_src_tgt_has_source(cfg: DatasetCfg, seed: int) -> Tuple[Iterator[List[object]], Iterator[List[object]], Iterator[List[object]]]:
    """
    SRC/TGT connect EdgeFacts to endpoints; HAS_SOURCE only for KNOWS (as per study description).
    """
    nP, nC = split_person_city(cfg.nV)

    def src_rows() -> Iterator[List[object]]:
        # LIVES_IN: source = person
        for i in range(nP):
            yield [efid(i), pid(i)]
        # KNOWS: source = person u
        for fact_idx, u, _ in knows_edges(cfg, seed, nP):
            yield [efid(fact_idx), pid(u)]

    def tgt_rows() -> Iterator[List[object]]:
        # LIVES_IN: target = assigned city
        for i in range(nP):
            j = assigned_city(cfg, seed, i, nC)
            yield [efid(i), cid(j)]
        # KNOWS: target = person v
        for fact_idx, _, v in knows_edges(cfg, seed, nP):
            yield [efid(fact_idx), pid(v)]

    def has_source_rows() -> Iterator[List[object]]:
        # Only KNOWS carry provenance source
        for fact_idx, _, _ in knows_edges(cfg, seed, nP):
            yield [efid(fact_idx), sid(edge_source_idx(seed, fact_idx))]

    return src_rows(), tgt_rows(), has_source_rows()

def rels_simple_header() -> List[str]:
    return [":START_ID", ":END_ID"]

# ---- Property occurrence reification (population) ----

def gen_population_propedges_and_values(cfg: DatasetCfg, seed: int, values_mode: str,
                                       pe_start: int, v_start: int) -> Tuple[Iterator[List[object]], Dict[str, Iterator[List[object]]], int, int]:
    """
    Returns:
      - PropEdge rows iterator (population occurrences only)
      - Value rows iterators dict, keyed by filename (depends on values_mode)
      - next PropEdge counter
      - next Value counter
    """
    _, nC = split_person_city(cfg.nV)
    extra_cols = max(0, cfg.q - 2)

    def propedge_rows() -> Iterator[List[object]]:
        nonlocal pe_start, v_start
        for city_idx in range(nC):
            base = 50_000 + choose(seed, 5_000_000, 780_001, city_idx)
            for occ in range(cfg.m):
                pe = peid(pe_start); pe_start += 1
                vv = vid(v_start); v_start += 1
                point = pop_point_in_time(seed, city_idx, occ)
                conf = pop_confidence(seed, city_idx, occ)
                extras = [extra_int(seed, 780_101 + k, city_idx, occ) for k in range(extra_cols)]
                # PropEdge node row
                yield [pe, "population", point, conf, *extras]
                # SRC/TGT relationships for this PropEdge are produced elsewhere, using (pe, city, vv)

    # For values, we stream and re-use the same counters/order as above
    # so that PropEdge->Value links can be emitted deterministically in the same traversal.
    def value_rows_num() -> Iterator[List[object]]:
        # numeric values for population
        nonlocal v_start
        # Reset local counter by recomputing in same order; we must mirror the allocation.
        # Instead, we generate values in a dedicated traversal with its own counter base.
        # We'll do that in the caller to avoid inconsistencies.
        raise RuntimeError("Internal: value_rows_num is built in caller")

    # We can't safely generate PropEdges and Values in separate passes unless we re-run the same
    # counter logic. We therefore also provide a helper that returns *materialized* link plan in caller.
    # The caller will generate values while generating SRC/TGT rels.
    value_iters: Dict[str, Iterator[List[object]]] = {}
    # placeholder; actual value rows are generated in caller
    return propedge_rows(), value_iters, pe_start, v_start

def propedges_header(cfg: DatasetCfg) -> List[str]:
    extra_cols = max(0, cfg.q - 2)
    header = ["id:ID", "pkey:string", "m_point_in_time:int", "m_confidence:float"]
    for k in range(extra_cols):
        header.append(f"m_extra{k+1}:int")
    return header

# ---- Component edges for structured address (gpgc) ----

def compedges_header(cfg: DatasetCfg) -> List[str]:
    extra_cols = max(0, cfg.q - 2)
    header = ["id:ID", "selKind:string", "selVal:string", "m_curator:string", "m_lastUpdated:date"]
    for k in range(extra_cols):
        header.append(f"m_extra{k+1}:int")
    return header

# ----------------------------
# Main generation orchestrator per variant
# ----------------------------

def generate_variant(cfg: DatasetCfg, out_root: Path, seed: int, variant: str, values_mode: str) -> None:
    nP, nC = split_person_city(cfg.nV)

    vdir = out_root / cfg.name / variant
    vdir.mkdir(parents=True, exist_ok=True)

    # ---- Entities & Sources (always) ----
    write_csv(vdir / "nodes_entities.csv", entities_header(variant), gen_entities_rows_gpg_like(cfg, seed, variant))
    write_csv(vdir / "nodes_sources.csv", sources_header(), gen_sources_rows())

    # ---- gpg baseline ----
    if variant == "gpg":
        write_csv(vdir / "rels_rel.csv", rels_gpg_header(cfg), gen_rels_gpg(cfg, seed))
        write_import_file(cfg, vdir, variant, values_mode=values_mode)
        return

    # ---- gpge/gpgp/gpgc: EdgeFacts + SRC/TGT + HAS_SOURCE ----
    write_csv(vdir / "nodes_edgefacts.csv", edgefacts_header(cfg), gen_edgefacts_rows(cfg, seed))
    src_rows, tgt_rows, hs_rows = gen_rels_src_tgt_has_source(cfg, seed)
    write_csv(vdir / "rels_src.csv", rels_simple_header(), src_rows)
    write_csv(vdir / "rels_tgt.csv", rels_simple_header(), tgt_rows)
    write_csv(vdir / "rels_has_source.csv", rels_simple_header(), hs_rows)

    # ---- gpgp/gpgc: population property occurrences as PropEdge + Value ----
    if variant in ("gpgp", "gpgc"):
        extra_cols = max(0, cfg.q - 2)

        # We'll generate PropEdges, their SRC/TGT rels, and Value nodes together in one streaming pass,
        # to avoid holding large structures and to keep counters consistent.
        pe_counter = 0
        v_counter = 0

        propedges_path = vdir / "nodes_propedges.csv"
        rels_src_path = vdir / "rels_src.csv"
        rels_tgt_path = vdir / "rels_tgt.csv"

        # Values files
        if values_mode == "split":
            values_num_path = vdir / "nodes_values_num.csv"
            values_str_path = vdir / "nodes_values_str.csv"
            values_single_path = None
        else:
            values_num_path = None
            values_str_path = None
            values_single_path = vdir / "nodes_values.csv"

        # Prepare writers (append to rels_src/tgt, which already contain EdgeFact SRC/TGT rows)
        # We will append by opening in append mode and not rewriting headers.
        # For simplicity and robustness, we rewrite SRC/TGT from scratch including EdgeFact rows.
        # This keeps the script stateless and avoids subtle append ordering issues.
        src_rows_edgefacts, tgt_rows_edgefacts, hs_rows_edgefacts = gen_rels_src_tgt_has_source(cfg, seed)

        def all_src_rows() -> Iterator[List[object]]:
            # EdgeFact SRC rows
            for r in src_rows_edgefacts:
                yield r
            # PropEdge SRC rows (population)
            nonlocal pe_counter, v_counter
            for city_idx in range(nC):
                for occ in range(cfg.m):
                    pe = peid(pe_counter); pe_counter += 1
                    vv = vid(v_counter); v_counter += 1
                    yield [pe, cid(city_idx)]
            # address PropEdge SRC rows (gpgc) are generated later in gpgc block

        # reset counters for parallel traversal
        pe_counter_src = 0
        v_counter_src = 0

        def all_tgt_rows() -> Iterator[List[object]]:
            for r in tgt_rows_edgefacts:
                yield r
            nonlocal pe_counter_src, v_counter_src
            for city_idx in range(nC):
                for occ in range(cfg.m):
                    pe = peid(pe_counter_src); pe_counter_src += 1
                    vv = vid(v_counter_src); v_counter_src += 1
                    yield [pe, vv]

        # reset counters again for PropEdge node generation + Value node generation
        pe_counter_nodes = 0
        v_counter_nodes = 0

        def propedge_rows_population() -> Iterator[List[object]]:
            nonlocal pe_counter_nodes, v_counter_nodes
            for city_idx in range(nC):
                for occ in range(cfg.m):
                    pe = peid(pe_counter_nodes); pe_counter_nodes += 1
                    _vv = vid(v_counter_nodes); v_counter_nodes += 1
                    point = pop_point_in_time(seed, city_idx, occ)
                    conf = pop_confidence(seed, city_idx, occ)
                    extras = [extra_int(seed, 780_101 + k, city_idx, occ) for k in range(extra_cols)]
                    yield [pe, "population", point, conf, *extras]

        # Values (population)
        v_counter_vals = 0
        def value_rows_population_num() -> Iterator[List[object]]:
            nonlocal v_counter_vals
            for city_idx in range(nC):
                base = 50_000 + choose(seed, 5_000_000, 780_001, city_idx)
                for occ in range(cfg.m):
                    vv = vid(v_counter_vals); v_counter_vals += 1
                    delta = choose(seed, 200_000, 780_201, city_idx, occ)
                    popv = base + delta
                    if values_mode == "split":
                        yield [vv, popv]
                    else:
                        yield [vv, str(popv)]

        # Now write population PropEdges and Values
        write_csv(propedges_path, propedges_header(cfg), propedge_rows_population())

        # Re-write SRC/TGT including EdgeFacts + population PropEdges (and later gpgc address)
        write_csv(rels_src_path, rels_simple_header(), all_src_rows())
        write_csv(rels_tgt_path, rels_simple_header(), all_tgt_rows())

        if values_mode == "split":
            write_csv(values_num_path, ["id:ID", "val:long"], value_rows_population_num())
            # We'll create values_str later (gpgc); create empty placeholder for gpgp
            if variant == "gpgp":
                write_csv(values_str_path, ["id:ID", "val:string"], [])
        else:
            write_csv(values_single_path, ["id:ID", "val"], value_rows_population_num())

        # ---- gpgc adds structured addresses + component edges ----
        if variant == "gpgc":
            # We must append address PropEdges/Values/CompEdges and also extend SRC/TGT and Values.
            # Since we already wrote SRC/TGT and (some) Values, we'll regenerate *everything*
            # (EdgeFacts + population + addresses) in one pass and overwrite the affected files.
            # This is still linear time and avoids fragile appends.

            # Counters restart; Value IDs remain stable for population as V000000000.. in order,
            # then addresses continue after.
            pe_counter = 0
            ce_counter = 0
            v_counter = 0

            # We will stream out:
            # - PropEdges: population first, then address propedges
            # - CompEdges: all component edges
            # - SRC/TGT rels: EdgeFacts first, then PropEdge/CompEdge
            # - Values: population numeric first, then address strings (root + fields)

            def propedges_all() -> Iterator[List[object]]:
                nonlocal pe_counter, v_counter
                # population
                for city_idx in range(nC):
                    for occ in range(cfg.m):
                        pe = peid(pe_counter); pe_counter += 1
                        _vv = vid(v_counter); v_counter += 1
                        point = pop_point_in_time(seed, city_idx, occ)
                        conf = pop_confidence(seed, city_idx, occ)
                        extras = [extra_int(seed, 790_101 + k, city_idx, occ) for k in range(extra_cols)]
                        yield [pe, "population", point, conf, *extras]
                # address (one PropEdge per addressed person)
                for person_idx in range(nP):
                    if not person_has_address(cfg, seed, person_idx):
                        continue
                    pe = peid(pe_counter); pe_counter += 1
                    _root = vid(v_counter); v_counter += 1
                    # dummy values for numeric meta fields, to keep schema uniform
                    point = pop_point_in_time(seed, person_idx, 0)
                    conf = 1.0
                    extras = [extra_int(seed, 791_001 + k, person_idx, 0) for k in range(extra_cols)]
                    yield [pe, "address", point, conf, *extras]

            def compedges_all() -> Iterator[List[object]]:
                nonlocal ce_counter, v_counter
                # values for components are allocated while generating values below; but CompEdges need IDs now.
                # We'll allocate component Value IDs here as well for stable linking:
                # after each address root Value, we allocate cfg.c field Value nodes.
                # Therefore, compedge generation must mirror value allocation order.
                # We thus generate CompEdges in the same address loop used for values.
                # But we need v_counter positioned after population values:
                base_v = nC * cfg.m  # population values count
                local_v = base_v

                for person_idx in range(nP):
                    if not person_has_address(cfg, seed, person_idx):
                        continue
                    root = vid(local_v); local_v += 1
                    # field values
                    for fi, field in enumerate(ADDRESS_FIELDS[: cfg.c]):
                        field_v = vid(local_v); local_v += 1
                        ce = ceid(ce_counter); ce_counter += 1
                        cur = curator(seed, person_idx, fi)
                        lu = last_updated(seed, person_idx, fi).isoformat()
                        extras = [extra_int(seed, 792_001 + k, person_idx, fi) for k in range(extra_cols)]
                        yield [ce, "field", field, cur, lu, *extras]

            def rels_src_all() -> Iterator[List[object]]:
                # EdgeFacts SRC first
                src_e, _, _ = gen_rels_src_tgt_has_source(cfg, seed)
                for r in src_e:
                    yield r

                # PropEdges (population + address)
                # population: pe0.., src is city
                pe_local = 0
                v_local = 0
                for city_idx in range(nC):
                    for _occ in range(cfg.m):
                        pe = peid(pe_local); pe_local += 1
                        _vv = vid(v_local); v_local += 1
                        yield [pe, cid(city_idx)]

                # address: continued PropEdge ids, src is person
                for person_idx in range(nP):
                    if not person_has_address(cfg, seed, person_idx):
                        continue
                    pe = peid(pe_local); pe_local += 1
                    _root = vid(v_local); v_local += 1
                    yield [pe, pid(person_idx)]
                    # component edges: SRC from CompEdge to root
                    root = vid(v_local - 1)  # the last allocated root
                    for fi in range(cfg.c):
                        # component Value IDs follow root
                        _field_v = vid(v_local); v_local += 1
                        ce = ceid((person_idx * cfg.c) + fi)  # placeholder; corrected below

                # Now component-edge SRC rows, using dedicated counters
                ce_local = 0
                v_base = nC * cfg.m
                v_local2 = v_base
                for person_idx in range(nP):
                    if not person_has_address(cfg, seed, person_idx):
                        continue
                    root = vid(v_local2); v_local2 += 1
                    for _fi in range(cfg.c):
                        _field_v = vid(v_local2); v_local2 += 1
                        ce = ceid(ce_local); ce_local += 1
                        yield [ce, root]

            def rels_tgt_all() -> Iterator[List[object]]:
                # EdgeFacts TGT first
                _, tgt_e, _ = gen_rels_src_tgt_has_source(cfg, seed)
                for r in tgt_e:
                    yield r

                # PropEdges (population + address)
                pe_local = 0
                v_local = 0
                for _city_idx in range(nC):
                    for _occ in range(cfg.m):
                        pe = peid(pe_local); pe_local += 1
                        vv = vid(v_local); v_local += 1
                        yield [pe, vv]

                for person_idx in range(nP):
                    if not person_has_address(cfg, seed, person_idx):
                        continue
                    pe = peid(pe_local); pe_local += 1
                    root = vid(v_local); v_local += 1
                    yield [pe, root]
                    # allocate component field values
                    for _fi in range(cfg.c):
                        _field_v = vid(v_local); v_local += 1

                # CompEdges TGT rows: to field values
                ce_local = 0
                v_base = nC * cfg.m
                v_local2 = v_base
                for person_idx in range(nP):
                    if not person_has_address(cfg, seed, person_idx):
                        continue
                    _root = vid(v_local2); v_local2 += 1
                    for _fi in range(cfg.c):
                        field_v = vid(v_local2); v_local2 += 1
                        ce = ceid(ce_local); ce_local += 1
                        yield [ce, field_v]

            def values_num_rows() -> Iterator[List[object]]:
                # population numeric values only
                v_local = 0
                for city_idx in range(nC):
                    base = 50_000 + choose(seed, 5_000_000, 780_001, city_idx)
                    for occ in range(cfg.m):
                        vv = vid(v_local); v_local += 1
                        delta = choose(seed, 200_000, 780_201, city_idx, occ)
                        popv = base + delta
                        yield [vv, popv]

            def values_str_rows() -> Iterator[List[object]]:
                # address root JSON + field values (strings)
                v_base = nC * cfg.m
                v_local = v_base
                for person_idx in range(nP):
                    if not person_has_address(cfg, seed, person_idx):
                        continue
                    city_idx = assigned_city(cfg, seed, person_idx, nC)
                    rec = address_record(cfg, seed, person_idx, city_idx)
                    root = vid(v_local); v_local += 1
                    yield [root, json.dumps(rec, separators=(",", ":"), ensure_ascii=False)]
                    # fields
                    for f in ADDRESS_FIELDS[: cfg.c]:
                        vv = vid(v_local); v_local += 1
                        yield [vv, rec.get(f, "")]

            # overwrite with complete gpgc content
            write_csv(propedges_path, propedges_header(cfg), propedges_all())
            write_csv(vdir / "nodes_compedges.csv", compedges_header(cfg), compedges_all())
            write_csv(rels_src_path, rels_simple_header(), rels_src_all())
            write_csv(rels_tgt_path, rels_simple_header(), rels_tgt_all())

            if values_mode == "split":
                write_csv(values_num_path, ["id:ID", "val:long"], values_num_rows())
                write_csv(values_str_path, ["id:ID", "val:string"], values_str_rows())
            else:
                # single: everything as string
                def values_single() -> Iterator[List[object]]:
                    for r in values_num_rows():
                        yield [r[0], str(r[1])]
                    for r in values_str_rows():
                        yield r
                write_csv(values_single_path, ["id:ID", "val"], values_single())

    write_import_file(cfg, vdir, variant, values_mode=values_mode)

def write_import_file(cfg: DatasetCfg, vdir: Path, variant: str, values_mode: str) -> None:
    """
    Writes import.apoc.cypher with the corresponding apoc.import.csv call.
    Assumes a local HTTP server serves the OUT root at http://localhost:8000/
    """
    dataset = cfg.name
    base_url = f"http://localhost:8000/{dataset}/{variant}/"

    nodes = [
        f"    {{fileName:'{base_url}nodes_entities.csv',  labels:['Entity']}}",
        f"    {{fileName:'{base_url}nodes_sources.csv',   labels:['Source']}}",
    ]
    rels = []

    if variant == "gpg":
        rels.append(f"    {{fileName:'{base_url}rels_rel.csv',        type:'REL'}}")
    else:
        nodes.append(f"    {{fileName:'{base_url}nodes_edgefacts.csv', labels:['EdgeFact']}}")
        rels.extend([
            f"    {{fileName:'{base_url}rels_src.csv',        type:'SRC'}}",
            f"    {{fileName:'{base_url}rels_tgt.csv',        type:'TGT'}}",
            f"    {{fileName:'{base_url}rels_has_source.csv', type:'HAS_SOURCE'}}",
        ])
        if variant in ("gpgp", "gpgc"):
            nodes.append(f"    {{fileName:'{base_url}nodes_propedges.csv', labels:['PropEdge']}}")
            if variant == "gpgc":
                nodes.append(f"    {{fileName:'{base_url}nodes_compedges.csv', labels:['CompEdge']}}")
            if values_mode == "split":
                nodes.append(f"    {{fileName:'{base_url}nodes_values_num.csv', labels:['Value']}}")
                nodes.append(f"    {{fileName:'{base_url}nodes_values_str.csv', labels:['Value']}}")
            else:
                nodes.append(f"    {{fileName:'{base_url}nodes_values.csv', labels:['Value']}}")

    cypher = [
        "// Auto-generated APOC import call",
        f"// Dataset: {dataset}",
        f"// Variant: {variant}",
        "",
        "CALL apoc.import.csv(",
        "  [",
        ",\n".join(nodes),
        "  ],",
        "  [",
        ",\n".join(rels),
        "  ],",
        "  {delimiter:',', arrayDelimiter:';', stringIds:true,",
        "   batchSize:50000, ignoreDuplicateNodes:true, ignoreDuplicateRelationships:true,",
        "   ignoreEmptyCellArray:true, ignoreBlankString:true}",
        ");",
        "",
    ]
    (vdir / "import.apoc.cypher").write_text("\n".join(cypher), encoding="utf-8")

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic D1/D2/D3 datasets for gpg/gpge/gpgp/gpgc (Neo4j APOC import.csv)."
    )
    parser.add_argument("--out", required=True, help="Output root directory, e.g., ./datasets")
    parser.add_argument("--seed", type=int, default=1, help="Deterministic seed (default: 1)")
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS.keys()), choices=list(DATASETS.keys()),
                        help="Which datasets to generate (default: all)")
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS), choices=list(VARIANTS),
                        help="Which variants to generate (default: all)")
    parser.add_argument("--values-mode", choices=["split", "single"], default="split",
                        help="How to store Value nodes for gpgp/gpgc (default: split)")
    args = parser.parse_args(argv)

    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    for dname in args.datasets:
        cfg = DATASETS[dname]
        for variant in args.variants:
            generate_variant(cfg, out_root, args.seed, variant, values_mode=args.values_mode)

    print(f"Done. Output in: {out_root}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
