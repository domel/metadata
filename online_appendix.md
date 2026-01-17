# Online Appendix — Experimental Study (Neo4j 5.26)

This appendix consolidates the Cypher listings and the data-loading workflow used in the experimental study:
- **Workloads (WQ-A–WQ-D)**: complete query listings.
- **Data loading**: localhost HTTP serving, `apoc.import.csv` calls, and required constraints/indexes.

---

## Conventions and parameters

All workload queries follow the paper’s *semantic* style: logical labels and relationship types are tested via
membership in `k_noval`. Parameters are supplied through the driver layer (or via `:param` in Neo4j Browser).

### Parameters (by convention)

- `$pid`: person/entity identifier
- `$cid`: city/entity identifier
- `$pop`: population threshold
- `$since`: temporal threshold (e.g., year or date value, consistent with dataset encoding)
- `$conf`: confidence threshold
- `$k`: integer (top-k)
- `$curator`: curator identifier/string
- `$field`: field name (e.g., `"zip"`, `"city"`)
- `$dateISO`: ISO date string, e.g., `"2024-01-15"`

---

# 1. Workloads (Cypher listings)

## WQ-A: Baseline topology and label-as-property access

### (WQ-A1) Label membership — count persons by label-as-property
```cypher
MATCH (n:Entity)
WHERE "Person" IN n.k_noval
RETURN count(n) AS persons;
```

### (WQ-A2) One-hop neighborhood — degree for a given person `$pid`

**Baseline GPG**
```cypher
MATCH (p:Entity {id:$pid})-[r:REL]->(q:Entity)
WHERE "KNOWS" IN r.k_noval
RETURN count(q) AS deg;
```

**Reified (GPG_e / GPG_p / GPG_c)**
```cypher
MATCH (p:Entity {id:$pid})<-[:SRC]-(f:EdgeFact)-[:TGT]->(q:Entity)
WHERE "KNOWS" IN f.k_noval
RETURN count(q) AS deg;
```

### (WQ-A3) City selection by population — threshold `$pop`

**Baseline GPG / GPG_e (population as value-set property on node)**
```cypher
MATCH (c:Entity)
WHERE "City" IN c.k_noval
  AND any(x IN c.population WHERE x > $pop)
RETURN count(c) AS cities;
```

**GPG_p / GPG_c (population as occurrences)**
```cypher
MATCH (c:Entity)<-[:SRC]-(p:PropEdge)-[:TGT]->(v:Value)
WHERE "City" IN c.k_noval
  AND p.pkey = "population"
  AND v.val > $pop
RETURN count(DISTINCT c) AS cities;
```

---

## WQ-B: Fact-level metadata and provenance (RQ1)

### (WQ-B1) Filter by temporal qualifier `since` — threshold `$since`

**Reified**
```cypher
MATCH (f:EdgeFact)-[:SRC]->(a:Entity), (f)-[:TGT]->(b:Entity)
WHERE "KNOWS" IN f.k_noval AND f.m_since >= $since
RETURN count(f) AS facts;
```

**Baseline GPG (metadata on relationship)**
```cypher
MATCH (:Entity)-[r:REL]->(:Entity)
WHERE "KNOWS" IN r.k_noval AND r.m_since >= $since
RETURN count(r) AS facts;
```

### (WQ-B2) Join with rich provenance source (name, URL)

**Reified**
```cypher
MATCH (f:EdgeFact)-[:HAS_SOURCE]->(s:Source)
WHERE "KNOWS" IN f.k_noval AND f.m_since >= $since
RETURN s.id, s.name, count(f) AS facts
ORDER BY facts DESC;
```

**Baseline GPG (value-based join via `m_sourceId`)**
```cypher
MATCH ()-[r:REL]->()
WHERE "KNOWS" IN r.k_noval AND r.m_since >= $since
MATCH (s:Source {id:r.m_sourceId})
RETURN s.id, s.name, count(r) AS facts
ORDER BY facts DESC;
```

---

## WQ-C: Property-occurrence metadata (RQ2)

**Applies to:** GPG_p and GPG_c
**Assumption:** `population` is stored as occurrences with qualifiers `m_point_in_time` and `m_confidence`.

### (WQ-C1) Qualified read for a given city and year
```cypher
MATCH (c:Entity {id:$cid})<-[:SRC]-(p:PropEdge)-[:TGT]->(v:Value)
WHERE p.pkey = "population"
  AND p.m_point_in_time = $year
  AND p.m_confidence >= $conf
RETURN v.val AS population, p.m_point_in_time AS year, p.m_confidence AS conf;
```

### (WQ-C2) Top-k most confident population occurrences
```cypher
MATCH (c:Entity {id:$cid})<-[:SRC]-(p:PropEdge)-[:TGT]->(v:Value)
WHERE p.pkey = "population"
RETURN v.val AS population, p.m_point_in_time AS year, p.m_confidence AS conf
ORDER BY conf DESC
LIMIT $k;
```

### (WQ-C3) Existential selection under qualifiers
```cypher
MATCH (c:Entity)
WHERE "City" IN c.k_noval AND EXISTS {
  MATCH (c)<-[:SRC]-(p:PropEdge)-[:TGT]->(v:Value)
  WHERE p.pkey = "population"
    AND v.val > $pop
    AND p.m_confidence >= $conf
}
RETURN count(c) AS cities;
```

---

## WQ-D: Component-level metadata over structured values (RQ3)

**Applies to:** GPG_c only
**Assumption:** an `address` property occurrence points to a root `:Value` node; each field is reachable via
`:CompEdge` selectors `selKind="field"` and `selVal=<fieldName>`.

### (WQ-D1) Find persons by curated ZIP field
```cypher
MATCH (p:Entity)
WHERE "Person" IN p.k_noval AND EXISTS {
  MATCH (p)<-[:SRC]-(pe:PropEdge)-[:TGT]->(root:Value)
  WHERE pe.pkey = "address"
  MATCH (ce:CompEdge)-[:SRC]->(root), (ce)-[:TGT]->(z:Value)
  WHERE ce.selKind = "field" AND ce.selVal = "zip"
    AND ce.m_curator = $curator
}
RETURN count(p) AS persons;
```

### (WQ-D2) Temporal selection on component metadata
> We store `m_lastUpdated` as an ISO date string; lexicographic order coincides with chronological order.
```cypher
MATCH (p:Entity)
WHERE "Person" IN p.k_noval AND EXISTS {
  MATCH (p)<-[:SRC]-(pe:PropEdge)-[:TGT]->(root:Value)
  WHERE pe.pkey = "address"
  MATCH (ce:CompEdge)-[:SRC]->(root)
  WHERE ce.selKind = "field" AND ce.selVal = $field
    AND ce.m_lastUpdated >= $dateISO
}
RETURN count(p) AS persons;
```

### (WQ-D3) Projection of structured payload with component qualifiers
```cypher
MATCH (p:Entity {id:$pid})<-[:SRC]-(pe:PropEdge)-[:TGT]->(root:Value)
WHERE pe.pkey = "address"
MATCH (ce:CompEdge)-[:SRC]->(root), (ce)-[:TGT]->(v:Value)
WHERE ce.selKind = "field" AND ce.selVal IN ["zip","city"]
RETURN root.val AS address_json,
       ce.selVal AS field, v.val AS field_value,
       ce.m_curator AS curator, ce.m_lastUpdated AS lastUpdated
ORDER BY field;
```

---


This section documents the reproducible data-loading workflow used in the experiments:
1) expose the dataset directory via a local HTTP server on `localhost`,
2) import CSVs using `apoc.import.csv`,
3) create constraints and indexes required by the workloads.

---

## A. Local HTTP server (localhost)

From the dataset root (the directory that contains `D1-S/`, `D1-M/`, etc.) run:

```bash
cd <DATASET_ROOT>
python3 -m http.server 8000
```

This serves files under:

- `http://localhost:8000/<path-under-DATASET_ROOT>`

Example:

- `http://localhost:8000/D1-S/gpg/nodes_entities.csv`

---

## B. APOC imports (`apoc.import.csv`)

### Common APOC options

We use a uniform option set across all imports:

- `delimiter: ','`
- `arrayDelimiter: ';'`
- `stringIds: true` (IDs are imported as strings)
- `batchSize: 50000` (tune if needed)
- duplicate-tolerant flags to make reruns convenient:
  - `ignoreDuplicateNodes: true`
  - `ignoreDuplicateRelationships: true`
- robustness flags:
  - `ignoreEmptyCellArray: true`
  - `ignoreBlankString: true`

> Practical note: when iterating, it is often safer to clear the database between runs than to rely on
> duplicate-tolerance. The options above exist primarily to avoid accidental failures during development.

---

### B1. Baseline GPG (native relationships)

Canonical import shape (illustrative):

```cypher
CALL apoc.import.csv(
  [
    {fileName:'http://localhost:8000/D1-S/gpg/nodes_entities.csv', labels:['Entity']},
    {fileName:'http://localhost:8000/D1-S/gpg/nodes_sources.csv',  labels:['Source']}
  ],
  [
    {fileName:'http://localhost:8000/D1-S/gpg/rels_rel.csv', type:'REL'}
  ],
  {delimiter:',', arrayDelimiter:';', stringIds:true,
   batchSize:50000, ignoreDuplicateNodes:true, ignoreDuplicateRelationships:true,
   ignoreEmptyCellArray:true, ignoreBlankString:true}
);
```

---

### B2. Reified encodings (GPG_e, GPG_p, GPG_c)

Canonical import shape (illustrative, shown for `gpgc`):

```cypher
CALL apoc.import.csv(
  [
    {fileName:'http://localhost:8000/D1-S/gpgc/nodes_entities.csv',  labels:['Entity']},
    {fileName:'http://localhost:8000/D1-S/gpgc/nodes_sources.csv',   labels:['Source']},
    {fileName:'http://localhost:8000/D1-S/gpgc/nodes_edgefacts.csv', labels:['EdgeFact']},
    {fileName:'http://localhost:8000/D1-S/gpgc/nodes_propedges.csv', labels:['PropEdge']},
    {fileName:'http://localhost:8000/D1-S/gpgc/nodes_compedges.csv', labels:['CompEdge']},
    {fileName:'http://localhost:8000/D1-S/gpgc/nodes_values.csv',    labels:['Value']}
  ],
  [
    {fileName:'http://localhost:8000/D1-S/gpgc/rels_src.csv',        type:'SRC'},
    {fileName:'http://localhost:8000/D1-S/gpgc/rels_tgt.csv',        type:'TGT'},
    {fileName:'http://localhost:8000/D1-S/gpgc/rels_has_source.csv', type:'HAS_SOURCE'}
  ],
  {delimiter:',', arrayDelimiter:';', stringIds:true,
   batchSize:50000, ignoreDuplicateNodes:true, ignoreDuplicateRelationships:true,
   ignoreEmptyCellArray:true, ignoreBlankString:true}
);
```

---

## C. Constraints and indexes (Neo4j 4.4 syntax)

After each import we create uniqueness constraints on `id` for all node kinds and add indexes that reflect
the workloads.

### C1. Uniqueness constraints

```cypher
CREATE CONSTRAINT entity_id    IF NOT EXISTS ON (n:Entity)   ASSERT n.id IS UNIQUE;
CREATE CONSTRAINT source_id    IF NOT EXISTS ON (n:Source)   ASSERT n.id IS UNIQUE;
CREATE CONSTRAINT edgefact_id  IF NOT EXISTS ON (n:EdgeFact) ASSERT n.id IS UNIQUE;
CREATE CONSTRAINT propedge_id  IF NOT EXISTS ON (n:PropEdge) ASSERT n.id IS UNIQUE;
CREATE CONSTRAINT compedge_id  IF NOT EXISTS ON (n:CompEdge) ASSERT n.id IS UNIQUE;
CREATE CONSTRAINT value_id     IF NOT EXISTS ON (n:Value)    ASSERT n.id IS UNIQUE;
```

### C2. Secondary indexes (workload-driven)

```cypher
CREATE INDEX propedge_pkey IF NOT EXISTS FOR (p:PropEdge) ON (p.pkey);
CREATE INDEX propedge_meta IF NOT EXISTS FOR (p:PropEdge) ON (p.pkey, p.m_point_in_time, p.m_confidence);

CREATE INDEX edgefact_meta IF NOT EXISTS FOR (e:EdgeFact) ON (e.m_since);
CREATE INDEX compedge_sel  IF NOT EXISTS FOR (c:CompEdge) ON (c.selKind, c.selVal);
CREATE INDEX compedge_meta IF NOT EXISTS FOR (c:CompEdge) ON (c.m_curator, c.m_lastUpdated);
```

---

## D. Minimal end-to-end checklist

1. Start HTTP server in `<DATASET_ROOT>` on port 8000.
2. Run the relevant `apoc.import.csv` call for the chosen dataset + encoding directory.
3. Create constraints and indexes.
4. Proceed with workload execution.
