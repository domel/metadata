// ================================================================
// APOC import script (hardcoded URLs, no parameters)
// Dataset: D3-L
// Variant: gpg
// Assumption: you serve the datasets directory via:
//   cd <DATASET_ROOT> && python3 -m http.server 8000
// Then run this script in Neo4j Browser (Neo4j 4.4.x + APOC 4.4.x).
// ================================================================

// Optional cleanup (UNCOMMENT if you want to clear the database first):
// MATCH (n) DETACH DELETE n;

CALL apoc.import.csv(
  [
    {fileName:'http://localhost:8000/D3-L/gpg/nodes_entities.csv', labels:['Entity']},
    {fileName:'http://localhost:8000/D3-L/gpg/nodes_sources.csv',  labels:['Source']}
  ],
  [
    {fileName:'http://localhost:8000/D3-L/gpg/rels_rel.csv', type:'REL'}
  ],
  {
    delimiter: ',',
    arrayDelimiter: ';',
    stringIds: true,
    quoteChar: '"',
    batchSize: 50000,
    ignoreDuplicateNodes: true,
    ignoreDuplicateRelationships: true,
    ignoreEmptyCellArray: true,
    ignoreBlankString: true
  }
);
