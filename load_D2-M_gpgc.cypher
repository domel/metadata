// ================================================================
// APOC import script (hardcoded URLs, no parameters)
// Dataset: D2-M
// Variant: gpgc
// Assumption: you serve the datasets directory via:
//   cd <DATASET_ROOT> && python3 -m http.server 8000
// Then run this script in Neo4j Browser (Neo4j 4.4.x + APOC 4.4.x).
// ================================================================

// Optional cleanup (UNCOMMENT if you want to clear the database first):
// MATCH (n) DETACH DELETE n;

CALL apoc.import.csv(
  [
    {fileName:'http://localhost:8000/D2-M/gpgc/nodes_entities.csv', labels:['Entity']},
    {fileName:'http://localhost:8000/D2-M/gpgc/nodes_sources.csv',  labels:['Source']},
    {fileName:'http://localhost:8000/D2-M/gpgc/nodes_edgefacts.csv', labels:['EdgeFact']},
    {fileName:'http://localhost:8000/D2-M/gpgc/nodes_propedges.csv', labels:['PropEdge']},
    {fileName:'http://localhost:8000/D2-M/gpgc/nodes_values_num.csv', labels:['Value']},
    {fileName:'http://localhost:8000/D2-M/gpgc/nodes_values_str.csv', labels:['Value']},
    {fileName:'http://localhost:8000/D2-M/gpgc/nodes_compedges.csv', labels:['CompEdge']}
  ],
  [
    {fileName:'http://localhost:8000/D2-M/gpgc/rels_src.csv',        type:'SRC'},
    {fileName:'http://localhost:8000/D2-M/gpgc/rels_tgt.csv',        type:'TGT'},
    {fileName:'http://localhost:8000/D2-M/gpgc/rels_has_source.csv', type:'HAS_SOURCE'}
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
