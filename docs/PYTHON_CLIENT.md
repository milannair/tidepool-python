# Tidepool Python Client (Dynamic Namespaces)

Tidepool Phase 8 adds dynamic namespaces. The Python client now supports a
default namespace plus per-request overrides so a single client can operate on
multiple namespaces.

## Client Initialization

```python
from tidepool import TidepoolClient

client = TidepoolClient(
    query_url="http://localhost:8080",
    ingest_url="http://localhost:8081",
    default_namespace="default",  # Optional default
)
```

`default_namespace` is used when a method call omits `namespace`.

## Method Signatures

```python
client.upsert(vectors, namespace=None, distance_metric=DistanceMetric.COSINE)
client.query(vector=None, top_k=10, namespace=None, distance_metric=DistanceMetric.COSINE,
             include_vectors=False, filters=None, ef_search=None, nprobe=None,
             text=None, mode=None, alpha=None, fusion=None, rrf_k=None)
client.delete(ids, namespace=None)

client.get_namespace(namespace=None)
client.list_namespaces()

client.get_namespace_status(namespace=None)
client.compact(namespace=None)

client.status()  # Ingest service status (global)
client.health(service="query" | "ingest")
```

## Full-Text & Hybrid Search

Include `text` on documents to enable BM25 search. For queries, set `mode="text"` for BM25-only or `mode="hybrid"` to fuse vector and text results. Hybrid queries support `alpha` (blend weight) and `fusion="rrf"` when you want reciprocal-rank fusion instead of score blending.

## Response Models

```python
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

@dataclass
class NamespaceStatus:
    last_run: Optional[datetime]
    wal_files: int
    wal_entries: int
    segments: int
    total_vecs: int
    dimensions: int

@dataclass
class QueryResponse:
    results: List[VectorResult]
    namespace: str

@dataclass
class VectorResult:
    id: str
    score: float
    vector: Optional[List[float]]
    attributes: Optional[dict]

@dataclass
class NamespaceInfo:
    namespace: str
    approx_count: int
    dimensions: int
    pending_compaction: Optional[bool]
```

`QueryResponse.namespace` returns the namespace that was queried.

`list_namespaces` returns a list of `NamespaceInfo` entries (not just names), matching the query service response.

## Usage Examples

### Multi-Tenant Application

```python
client = TidepoolClient(ingest_url="...", query_url="...")

# Each tenant gets their own namespace
def index_tenant_data(tenant_id: str, documents: List[dict]):
    vectors = [embed(doc) for doc in documents]
    client.upsert(vectors, namespace=f"tenant_{tenant_id}")

def search_tenant(tenant_id: str, query: str, top_k: int = 10):
    query_vec = embed(query)
    return client.query(
        vector=query_vec,
        text=query,
        mode="hybrid",
        alpha=0.7,
        top_k=top_k,
        namespace=f"tenant_{tenant_id}",
    )
```

### Different Data Types

```python
client = TidepoolClient(default_namespace="products")

# Index different types of data in separate namespaces
client.upsert(product_vectors, namespace="products")
client.upsert(user_vectors, namespace="users")
client.upsert(doc_vectors, namespace="documents")

# Query specific namespace
response = client.query(query_vec, namespace="products")
results = response.results

# Check namespace status
status = client.get_namespace_status("products")
print(f"Products: {status.total_vecs} vectors in {status.segments} segments")
```

### Namespace Management

```python
# Check if namespace needs compaction
status = client.get_namespace_status("products")
if status.wal_entries > 1000:
    client.compact("products")
    print("Compaction triggered")
```

## Error Handling

If a namespace is restricted by `ALLOWED_NAMESPACES`, the API returns:

```
404 Not Found
{"error": "namespace not found"}
```

The client surfaces this as `NotFoundError` with the provided message.
