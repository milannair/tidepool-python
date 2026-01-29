# Tidepool Python Client

A lightweight Python client for Tidepool's query and ingest HTTP services.

## Features

- Sync and async clients
- Vector upserts, queries, deletes
- Namespace inspection
- Ingest status + compaction
- Validation + helpful error mapping

## Install

```bash
pip install tidepool
```

If you're installing from source:

```bash
pip install -e .
```

## Quickstart

```python
from tidepool import TidepoolClient, Document

client = TidepoolClient(
    query_url="http://localhost:8080",
    ingest_url="http://localhost:8081",
)

client.upsert(
    [
        Document(
            id="doc-1",
            vector=[0.1, 0.2, 0.3, 0.4],
            attributes={"category": "news"},
        )
    ]
)

client.compact()

results = client.query(vector=[0.1, 0.2, 0.3, 0.4], top_k=5)
for result in results:
    print(result.id, result.dist)
```

## Async Usage

```python
import asyncio
from tidepool import AsyncTidepoolClient, Document

async def main():
    client = AsyncTidepoolClient()
    await client.upsert([
        Document(id="doc-1", vector=[0.1, 0.2, 0.3], attributes={"tag": "a"})
    ])
    results = await client.query(vector=[0.1, 0.2, 0.3], top_k=3)
    await client.close()
    return results

asyncio.run(main())
```

## API Overview

- `TidepoolClient.health(service="query" | "ingest")`
- `TidepoolClient.upsert(vectors, namespace=None, distance_metric=DistanceMetric.COSINE)`
- `TidepoolClient.query(vector, top_k=10, namespace=None, distance_metric=DistanceMetric.COSINE, include_vectors=False, filters=None, ef_search=None, nprobe=None)`
- `TidepoolClient.delete(ids, namespace=None)`
- `TidepoolClient.get_namespace(namespace=None)`
- `TidepoolClient.list_namespaces()`
- `TidepoolClient.status()`
- `TidepoolClient.compact()`

The async client mirrors the same API.

## License

MIT. See `LICENSE`.
