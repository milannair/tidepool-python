from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence

import httpx

from .errors import (
    NotFoundError,
    ServiceUnavailableError,
    TidepoolError,
    ValidationError,
)
from .types import (
    AttrValue,
    DistanceMetric,
    Document,
    IngestStatus,
    NamespaceInfo,
    Vector,
    VectorResult,
)

def _normalize_namespace(namespace: Optional[str], default: str) -> str:
    if namespace is None:
        namespace = default
    if not isinstance(namespace, str) or not namespace.strip():
        raise ValidationError("Namespace must be a non-empty string")
    return namespace


def _normalize_distance_metric(
    distance_metric: DistanceMetric | str | None,
) -> Optional[str]:
    if distance_metric is None:
        return None
    if isinstance(distance_metric, DistanceMetric):
        return distance_metric.value
    if isinstance(distance_metric, str):
        try:
            return DistanceMetric(distance_metric).value
        except ValueError as exc:
            raise ValidationError(
                "Distance metric must be one of: cosine_distance, euclidean_squared, dot_product"
            ) from exc
    raise ValidationError("Distance metric must be a DistanceMetric or string")


def _validate_vector(vector: Sequence[float], expected_dims: Optional[int] = None) -> List[float]:
    if not isinstance(vector, (list, tuple)):
        raise ValidationError("Vector must be a list of numbers")
    if not vector:
        raise ValidationError("Vector cannot be empty")
    for value in vector:
        if not isinstance(value, (int, float)):
            raise ValidationError("Vector must contain only numbers")
    if expected_dims is not None and len(vector) != expected_dims:
        raise ValidationError(f"Expected {expected_dims} dimensions, got {len(vector)}")
    return list(vector)


def _is_attr_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, list):
        return all(_is_attr_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_attr_value(val) for key, val in value.items())
    return False


def _validate_attributes(attributes: Optional[Dict[str, AttrValue]]) -> None:
    if attributes is None:
        return
    if not isinstance(attributes, dict):
        raise ValidationError("Attributes must be a dictionary")
    for key, value in attributes.items():
        if not isinstance(key, str):
            raise ValidationError("Attribute keys must be strings")
        if not _is_attr_value(value):
            raise ValidationError(f"Invalid attribute value for key '{key}'")


def _validate_filters(filters: Optional[Dict[str, AttrValue]]) -> None:
    if filters is None:
        return
    if not isinstance(filters, dict):
        raise ValidationError("Filters must be a dictionary")
    for key, value in filters.items():
        if not isinstance(key, str):
            raise ValidationError("Filter keys must be strings")
        if not _is_attr_value(value):
            raise ValidationError(f"Invalid filter value for key '{key}'")


def _validate_ids(ids: Iterable[str]) -> List[str]:
    if not isinstance(ids, (list, tuple)):
        raise ValidationError("Ids must be a list of strings")
    if not ids:
        raise ValidationError("Ids list cannot be empty")
    normalized: List[str] = []
    for value in ids:
        if not isinstance(value, str) or not value.strip():
            raise ValidationError("Each id must be a non-empty string")
        normalized.append(value)
    return normalized


def _validate_documents(vectors: Sequence[Document]) -> List[Document]:
    if not isinstance(vectors, (list, tuple)):
        raise ValidationError("Vectors must be a list of Document objects")
    if not vectors:
        raise ValidationError("Vectors list cannot be empty")
    expected_dims: Optional[int] = None
    normalized: List[Document] = []
    for doc in vectors:
        if not isinstance(doc, Document):
            raise ValidationError("Each vector must be a Document")
        if not isinstance(doc.id, str) or not doc.id.strip():
            raise ValidationError("Document id must be a non-empty string")
        vector = _validate_vector(doc.vector, expected_dims)
        if expected_dims is None:
            expected_dims = len(vector)
        _validate_attributes(doc.attributes)
        normalized.append(Document(id=doc.id, vector=vector, attributes=doc.attributes))
    return normalized


def _validate_positive_int(value: Optional[int], name: str) -> Optional[int]:
    if value is None:
        return None
    if not isinstance(value, int) or value <= 0:
        raise ValidationError(f"{name} must be a positive integer")
    return value


def _document_to_payload(doc: Document) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"id": doc.id, "vector": list(doc.vector)}
    if doc.attributes is not None:
        payload["attributes"] = doc.attributes
    return payload


def _extract_error_message(response: httpx.Response) -> str:
    try:
        data = response.json()
        if isinstance(data, dict):
            message = data.get("error") or data.get("message")
            if isinstance(message, str) and message.strip():
                return message
    except ValueError:
        pass
    text = response.text.strip()
    if text:
        return text
    return f"HTTP {response.status_code}"


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


def _parse_vector_results(data: Any) -> List[VectorResult]:
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        if isinstance(data.get("results"), list):
            items = data["results"]
        elif isinstance(data.get("vectors"), list):
            items = data["vectors"]
        else:
            raise TidepoolError("Invalid query response")
    else:
        raise TidepoolError("Invalid query response")

    results: List[VectorResult] = []
    for item in items:
        if not isinstance(item, dict):
            raise TidepoolError("Invalid query response")
        dist_value = item.get("dist", item.get("distance", 0.0))
        results.append(
            VectorResult(
                id=item.get("id"),
                dist=float(dist_value or 0.0),
                vector=item.get("vector"),
                attributes=item.get("attributes"),
            )
        )
    return results


def _parse_namespace_info(data: Dict[str, Any]) -> NamespaceInfo:
    return NamespaceInfo(
        namespace=data.get("namespace"),
        approx_count=int(data.get("approx_count", 0)),
        dimensions=int(data.get("dimensions", 0)),
    )


def _parse_ingest_status(data: Dict[str, Any]) -> IngestStatus:
    return IngestStatus(
        last_run=_parse_datetime(data.get("last_run")),
        wal_files=int(data.get("wal_files", 0)),
        wal_entries=int(data.get("wal_entries", 0)),
        segments=int(data.get("segments", 0)),
        total_vecs=int(data.get("total_vecs", 0)),
        dimensions=int(data.get("dimensions", 0)),
    )


class TidepoolClient:
    """Synchronous Tidepool client."""

    max_retries: int = 3
    base_delay: float = 0.5
    max_delay: float = 10.0

    def __init__(
        self,
        query_url: str = "http://localhost:8080",
        ingest_url: str = "http://localhost:8081",
        timeout: float = 30.0,
        namespace: str = "default",
    ) -> None:
        self._namespace = namespace
        limits = httpx.Limits(max_connections=10, max_keepalive_connections=10)
        self._query_client = httpx.Client(
            base_url=query_url,
            timeout=timeout,
            limits=limits,
        )
        self._ingest_client = httpx.Client(
            base_url=ingest_url,
            timeout=timeout,
            limits=limits,
        )

    def close(self) -> None:
        self._query_client.close()
        self._ingest_client.close()

    def __enter__(self) -> "TidepoolClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _select_client(self, service: str) -> httpx.Client:
        if service == "query":
            return self._query_client
        if service == "ingest":
            return self._ingest_client
        raise ValidationError("Service must be 'query' or 'ingest'")

    def _with_retry(self, func):
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                return func()
            except ServiceUnavailableError as exc:
                last_error = exc
            except httpx.RequestError as exc:
                last_error = exc
            if attempt == self.max_retries:
                break
            delay = min(self.base_delay * (2**attempt), self.max_delay)
            time.sleep(delay)
        if isinstance(last_error, TidepoolError):
            raise last_error
        raise TidepoolError(str(last_error))

    def _request_json(
        self, client: httpx.Client, method: str, path: str, json_body: Any = None
    ) -> Any:
        def send() -> httpx.Response:
            response = client.request(method, path, json=json_body)
            if response.status_code == 503:
                message = _extract_error_message(response)
                raise ServiceUnavailableError(message, status_code=503)
            return response

        response = self._with_retry(send)
        if response.status_code >= 400:
            self._raise_for_error(response)
        if response.status_code == 204 or not response.content:
            return None
        try:
            return response.json()
        except ValueError as exc:
            raise TidepoolError("Invalid JSON response", status_code=response.status_code) from exc

    def _raise_for_error(self, response: httpx.Response) -> None:
        message = _extract_error_message(response)
        status = response.status_code
        if status == 400:
            raise ValidationError(message, status_code=status)
        if status == 404:
            raise NotFoundError(message, status_code=status)
        if status == 413:
            raise ValidationError(message, status_code=status)
        if status == 503:
            raise ServiceUnavailableError(message, status_code=status)
        raise TidepoolError(message, status_code=status)

    def health(self, service: str = "query") -> Dict[str, Any]:
        data = self._request_json(self._select_client(service), "GET", "/health")
        if isinstance(data, dict):
            status = data.get("status")
            if status and status != "healthy":
                raise TidepoolError(f"Service unhealthy: {status}")
            return data
        raise TidepoolError("Invalid health response")

    def upsert(
        self,
        vectors: List[Document],
        namespace: Optional[str] = None,
        distance_metric: DistanceMetric | str | None = DistanceMetric.COSINE,
    ) -> None:
        normalized = _validate_documents(vectors)
        metric = _normalize_distance_metric(distance_metric)
        payload = {
            "vectors": [_document_to_payload(doc) for doc in normalized],
        }
        if metric is not None:
            payload["distance_metric"] = metric
        namespace = _normalize_namespace(namespace, self._namespace)
        self._request_json(
            self._ingest_client, "POST", f"/v1/vectors/{namespace}", json_body=payload
        )

    def query(
        self,
        vector: Vector,
        top_k: int = 10,
        namespace: Optional[str] = None,
        distance_metric: DistanceMetric | str | None = DistanceMetric.COSINE,
        include_vectors: bool = False,
        filters: Optional[Dict[str, AttrValue]] = None,
        ef_search: Optional[int] = None,
        nprobe: Optional[int] = None,
    ) -> List[VectorResult]:
        vec = _validate_vector(vector)
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValidationError("top_k must be a positive integer")
        _validate_filters(filters)
        ef_search = _validate_positive_int(ef_search, "ef_search")
        nprobe = _validate_positive_int(nprobe, "nprobe")
        metric = _normalize_distance_metric(distance_metric)
        payload: Dict[str, Any] = {
            "vector": vec,
            "top_k": top_k,
            "include_vectors": bool(include_vectors),
        }
        if metric is not None:
            payload["distance_metric"] = metric
        if filters is not None:
            payload["filters"] = filters
        if ef_search is not None:
            payload["ef_search"] = ef_search
        if nprobe is not None:
            payload["nprobe"] = nprobe
        namespace = _normalize_namespace(namespace, self._namespace)
        data = self._request_json(
            self._query_client, "POST", f"/v1/vectors/{namespace}", json_body=payload
        )
        return _parse_vector_results(data)

    def delete(self, ids: List[str], namespace: Optional[str] = None) -> None:
        normalized = _validate_ids(ids)
        payload = {"ids": normalized}
        namespace = _normalize_namespace(namespace, self._namespace)
        self._request_json(
            self._ingest_client, "DELETE", f"/v1/vectors/{namespace}", json_body=payload
        )

    def get_namespace(self, namespace: Optional[str] = None) -> NamespaceInfo:
        namespace = _normalize_namespace(namespace, self._namespace)
        data = self._request_json(self._query_client, "GET", f"/v1/namespaces/{namespace}")
        if not isinstance(data, dict):
            raise TidepoolError("Invalid namespace response")
        return _parse_namespace_info(data)

    def list_namespaces(self) -> List[str]:
        data = self._request_json(self._query_client, "GET", "/v1/namespaces")
        if isinstance(data, list):
            return [str(name) for name in data]
        if isinstance(data, dict):
            namespaces = data.get("namespaces")
            if isinstance(namespaces, list):
                return [str(name) for name in namespaces]
            namespace_list = data.get("namespace_list")
            if isinstance(namespace_list, list):
                return [str(name) for name in namespace_list]
        raise TidepoolError("Invalid namespaces response")

    def status(self) -> IngestStatus:
        data = self._request_json(self._ingest_client, "GET", "/status")
        if not isinstance(data, dict):
            raise TidepoolError("Invalid status response")
        return _parse_ingest_status(data)

    def compact(self) -> None:
        self._request_json(self._ingest_client, "POST", "/compact")


class AsyncTidepoolClient:
    """Asynchronous Tidepool client."""

    max_retries: int = 3
    base_delay: float = 0.5
    max_delay: float = 10.0

    def __init__(
        self,
        query_url: str = "http://localhost:8080",
        ingest_url: str = "http://localhost:8081",
        timeout: float = 30.0,
        namespace: str = "default",
    ) -> None:
        self._namespace = namespace
        limits = httpx.Limits(max_connections=10, max_keepalive_connections=10)
        self._query_client = httpx.AsyncClient(
            base_url=query_url,
            timeout=timeout,
            limits=limits,
        )
        self._ingest_client = httpx.AsyncClient(
            base_url=ingest_url,
            timeout=timeout,
            limits=limits,
        )

    async def close(self) -> None:
        await self._query_client.aclose()
        await self._ingest_client.aclose()

    async def __aenter__(self) -> "AsyncTidepoolClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    def _select_client(self, service: str) -> httpx.AsyncClient:
        if service == "query":
            return self._query_client
        if service == "ingest":
            return self._ingest_client
        raise ValidationError("Service must be 'query' or 'ingest'")

    async def _with_retry(self, func):
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                return await func()
            except ServiceUnavailableError as exc:
                last_error = exc
            except httpx.RequestError as exc:
                last_error = exc
            if attempt == self.max_retries:
                break
            delay = min(self.base_delay * (2**attempt), self.max_delay)
            await asyncio.sleep(delay)
        if isinstance(last_error, TidepoolError):
            raise last_error
        raise TidepoolError(str(last_error))

    async def _request_json(
        self, client: httpx.AsyncClient, method: str, path: str, json_body: Any = None
    ) -> Any:
        async def send() -> httpx.Response:
            response = await client.request(method, path, json=json_body)
            if response.status_code == 503:
                message = _extract_error_message(response)
                raise ServiceUnavailableError(message, status_code=503)
            return response

        response = await self._with_retry(send)
        if response.status_code >= 400:
            self._raise_for_error(response)
        if response.status_code == 204 or not response.content:
            return None
        try:
            return response.json()
        except ValueError as exc:
            raise TidepoolError("Invalid JSON response", status_code=response.status_code) from exc

    def _raise_for_error(self, response: httpx.Response) -> None:
        message = _extract_error_message(response)
        status = response.status_code
        if status == 400:
            raise ValidationError(message, status_code=status)
        if status == 404:
            raise NotFoundError(message, status_code=status)
        if status == 413:
            raise ValidationError(message, status_code=status)
        if status == 503:
            raise ServiceUnavailableError(message, status_code=status)
        raise TidepoolError(message, status_code=status)

    async def health(self, service: str = "query") -> Dict[str, Any]:
        data = await self._request_json(self._select_client(service), "GET", "/health")
        if isinstance(data, dict):
            status = data.get("status")
            if status and status != "healthy":
                raise TidepoolError(f"Service unhealthy: {status}")
            return data
        raise TidepoolError("Invalid health response")

    async def upsert(
        self,
        vectors: List[Document],
        namespace: Optional[str] = None,
        distance_metric: DistanceMetric | str | None = DistanceMetric.COSINE,
    ) -> None:
        normalized = _validate_documents(vectors)
        metric = _normalize_distance_metric(distance_metric)
        payload = {
            "vectors": [_document_to_payload(doc) for doc in normalized],
        }
        if metric is not None:
            payload["distance_metric"] = metric
        namespace = _normalize_namespace(namespace, self._namespace)
        await self._request_json(
            self._ingest_client, "POST", f"/v1/vectors/{namespace}", json_body=payload
        )

    async def query(
        self,
        vector: Vector,
        top_k: int = 10,
        namespace: Optional[str] = None,
        distance_metric: DistanceMetric | str | None = DistanceMetric.COSINE,
        include_vectors: bool = False,
        filters: Optional[Dict[str, AttrValue]] = None,
        ef_search: Optional[int] = None,
        nprobe: Optional[int] = None,
    ) -> List[VectorResult]:
        vec = _validate_vector(vector)
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValidationError("top_k must be a positive integer")
        _validate_filters(filters)
        ef_search = _validate_positive_int(ef_search, "ef_search")
        nprobe = _validate_positive_int(nprobe, "nprobe")
        metric = _normalize_distance_metric(distance_metric)
        payload: Dict[str, Any] = {
            "vector": vec,
            "top_k": top_k,
            "include_vectors": bool(include_vectors),
        }
        if metric is not None:
            payload["distance_metric"] = metric
        if filters is not None:
            payload["filters"] = filters
        if ef_search is not None:
            payload["ef_search"] = ef_search
        if nprobe is not None:
            payload["nprobe"] = nprobe
        namespace = _normalize_namespace(namespace, self._namespace)
        data = await self._request_json(
            self._query_client, "POST", f"/v1/vectors/{namespace}", json_body=payload
        )
        return _parse_vector_results(data)

    async def delete(self, ids: List[str], namespace: Optional[str] = None) -> None:
        normalized = _validate_ids(ids)
        payload = {"ids": normalized}
        namespace = _normalize_namespace(namespace, self._namespace)
        await self._request_json(
            self._ingest_client, "DELETE", f"/v1/vectors/{namespace}", json_body=payload
        )

    async def get_namespace(self, namespace: Optional[str] = None) -> NamespaceInfo:
        namespace = _normalize_namespace(namespace, self._namespace)
        data = await self._request_json(
            self._query_client, "GET", f"/v1/namespaces/{namespace}"
        )
        if not isinstance(data, dict):
            raise TidepoolError("Invalid namespace response")
        return _parse_namespace_info(data)

    async def list_namespaces(self) -> List[str]:
        data = await self._request_json(self._query_client, "GET", "/v1/namespaces")
        if isinstance(data, list):
            return [str(name) for name in data]
        if isinstance(data, dict):
            namespaces = data.get("namespaces")
            if isinstance(namespaces, list):
                return [str(name) for name in namespaces]
            namespace_list = data.get("namespace_list")
            if isinstance(namespace_list, list):
                return [str(name) for name in namespace_list]
        raise TidepoolError("Invalid namespaces response")

    async def status(self) -> IngestStatus:
        data = await self._request_json(self._ingest_client, "GET", "/status")
        if not isinstance(data, dict):
            raise TidepoolError("Invalid status response")
        return _parse_ingest_status(data)

    async def compact(self) -> None:
        await self._request_json(self._ingest_client, "POST", "/compact")
