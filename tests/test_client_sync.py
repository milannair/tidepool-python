import time
import unittest
from unittest import mock

import httpx

from tidepool import Document, TidepoolClient
from tidepool.client import _normalize_alpha
from tidepool.errors import NotFoundError, ServiceUnavailableError, TidepoolError, ValidationError
from tidepool.types import DistanceMetric


class TidepoolClientSyncTests(unittest.TestCase):
    def test_health_unhealthy_raises(self) -> None:
        def handler(client_self, method, path, json=None, **kwargs):
            return httpx.Response(200, json={"service": "query", "status": "unhealthy"})

        with mock.patch.object(httpx.Client, "request", new=handler):
            with TidepoolClient() as client:
                with self.assertRaises(TidepoolError):
                    client.health("query")

    def test_request_json_invalid_json(self) -> None:
        def handler(client_self, method, path, json=None, **kwargs):
            return httpx.Response(200, content=b"not-json", headers={"content-type": "application/json"})

        with mock.patch.object(httpx.Client, "request", new=handler):
            with TidepoolClient() as client:
                with self.assertRaises(TidepoolError):
                    client._request_json(client._query_client, "GET", "/health")

    def test_request_json_retries_on_503(self) -> None:
        calls = {"count": 0}

        def handler(client_self, method, path, json=None, **kwargs):
            calls["count"] += 1
            if calls["count"] < 3:
                return httpx.Response(503, json={"error": "busy"})
            return httpx.Response(200, json={"ok": True})

        with mock.patch.object(httpx.Client, "request", new=handler), mock.patch.object(
            time, "sleep", return_value=None
        ):
            with TidepoolClient() as client:
                result = client._request_json(client._query_client, "GET", "/health")
                self.assertEqual(result["ok"], True)
                self.assertEqual(calls["count"], 3)

    def test_error_mapping(self) -> None:
        def handler(client_self, method, path, json=None, **kwargs):
            if path == "/v1/namespaces/missing":
                return httpx.Response(404, json={"error": "missing"})
            if path == "/v1/vectors/default":
                return httpx.Response(400, json={"error": "bad"})
            return httpx.Response(503, json={"error": "down"})

        with mock.patch.object(httpx.Client, "request", new=handler):
            with TidepoolClient() as client:
                client.max_retries = 0
                with self.assertRaises(NotFoundError):
                    client.get_namespace("missing")
                with self.assertRaises(ValidationError):
                    client.query(vector=[0.1, 0.2, 0.3])
                with self.assertRaises(ServiceUnavailableError):
                    client.status()

    def test_query_payload_and_alpha_clamp(self) -> None:
        calls = []

        def handler(client_self, method, path, json=None, **kwargs):
            calls.append({"method": method, "path": path, "json": json})
            if method == "POST" and path.startswith("/v1/vectors/"):
                namespace = path.rsplit("/", 1)[-1]
                return httpx.Response(
                    200, json={"namespace": namespace, "results": [{"id": "a", "score": 0.1}]}
                )
            return httpx.Response(204)

        with mock.patch.object(httpx.Client, "request", new=handler):
            with TidepoolClient(default_namespace="default") as client:
                response = client.query(
                    vector=[0.1, 0.2, 0.3],
                    text=" hello ",
                    mode="hybrid",
                    top_k=5,
                    include_vectors=True,
                    distance_metric=DistanceMetric.DOT_PRODUCT,
                    filters={"tag": "a"},
                    alpha=2.0,
                    fusion="rrf",
                    rrf_k=10,
                )
                self.assertEqual(response.namespace, "default")
                payload = calls[-1]["json"]
                self.assertEqual(payload["mode"], "hybrid")
                self.assertEqual(payload["text"], "hello")
                self.assertEqual(payload["alpha"], 1.0)
                self.assertEqual(payload["fusion"], "rrf")
                self.assertEqual(payload["rrf_k"], 10)

    def test_text_only_query_omits_vector(self) -> None:
        calls = []

        def handler(client_self, method, path, json=None, **kwargs):
            calls.append({"method": method, "path": path, "json": json})
            return httpx.Response(
                200, json={"namespace": "docs", "results": [{"id": "a", "score": 0.1}]}
            )

        with mock.patch.object(httpx.Client, "request", new=handler):
            with TidepoolClient(default_namespace="docs") as client:
                client.query(text="machine learning", mode="text")
                payload = calls[-1]["json"]
                self.assertEqual(payload["mode"], "text")
                self.assertEqual(payload["text"], "machine learning")
                self.assertNotIn("vector", payload)

    def test_upsert_payload_includes_distance_metric(self) -> None:
        calls = []

        def handler(client_self, method, path, json=None, **kwargs):
            calls.append({"method": method, "path": path, "json": json})
            return httpx.Response(204)

        with mock.patch.object(httpx.Client, "request", new=handler):
            with TidepoolClient(default_namespace="default") as client:
                client.upsert(
                    [Document(id="doc-1", vector=[0.1, 0.2, 0.3])],
                    distance_metric=DistanceMetric.EUCLIDEAN,
                )
                payload = calls[-1]["json"]
                self.assertEqual(payload["distance_metric"], DistanceMetric.EUCLIDEAN.value)

    def test_delete_validation(self) -> None:
        with TidepoolClient(default_namespace="default") as client:
            with self.assertRaises(ValidationError):
                client.delete([])

    def test_normalize_alpha_helper(self) -> None:
        self.assertEqual(_normalize_alpha(1.2), 1.0)
        self.assertEqual(_normalize_alpha(-0.2), 0.0)


if __name__ == "__main__":
    unittest.main()
