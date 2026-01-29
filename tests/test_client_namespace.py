import unittest
from unittest import mock

import httpx

from tidepool import Document, TidepoolClient


class NamespaceClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calls = []

        def handler(client_self, method, path, json=None, **kwargs):
            self.calls.append({"method": method, "path": path, "json": json})
            if path == "/v1/namespaces":
                return httpx.Response(
                    200,
                    json={
                        "namespaces": [
                            {
                                "namespace": "default",
                                "approx_count": 10,
                                "dimensions": 3,
                                "pending_compaction": True,
                            },
                            {"namespace": "products", "approx_count": 5, "dimensions": 3},
                        ]
                    },
                )
            if path.endswith("/status"):
                return httpx.Response(
                    200,
                    json={
                        "last_run": "2025-01-01T00:00:00Z",
                        "wal_files": 2,
                        "wal_entries": 3,
                        "segments": 4,
                        "total_vecs": 5,
                        "dimensions": 6,
                    },
                )
            if path.endswith("/compact"):
                return httpx.Response(204)
            if method == "POST" and path.startswith("/v1/vectors/"):
                namespace = path.rsplit("/", 1)[-1]
                return httpx.Response(
                    200,
                    json={"namespace": namespace, "results": [{"id": "a", "score": 0.1}]},
                )
            return httpx.Response(204)

        self.patcher = mock.patch.object(httpx.Client, "request", new=handler)
        self.patcher.start()

    def tearDown(self) -> None:
        self.patcher.stop()

    def test_explicit_namespace_on_methods(self) -> None:
        client = TidepoolClient(default_namespace="default")
        client.upsert([Document(id="doc-1", vector=[0.1, 0.2, 0.3])], namespace="products")
        response = client.query([0.1, 0.2, 0.3], namespace="products")
        client.delete(["doc-1"], namespace="products")

        paths = [call["path"] for call in self.calls]
        self.assertEqual(response.namespace, "products")
        self.assertIn("/v1/vectors/products", paths)
        self.assertEqual(paths.count("/v1/vectors/products"), 3)

    def test_default_namespace_fallback(self) -> None:
        client = TidepoolClient(default_namespace="default")
        client.upsert([Document(id="doc-1", vector=[0.1, 0.2, 0.3])])

        self.assertIn("/v1/vectors/default", [call["path"] for call in self.calls])

    def test_get_namespace_status_and_compact(self) -> None:
        client = TidepoolClient(default_namespace="default")
        status = client.get_namespace_status("products")
        client.compact("products")

        self.assertEqual(status.wal_entries, 3)
        self.assertEqual(status.dimensions, 6)
        paths = [call["path"] for call in self.calls]
        self.assertIn("/v1/namespaces/products/status", paths)
        self.assertIn("/v1/namespaces/products/compact", paths)

    def test_cross_namespace_isolation(self) -> None:
        client = TidepoolClient(default_namespace="default")
        response_a = client.query([0.1, 0.2, 0.3], namespace="tenant_a")
        response_b = client.query([0.1, 0.2, 0.3], namespace="tenant_b")

        self.assertEqual(response_a.namespace, "tenant_a")
        self.assertEqual(response_b.namespace, "tenant_b")
        paths = [call["path"] for call in self.calls]
        self.assertIn("/v1/vectors/tenant_a", paths)
        self.assertIn("/v1/vectors/tenant_b", paths)

    def test_text_only_query(self) -> None:
        client = TidepoolClient(default_namespace="default")
        response = client.query(text="machine learning", mode="text")

        self.assertEqual(response.namespace, "default")
        last_call = self.calls[-1]
        self.assertEqual(last_call["path"], "/v1/vectors/default")
        self.assertEqual(last_call["json"]["mode"], "text")
        self.assertEqual(last_call["json"]["text"], "machine learning")
        self.assertNotIn("vector", last_call["json"])

    def test_list_namespaces_returns_info(self) -> None:
        client = TidepoolClient(default_namespace="default")
        namespaces = client.list_namespaces()

        self.assertEqual(len(namespaces), 2)
        self.assertEqual(namespaces[0].namespace, "default")
        self.assertEqual(namespaces[0].approx_count, 10)
        self.assertEqual(namespaces[0].dimensions, 3)
        self.assertTrue(namespaces[0].pending_compaction)


if __name__ == "__main__":
    unittest.main()
