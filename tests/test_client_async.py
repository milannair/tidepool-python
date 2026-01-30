import asyncio
import unittest
from unittest import mock

import httpx

from tidepool import AsyncTidepoolClient, Document
from tidepool.errors import ServiceUnavailableError, TidepoolError, ValidationError


class TidepoolClientAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_query_text_only(self) -> None:
        calls = []

        async def handler(client_self, method, path, json=None, **kwargs):
            calls.append({"method": method, "path": path, "json": json})
            return httpx.Response(
                200, json={"namespace": "docs", "results": [{"id": "a", "score": 0.1}]}
            )

        with mock.patch.object(httpx.AsyncClient, "request", new=handler):
            client = AsyncTidepoolClient(default_namespace="docs")
            response = await client.query(text="hello", mode="text")
            await client.close()

            self.assertEqual(response.namespace, "docs")
            payload = calls[-1]["json"]
            self.assertEqual(payload["mode"], "text")
            self.assertEqual(payload["text"], "hello")
            self.assertNotIn("vector", payload)

    async def test_async_retries_on_503(self) -> None:
        calls = {"count": 0}

        async def handler(client_self, method, path, json=None, **kwargs):
            calls["count"] += 1
            if calls["count"] < 2:
                return httpx.Response(503, json={"error": "busy"})
            return httpx.Response(200, json={"ok": True})

        async def noop_sleep(_):
            return None

        with mock.patch.object(httpx.AsyncClient, "request", new=handler), mock.patch.object(
            asyncio, "sleep", new=noop_sleep
        ):
            client = AsyncTidepoolClient()
            result = await client._request_json(client._query_client, "GET", "/health")
            await client.close()

            self.assertEqual(result["ok"], True)
            self.assertEqual(calls["count"], 2)

    async def test_async_error_mapping(self) -> None:
        async def handler(client_self, method, path, json=None, **kwargs):
            if path == "/v1/vectors/default":
                return httpx.Response(400, json={"error": "bad"})
            return httpx.Response(503, json={"error": "down"})

        with mock.patch.object(httpx.AsyncClient, "request", new=handler):
            client = AsyncTidepoolClient()
            client.max_retries = 0
            with self.assertRaises(ValidationError):
                await client.query(vector=[0.1, 0.2, 0.3])
            with self.assertRaises(ServiceUnavailableError):
                await client.status()
            await client.close()

    async def test_async_invalid_json_response(self) -> None:
        async def handler(client_self, method, path, json=None, **kwargs):
            return httpx.Response(200, content=b"not-json", headers={"content-type": "application/json"})

        with mock.patch.object(httpx.AsyncClient, "request", new=handler):
            client = AsyncTidepoolClient()
            with self.assertRaises(TidepoolError):
                await client._request_json(client._query_client, "GET", "/health")
            await client.close()

    async def test_async_upsert_validation(self) -> None:
        client = AsyncTidepoolClient()
        with self.assertRaises(ValidationError):
            await client.upsert([Document(id="doc-1", vector=[])] )
        await client.close()


if __name__ == "__main__":
    unittest.main()
