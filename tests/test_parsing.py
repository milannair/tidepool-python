import unittest

import httpx

from tidepool.client import (
    _extract_error_message,
    _parse_datetime,
    _parse_ingest_status,
    _parse_namespace_info,
    _parse_namespace_status,
    _parse_namespaces,
    _parse_query_response,
    _parse_vector_results,
)
from tidepool.errors import TidepoolError


class ParsingHelpersTests(unittest.TestCase):
    def test_extract_error_message(self) -> None:
        response = httpx.Response(400, json={"error": "bad"})
        self.assertEqual(_extract_error_message(response), "bad")
        response = httpx.Response(400, json={"message": "nope"})
        self.assertEqual(_extract_error_message(response), "nope")
        response = httpx.Response(500, content=b"oh no")
        self.assertEqual(_extract_error_message(response), "oh no")
        response = httpx.Response(500, content=b"")
        self.assertEqual(_extract_error_message(response), "HTTP 500")

    def test_parse_datetime(self) -> None:
        parsed = _parse_datetime("2025-01-01T00:00:00Z")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.year, 2025)
        self.assertIsNone(_parse_datetime(None))

    def test_parse_vector_results(self) -> None:
        results = _parse_vector_results([
            {"id": "a", "score": 0.1},
            {"id": "b", "dist": 0.2},
            {"id": "c", "distance": 0.3},
        ])
        self.assertEqual(results[0].score, 0.1)
        self.assertEqual(results[1].score, 0.2)
        self.assertEqual(results[2].score, 0.3)

        results = _parse_vector_results({"results": [{"id": "a", "score": 0.4}]})
        self.assertEqual(results[0].id, "a")

        results = _parse_vector_results({"vectors": [{"id": "b", "score": 0.5}]})
        self.assertEqual(results[0].id, "b")

        with self.assertRaises(TidepoolError):
            _parse_vector_results({"bad": "shape"})
        with self.assertRaises(TidepoolError):
            _parse_vector_results(["bad"])

    def test_parse_namespace_info(self) -> None:
        info = _parse_namespace_info(
            {
                "namespace": "default",
                "approx_count": 10,
                "dimensions": 3,
                "pending_compaction": True,
            }
        )
        self.assertEqual(info.namespace, "default")
        self.assertTrue(info.pending_compaction)

        info = _parse_namespace_info(
            {
                "namespace": "default",
                "pendingCompaction": False,
            }
        )
        self.assertFalse(info.pending_compaction)

        info = _parse_namespace_info({"namespace": "default", "pending_compaction": "no"})
        self.assertIsNone(info.pending_compaction)

    def test_parse_namespaces(self) -> None:
        infos = _parse_namespaces(["a", "b"])
        self.assertEqual(infos[0].namespace, "a")

        infos = _parse_namespaces({"namespaces": [{"namespace": "c"}]})
        self.assertEqual(infos[0].namespace, "c")

        infos = _parse_namespaces({"namespace_list": ["d", "e"]})
        self.assertEqual(infos[1].namespace, "e")

        with self.assertRaises(TidepoolError):
            _parse_namespaces({"bad": "shape"})

    def test_parse_query_response(self) -> None:
        response = _parse_query_response(
            {"namespace": "ns", "results": [{"id": "a", "score": 0.1}]},
            "fallback",
        )
        self.assertEqual(response.namespace, "ns")
        self.assertEqual(response.results[0].id, "a")

        response = _parse_query_response(
            [{"id": "b", "score": 0.2}],
            "fallback",
        )
        self.assertEqual(response.namespace, "fallback")

    def test_parse_status(self) -> None:
        ingest = _parse_ingest_status(
            {
                "last_run": "2025-01-01T00:00:00Z",
                "wal_files": 1,
                "wal_entries": 2,
                "segments": 3,
                "total_vecs": 4,
                "dimensions": 5,
            }
        )
        self.assertEqual(ingest.wal_files, 1)
        namespace = _parse_namespace_status(
            {
                "last_run": "2025-01-01T00:00:00Z",
                "wal_files": 2,
                "wal_entries": 3,
                "segments": 4,
                "total_vecs": 5,
                "dimensions": 6,
            }
        )
        self.assertEqual(namespace.dimensions, 6)


if __name__ == "__main__":
    unittest.main()
