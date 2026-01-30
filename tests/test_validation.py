import math
import unittest

from tidepool.client import (
    _document_to_payload,
    _normalize_alpha,
    _normalize_distance_metric,
    _normalize_fusion_mode,
    _normalize_namespace,
    _normalize_query_mode,
    _normalize_query_text,
    _validate_attributes,
    _validate_documents,
    _validate_filters,
    _validate_ids,
    _validate_positive_int,
    _validate_vector,
)
from tidepool.errors import ValidationError
from tidepool.types import Document, DistanceMetric, FusionMode, QueryMode


class ValidationHelpersTests(unittest.TestCase):
    def test_normalize_namespace(self) -> None:
        self.assertEqual(_normalize_namespace(None, "default"), "default")
        self.assertEqual(_normalize_namespace("  tenant ", "default"), "tenant")
        with self.assertRaises(ValidationError):
            _normalize_namespace("", "default")
        with self.assertRaises(ValidationError):
            _normalize_namespace("   ", "default")

    def test_normalize_distance_metric(self) -> None:
        self.assertEqual(
            _normalize_distance_metric(DistanceMetric.COSINE),
            DistanceMetric.COSINE.value,
        )
        self.assertEqual(
            _normalize_distance_metric("dot_product"),
            DistanceMetric.DOT_PRODUCT.value,
        )
        self.assertIsNone(_normalize_distance_metric(None))
        with self.assertRaises(ValidationError):
            _normalize_distance_metric("bad")
        with self.assertRaises(ValidationError):
            _normalize_distance_metric(123)

    def test_normalize_query_text(self) -> None:
        self.assertIsNone(_normalize_query_text(None))
        self.assertIsNone(_normalize_query_text("   "))
        self.assertEqual(_normalize_query_text("  hello "), "hello")
        with self.assertRaises(ValidationError):
            _normalize_query_text(123)

    def test_normalize_query_mode(self) -> None:
        self.assertEqual(
            _normalize_query_mode(None, has_vector=True, has_text=False),
            QueryMode.VECTOR.value,
        )
        self.assertEqual(
            _normalize_query_mode(None, has_vector=False, has_text=True),
            QueryMode.TEXT.value,
        )
        self.assertEqual(
            _normalize_query_mode(None, has_vector=True, has_text=True),
            QueryMode.HYBRID.value,
        )
        self.assertEqual(
            _normalize_query_mode(QueryMode.TEXT, has_vector=True, has_text=True),
            QueryMode.TEXT.value,
        )
        with self.assertRaises(ValidationError):
            _normalize_query_mode("bad", has_vector=True, has_text=False)
        with self.assertRaises(ValidationError):
            _normalize_query_mode(123, has_vector=True, has_text=False)

    def test_normalize_fusion_mode(self) -> None:
        self.assertEqual(_normalize_fusion_mode(FusionMode.BLEND), FusionMode.BLEND.value)
        self.assertEqual(_normalize_fusion_mode("rrf"), FusionMode.RRF.value)
        self.assertIsNone(_normalize_fusion_mode(None))
        with self.assertRaises(ValidationError):
            _normalize_fusion_mode("bad")
        with self.assertRaises(ValidationError):
            _normalize_fusion_mode(123)

    def test_normalize_alpha(self) -> None:
        self.assertEqual(_normalize_alpha(1.5), 1.0)
        self.assertEqual(_normalize_alpha(-1), 0.0)
        self.assertEqual(_normalize_alpha(0.3), 0.3)
        with self.assertRaises(ValidationError):
            _normalize_alpha(math.nan)
        with self.assertRaises(ValidationError):
            _normalize_alpha("bad")

    def test_validate_vector(self) -> None:
        with self.assertRaises(ValidationError):
            _validate_vector("bad")
        with self.assertRaises(ValidationError):
            _validate_vector([])
        with self.assertRaises(ValidationError):
            _validate_vector(["a"])
        with self.assertRaises(ValidationError):
            _validate_vector([1.0, 2.0], expected_dims=3)
        result = _validate_vector((1.0, 2.0), expected_dims=2)
        self.assertEqual(result, [1.0, 2.0])

    def test_validate_attributes_and_filters(self) -> None:
        _validate_attributes({"a": True, "b": [1, "x"], "c": {"d": 2}})
        _validate_filters({"a": False, "b": ["x", "y"], "c": {"d": None}})
        with self.assertRaises(ValidationError):
            _validate_attributes(["bad"])
        with self.assertRaises(ValidationError):
            _validate_filters({1: "bad"})
        with self.assertRaises(ValidationError):
            _validate_attributes({"a": {"b": set([1])}})

    def test_validate_ids(self) -> None:
        with self.assertRaises(ValidationError):
            _validate_ids("bad")
        with self.assertRaises(ValidationError):
            _validate_ids([])
        with self.assertRaises(ValidationError):
            _validate_ids([""])
        self.assertEqual(_validate_ids(["a", "b"]), ["a", "b"])

    def test_validate_documents(self) -> None:
        doc = Document(id="doc-1", vector=[1.0, 2.0])
        self.assertEqual(_validate_documents([doc])[0].id, "doc-1")
        with self.assertRaises(ValidationError):
            _validate_documents("bad")
        with self.assertRaises(ValidationError):
            _validate_documents([])
        with self.assertRaises(ValidationError):
            _validate_documents([Document(id="", vector=[1.0])])
        with self.assertRaises(ValidationError):
            _validate_documents([Document(id="doc-1", vector=[1.0]), Document(id="doc-2", vector=[1.0, 2.0])])
        with self.assertRaises(ValidationError):
            _validate_documents([Document(id="doc-1", vector=[1.0], text=123)])

    def test_validate_positive_int(self) -> None:
        self.assertIsNone(_validate_positive_int(None, "n"))
        self.assertEqual(_validate_positive_int(3, "n"), 3)
        with self.assertRaises(ValidationError):
            _validate_positive_int(0, "n")
        with self.assertRaises(ValidationError):
            _validate_positive_int(-1, "n")
        with self.assertRaises(ValidationError):
            _validate_positive_int(1.2, "n")

    def test_document_to_payload(self) -> None:
        doc = Document(
            id="doc-1",
            vector=[0.1, 0.2],
            text="hello",
            attributes={"tag": "a"},
        )
        payload = _document_to_payload(doc)
        self.assertEqual(payload["id"], "doc-1")
        self.assertEqual(payload["text"], "hello")
        self.assertEqual(payload["attributes"], {"tag": "a"})


if __name__ == "__main__":
    unittest.main()
