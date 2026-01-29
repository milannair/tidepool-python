from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union

Vector = List[float]

AttrValue = Union[
    None,
    bool,
    int,
    float,
    str,
    List["AttrValue"],
    Dict[str, "AttrValue"],
]


@dataclass
class Document:
    id: str
    vector: Vector
    text: Optional[str] = None
    attributes: Optional[Dict[str, AttrValue]] = None


@dataclass
class VectorResult:
    id: str
    score: float
    vector: Optional[Vector] = None
    attributes: Optional[Dict[str, AttrValue]] = None


@dataclass
class QueryResponse:
    results: List[VectorResult]
    namespace: str


class DistanceMetric(str, Enum):
    COSINE = "cosine_distance"
    EUCLIDEAN = "euclidean_squared"
    DOT_PRODUCT = "dot_product"


class QueryMode(str, Enum):
    VECTOR = "vector"
    TEXT = "text"
    HYBRID = "hybrid"


class FusionMode(str, Enum):
    BLEND = "blend"
    RRF = "rrf"


@dataclass
class NamespaceInfo:
    namespace: str
    approx_count: int
    dimensions: int
    pending_compaction: Optional[bool] = None


@dataclass
class NamespaceStatus:
    last_run: Optional[datetime]
    wal_files: int
    wal_entries: int
    segments: int
    total_vecs: int
    dimensions: int


@dataclass
class IngestStatus:
    last_run: Optional[datetime]
    wal_files: int
    wal_entries: int
    segments: int
    total_vecs: int
    dimensions: int
