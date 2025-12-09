"""Dataset coverage analysis utilities."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping


def _count_field(values: Iterable[str]) -> Mapping[str, int]:
    counter = Counter(values)
    return dict(counter)


def sensor_histogram(sensor_ids: Iterable[str]) -> Mapping[str, int]:
    """Histogram over sensor identifiers."""

    return _count_field(sensor_ids)


def geography_histogram(regions: Iterable[str]) -> Mapping[str, int]:
    """Histogram over coarse geography labels."""

    return _count_field(regions)


def landcover_histogram(landcover_classes: Iterable[str]) -> Mapping[str, int]:
    """Histogram over land cover types."""

    return _count_field(landcover_classes)


def ontology_coverage(ontology_terms: Iterable[str]) -> Mapping[str, float]:
    """Compute coverage percentages for ontology terms."""

    counts = Counter(ontology_terms)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {term: count / total for term, count in counts.items()}


def summarize_gaps(metadata: Mapping[str, Iterable[str]]) -> Mapping[str, object]:
    """Produce simple summaries of data gaps across metadata facets."""

    summaries = {
        "sensors": sensor_histogram(metadata.get("sensors", [])),
        "geography": geography_histogram(metadata.get("geography", [])),
        "landcover": landcover_histogram(metadata.get("landcover", [])),
    }
    ontology = metadata.get("ontology", [])
    if ontology:
        summaries["ontology"] = ontology_coverage(ontology)
    # Identify keys with no entries
    summaries["missing_fields"] = [
        key for key, values in metadata.items() if len(list(values)) == 0
    ]
    return summaries


__all__ = [
    "sensor_histogram",
    "geography_histogram",
    "landcover_histogram",
    "ontology_coverage",
    "summarize_gaps",
]
