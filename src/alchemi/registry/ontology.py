from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class OntologyNode:
    node_id: str
    name: str
    parent_id: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


_NODE_LIST = [
    OntologyNode("solids", "Solids", None),
    OntologyNode("carbonates", "Carbonates", "solids"),
    OntologyNode("clays", "Clays", "solids"),
    OntologyNode("sulfates", "Sulfates", "solids"),
    OntologyNode("silicates", "Silicates", "solids"),
    OntologyNode("oxides", "Oxides", "solids"),
    OntologyNode("organics", "Organics", "solids"),
    OntologyNode("calcite", "Calcite", "carbonates"),
    OntologyNode("dolomite", "Dolomite", "carbonates"),
    OntologyNode("magnesite", "Magnesite", "carbonates"),
    OntologyNode("kaolinite", "Kaolinite", "clays"),
    OntologyNode("montmorillonite", "Montmorillonite", "clays"),
    OntologyNode("illite", "Illite", "clays"),
    OntologyNode("gypsum", "Gypsum", "sulfates"),
    OntologyNode("jarosite", "Jarosite", "sulfates"),
    OntologyNode("epidote", "Epidote", "silicates"),
    OntologyNode("olivine", "Olivine", "silicates"),
    OntologyNode("hematite", "Hematite", "oxides"),
    OntologyNode("goethite", "Goethite", "oxides"),
    OntologyNode("asphalt", "Asphalt", "organics"),
    OntologyNode("charcoal", "Charcoal", "organics"),
]

_NODES: dict[str, OntologyNode] = {n.node_id: n for n in _NODE_LIST}


def get_node(node_id: str) -> OntologyNode:
    try:
        return _NODES[node_id]
    except KeyError as exc:  # pragma: no cover - defensive guard
        msg = f"Unknown ontology node: {node_id!r}"
        raise KeyError(msg) from exc


def get_children(node_id: str) -> list[OntologyNode]:
    return [node for node in _NODE_LIST if node.parent_id == node_id]


def get_path_to_root(node_id: str) -> list[OntologyNode]:
    path: list[OntologyNode] = []
    current_id: str | None = node_id
    while current_id is not None:
        node = get_node(current_id)
        path.append(node)
        current_id = node.parent_id
    return list(reversed(path))


def aggregate_leaves_to_family(predictions_per_leaf: Mapping[str, float]) -> dict[str, float]:
    """Aggregate leaf predictions to their immediate family nodes."""

    totals: dict[str, float] = {}
    for leaf_id, score in predictions_per_leaf.items():
        node = get_node(leaf_id)
        if node.parent_id is None:
            totals[leaf_id] = totals.get(leaf_id, 0.0) + float(score)
            continue
        totals[node.parent_id] = totals.get(node.parent_id, 0.0) + float(score)
    return totals


__all__ = [
    "OntologyNode",
    "get_node",
    "get_children",
    "get_path_to_root",
    "aggregate_leaves_to_family",
]
