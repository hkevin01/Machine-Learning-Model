"""Algorithm metadata registry.

This module centralizes descriptive metadata for algorithms used by the GUI.
It is pure-data / lightweight logic so it can be imported in tests without
triggering heavy GUI dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass

try:  # pragma: no cover - simple import fallback
    from . import registry_data  # type: ignore
except ImportError:  # pragma: no cover
    # Fallback if running in an environment where relative import fails due to path issues.
    import importlib
    registry_data = importlib.import_module('machine_learning_model.gui.models.registry_data')  # type: ignore


@dataclass(slots=True)
class AlgorithmInfo:
    """Metadata container for algorithm descriptions."""

    name: str
    description: str
    use_cases: str
    pros: str
    cons: str
    complexity: str
    type: str  # Task applicability (Classification, Regression, Both, Clustering, Dimensionality Reduction)
    status: str
    implementation_status: str
    examples: str

    def short_summary(self) -> str:  # pragma: no cover - trivial helper
        text = self.description
        if len(text) > 120:
            text = f"{text[:120]}..."
        return f"{self.name}: {text}"


def _build_registry() -> dict[str, AlgorithmInfo]:
    combined: dict[str, AlgorithmInfo] = {}
    for source in (
        registry_data.SUPERVISED_ALGORITHMS_DATA,
        registry_data.UNSUPERVISED_ALGORITHMS_DATA,
        registry_data.SEMI_SUPERVISED_ALGORITHMS_DATA,
    ):
        for name, meta in source.items():
            combined[name] = AlgorithmInfo(
                name=name,
                description=meta.get("description", ""),
                use_cases=meta.get("use_cases", ""),
                pros=meta.get("pros", ""),
                cons=meta.get("cons", ""),
                complexity=meta.get("complexity", "Unknown"),
                type=meta.get("type", "Unknown"),
                status=meta.get("status", ""),
                implementation_status=meta.get("implementation_status", "unknown"),
                examples=meta.get("examples", ""),
            )
    return combined


ALGORITHM_REGISTRY: dict[str, AlgorithmInfo] = _build_registry()


def list_algorithms(kind: str | None = None) -> list[str]:
    if kind is None:
        return sorted(ALGORITHM_REGISTRY.keys())
    kind_lower = kind.lower()
    return sorted(
        name
        for name, info in ALGORITHM_REGISTRY.items()
        if info.type.lower() == kind_lower or info.type.lower() == "both"
    )


def get_algorithm_info(name: str) -> AlgorithmInfo | None:
    return ALGORITHM_REGISTRY.get(name)


__all__ = [
    "AlgorithmInfo",
    "ALGORITHM_REGISTRY",
    "list_algorithms",
    "get_algorithm_info",
]
