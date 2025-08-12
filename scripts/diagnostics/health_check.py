#!/usr/bin/env python3
"""Comprehensive environment & GUI/ML health diagnostics.

Performs:
 1. Python version + platform report
 2. Critical dependency import checks (PyQt6, numpy, pandas, sklearn)
 3. GUI headless initialization smoke test (QApplication)
 4. Basic ML workflow smoke (fit a tiny model)
 5. Performance timing summary
 6. Recommendations for detected issues

Exit codes:
 0 = all good
 1 = non-fatal warnings (still usable)
 2 = hard failure (missing critical component)
"""
from __future__ import annotations

import importlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

CRITICAL = ["PyQt6", "numpy", "pandas", "sklearn"]
OPTIONAL = ["xgboost", "mlflow"]

@dataclass
class CheckResult:
    name: str
    status: str
    detail: str = ""
    recommendation: str = ""


@dataclass
class HealthReport:
    python: str
    platform: str
    checks: list[CheckResult]
    elapsed: float
    overall_status: str
    recommendations: list[str]


def import_check(pkg: str) -> CheckResult:
    t0 = time.perf_counter()
    try:
        importlib.import_module(pkg)
        dt = time.perf_counter() - t0
        return CheckResult(pkg, "ok", f"import in {dt * 1000:.1f} ms")
    except ModuleNotFoundError as exc:  # pragma: no cover
        rec = "pip install -r requirements.txt" if pkg in CRITICAL else f"pip install {pkg}"  # noqa: E501
        return CheckResult(pkg, "missing", detail=str(exc), recommendation=rec)
    except Exception as exc:  # pragma: no cover  # fallback, keep broad for unexpected import errors
        return CheckResult(pkg, "fail", detail=str(exc), recommendation="Investigate import error")


def gui_smoke() -> CheckResult:
    try:
        from PyQt6.QtWidgets import QApplication  # pragma: no cover
        app = QApplication([])
        app.processEvents()
        app.quit()
        return CheckResult("PyQt6_GUI", "ok", "QApplication initialized")
    except ModuleNotFoundError as exc:  # pragma: no cover
        return CheckResult(
            "PyQt6_GUI",
            "missing",
            detail=str(exc),
            recommendation="pip install PyQt6 (or run inside provided Docker image)",
        )
    except Exception as exc:  # pragma: no cover  # unexpected GUI runtime failure
        return CheckResult(
            "PyQt6_GUI",
            "fail",
            detail=str(exc),
            recommendation="Ensure DISPLAY / X11 permissions or use --headless",
        )


def tiny_ml() -> CheckResult:
    try:
        from sklearn.datasets import load_iris  # pragma: no cover
        from sklearn.linear_model import LogisticRegression  # pragma: no cover
        from sklearn.model_selection import train_test_split  # pragma: no cover

        X, y = load_iris(return_X_y=True)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression(max_iter=200)
        model.fit(Xtr, ytr)
        acc = model.score(Xte, yte)
        return CheckResult("sklearn_basic_fit", "ok", f"accuracy={acc:.3f}")
    except ModuleNotFoundError as exc:  # pragma: no cover
        return CheckResult(
            "sklearn_basic_fit",
            "missing",
            detail=str(exc),
            recommendation="pip install scikit-learn",
        )
    except Exception as exc:  # pragma: no cover  # unexpected ML stack failure
        return CheckResult(
            "sklearn_basic_fit",
            "fail",
            detail=str(exc),
            recommendation="Check scikit-learn install and BLAS libraries",
        )


def aggregate_status(checks: list[CheckResult]) -> str:
    if any(c.status in {"fail", "missing"} and c.name in CRITICAL + ["PyQt6_GUI"] for c in checks):
        return "degraded"
    if any(c.status in {"fail", "missing"} for c in checks):
        return "warn"
    return "ok"


def collect_recommendations(checks: list[CheckResult]) -> list[str]:
    recs: list[str] = []
    for c in checks:
        if c.recommendation and c.status in {"missing", "fail"}:
            recs.append(f"{c.name}: {c.recommendation}")
    if not recs:
        recs.append("Environment healthy. No action needed.")
    return recs


def main() -> int:
    t0 = time.perf_counter()
    checks: list[CheckResult] = []

    # Imports
    for pkg in CRITICAL + OPTIONAL:
        checks.append(import_check(pkg))

    # GUI
    checks.append(gui_smoke())

    # Tiny ML
    checks.append(tiny_ml())

    elapsed = time.perf_counter() - t0
    overall = aggregate_status(checks)
    recs = collect_recommendations(checks)

    report = HealthReport(
        python=sys.version.split()[0],
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        checks=checks,
        elapsed=elapsed,
        overall_status=overall,
        recommendations=recs,
    )

    # Pretty print table
    print("== Environment Health Check ==")
    print(f"Python: {report.python}")
    print(f"Platform: {report.platform}")
    print(f"Overall Status: {report.overall_status}")
    print("")
    width = max(len(c.name) for c in checks) + 2
    for c in checks:
        print(f"{c.name.ljust(width)} {c.status.upper():7} {c.detail}")
    print("")
    print("Recommendations:")
    for r in recs:
        print(f" - {r}")

    # JSON (optional machine parsing)
    if os.environ.get("HEALTH_CHECK_JSON"):
        print("\nJSON:")
        print(json.dumps(asdict(report), indent=2))

    if overall == "ok":
        return 0
    if overall == "warn":
        return 1
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
