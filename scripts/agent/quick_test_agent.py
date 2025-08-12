#!/usr/bin/env python3
"""Relocated quick test agent script."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)


def test_agent_relocated() -> bool:
    """Lightweight inline test (copied to avoid circular import)."""
    try:
        print("🧪 Testing ML Agent Workflow System (relocated)...")
        from machine_learning_model.workflow.ml_agent import MLAgent  # type: ignore
        agent = MLAgent('test_project', '.', auto_save=False)
        recommendations = agent.get_recommendations()
        completed, total, progress = agent.get_progress()
        _ = agent.get_workflow_summary()
        print(f"✅ Steps: {len(agent.steps)} | Recs: {len(recommendations)} | Progress: {completed}/{total} ({progress:.1f}%)")
        print("🎉 Agent quick test passed")
        return True
    except Exception as exc:  # pragma: no cover - diagnostic
        print(f"❌ Agent quick test failed: {exc}")
        return False


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(0 if test_agent_relocated() else 1)
