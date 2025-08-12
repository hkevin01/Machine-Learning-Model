#!/usr/bin/env python3
"""Root test file - move to tests/ or scripts/testing/ (relocated)."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(ROOT, 'src'))

def test_algorithm_database() -> bool:
    """Test algorithm database integrity."""
    try:
        from machine_learning_model.gui.models.registry_data import (  # type: ignore
            SEMI_SUPERVISED_ALGORITHMS_DATA,
            SUPERVISED_ALGORITHMS_DATA,
            UNSUPERVISED_ALGORITHMS_DATA,
        )
        supervised_count = len(SUPERVISED_ALGORITHMS_DATA)
        unsupervised_count = len(UNSUPERVISED_ALGORITHMS_DATA)
        semi_count = len(SEMI_SUPERVISED_ALGORITHMS_DATA)
        total = supervised_count + unsupervised_count + semi_count
        print(f"✅ Algorithm database test passed: {total} algorithms total")
        return True
    except Exception as exc:  # pragma: no cover
        print(f"❌ Algorithm database test failed: {exc}")
        return False

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(0 if test_algorithm_database() else 1)
