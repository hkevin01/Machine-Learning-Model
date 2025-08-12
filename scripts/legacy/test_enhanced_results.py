#!/usr/bin/env python3
"""DEPRECATED: This file has been moved to scripts/testing/test_enhanced_results.py"""

import os
import subprocess
import sys

print("⚠️  DEPRECATION WARNING: test_enhanced_results.py has been moved!")
print("   New location: scripts/testing/test_enhanced_results.py")
print("   Redirecting to new location...")
print()

# Redirect to the new location
script_path = os.path.join(os.path.dirname(__file__), 'scripts', 'testing', 'test_enhanced_results.py')
if os.path.exists(script_path):
    sys.exit(subprocess.call([sys.executable, script_path] + sys.argv[1:]))
else:
    print(f"ERROR: {script_path} not found!")
    print("Please run: python scripts/testing/test_enhanced_results.py")
    sys.exit(1)
