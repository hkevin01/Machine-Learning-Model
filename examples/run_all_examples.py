"""
Run all available algorithm examples.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_all_examples():
    """Run all example scripts."""
    
    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
    
    examples_dir = Path(__file__).parent / 'supervised_examples'
    
    example_files = [
        'decision_tree_example.py',
        'random_forest_example.py',
        'linear_regression_example.py',
        'logistic_regression_example.py'
    ]
    
    print("🚀 Running All Algorithm Examples")
    print("=" * 50)
    
    results = {}
    
    for example_file in example_files:
        example_path = examples_dir / example_file
        
        if example_path.exists():
            print(f"\n🔄 Running {example_file}...")
            
            try:
                result = subprocess.run(
                    [sys.executable, str(example_path)],
                    capture_output=True,
                    text=True,
                    cwd=str(Path(__file__).parent.parent)
                )
                
                if result.returncode == 0:
                    print(f"✅ {example_file} completed successfully")
                    results[example_file] = "SUCCESS"
                else:
                    print(f"❌ {example_file} failed")
                    print(f"Error: {result.stderr[:200]}...")
                    results[example_file] = "FAILED"
                    
            except Exception as e:
                print(f"❌ {example_file} failed with exception: {e}")
                results[example_file] = "ERROR"
        else:
            print(f"⚠️ {example_file} not found")
            results[example_file] = "NOT_FOUND"
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SUMMARY")
    print("=" * 50)
    
    success_count = sum(1 for status in results.values() if status == "SUCCESS")
    total_count = len(example_files)
    
    for example_file, status in results.items():
        status_icon = {"SUCCESS": "✅", "FAILED": "❌", "ERROR": "💥", "NOT_FOUND": "⚠️"}
        print(f"{status_icon.get(status, '?')} {example_file}: {status}")
    
    print(f"\n🎯 Results: {success_count}/{total_count} examples ran successfully")
    
    if success_count == total_count:
        print("🎉 All examples completed successfully!")
    
    print("\n📁 Check 'test-outputs/artifacts/' for generated visualizations")

if __name__ == "__main__":
    # Create output directories
    os.makedirs('test-outputs/artifacts', exist_ok=True)
    
    run_all_examples()
