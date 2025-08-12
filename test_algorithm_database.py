#!/usr/bin/env python3
"""Test script to verify algorithm database and categorization works correctly."""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def test_algorithm_database():
    """Test that algorithm database contains proper categorized information."""
    try:
        print("Testing algorithm database...")
        from machine_learning_model.gui.main_window_pyqt6 import AlgorithmDatabase
        
        # Test supervised algorithms
        supervised = AlgorithmDatabase.SUPERVISED_ALGORITHMS
        print(f"✅ Supervised algorithms loaded: {len(supervised)} algorithms")
        for name, info in supervised.items():
            assert 'description' in info, f"Missing description for {name}"
            assert 'use_cases' in info, f"Missing use_cases for {name}"
            assert 'pros' in info, f"Missing pros for {name}"
            assert 'cons' in info, f"Missing cons for {name}"
            assert 'status' in info, f"Missing status for {name}"
            print(f"  ✅ {name}: {info['status']}")
        
        # Test unsupervised algorithms
        unsupervised = AlgorithmDatabase.UNSUPERVISED_ALGORITHMS
        print(f"✅ Unsupervised algorithms loaded: {len(unsupervised)} algorithms")
        for name, info in unsupervised.items():
            assert 'description' in info, f"Missing description for {name}"
            print(f"  ✅ {name}: {info['status']}")
        
        # Test semi-supervised algorithms
        semi_supervised = AlgorithmDatabase.SEMI_SUPERVISED_ALGORITHMS
        print(f"✅ Semi-supervised algorithms loaded: {len(semi_supervised)} algorithms")
        for name, info in semi_supervised.items():
            assert 'description' in info, f"Missing description for {name}"
            print(f"  ✅ {name}: {info['status']}")
        
        total_algorithms = len(supervised) + len(unsupervised) + len(semi_supervised)
        print(f"✅ Total algorithms in database: {total_algorithms}")
        
        return True
        
    except Exception as e:
        print(f"❌ Algorithm database test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_implementation_status():
    """Test implementation status categorization."""
    try:
        print("Testing implementation status categorization...")
        from machine_learning_model.gui.main_window_pyqt6 import AlgorithmDatabase
        
        complete_count = 0
        ready_count = 0 
        planned_count = 0
        future_count = 0
        
        all_algorithms = {
            **AlgorithmDatabase.SUPERVISED_ALGORITHMS,
            **AlgorithmDatabase.UNSUPERVISED_ALGORITHMS,
            **AlgorithmDatabase.SEMI_SUPERVISED_ALGORITHMS
        }
        
        for name, info in all_algorithms.items():
            status = info.get('implementation_status', 'unknown')
            if status == 'complete':
                complete_count += 1
            elif status == 'ready':
                ready_count += 1
            elif status == 'planned':
                planned_count += 1
            elif status == 'future':
                future_count += 1
        
        print(f"  ✅ Complete implementations: {complete_count}")
        print(f"  ✅ Ready for implementation: {ready_count}")
        print(f"  ✅ Planned implementations: {planned_count}")
        print(f"  ✅ Future implementations: {future_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Implementation status test failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Testing Algorithm Database and Categorization...")
    
    # Test algorithm database
    db_ok = test_algorithm_database()
    if not db_ok:
        print("💥 Database tests failed!")
        sys.exit(1)
    
    # Test implementation status
    status_ok = test_implementation_status()
    if not status_ok:
        print("💥 Implementation status tests failed!")
        sys.exit(1)
    
    print("🎉 All algorithm database tests passed!")
    print("💡 The PyQt6 GUI should display all algorithms correctly in their categories.")
    sys.exit(0)
