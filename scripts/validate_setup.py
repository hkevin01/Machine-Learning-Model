"""
Setup validation script for cross-platform compatibility.
This script validates that the ML framework setup works correctly.
"""

import importlib.util
import os
import platform
import subprocess
import sys
from pathlib import Path


def check_python_version():
    """Check Python version compatibility."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported. Please use Python 3.8+")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def check_virtual_environment():
    """Check if virtual environment is active."""
    print("\n🔧 Checking virtual environment...")
    
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment is active")
        print(f"   Environment path: {sys.prefix}")
        return True
    else:
        print("⚠️ Virtual environment is not active")
        print("   Consider activating with: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate.bat (Windows)")
        return False


def check_required_packages():
    """Check if required packages are installed."""
    print("\n📦 Checking required packages...")
    
    required_packages = [
        'numpy', 'pandas', 'scikit-learn', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            spec = importlib.util.find_spec(package)
            if spec is not None:
                print(f"✅ {package} is installed")
            else:
                print(f"❌ {package} is missing")
                missing_packages.append(package)
        except ImportError:
            print(f"❌ {package} is missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📋 To install missing packages, run:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True


def check_project_structure():
    """Check if project structure is correct."""
    print("\n📁 Checking project structure...")
    
    required_dirs = [
        'src/machine_learning_model',
        'tests',
        'docs',
        'examples',
        'scripts'
    ]
    
    required_files = [
        'src/machine_learning_model/__init__.py',
        'src/machine_learning_model/supervised/__init__.py',
        'run_gui.py'
    ]
    
    all_good = True
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ Directory {dir_path} exists")
        else:
            print(f"❌ Directory {dir_path} is missing")
            all_good = False
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ File {file_path} exists")
        else:
            print(f"❌ File {file_path} is missing")
            all_good = False
    
    return all_good


def check_platform_specific():
    """Check platform-specific requirements."""
    print(f"\n🖥️ Checking platform-specific requirements for {platform.system()}...")
    
    current_platform = platform.system()
    
    if current_platform == "Windows":
        # Check Windows-specific requirements
        batch_files = [
            'scripts/setup_windows.bat',
            'scripts/run_gui_windows.bat'
        ]
        
        for batch_file in batch_files:
            if Path(batch_file).exists():
                print(f"✅ Windows script {batch_file} exists")
            else:
                print(f"⚠️ Windows script {batch_file} is missing")
        
        # Check if tkinter works on Windows
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            root.destroy()
            print("✅ GUI (tkinter) is working")
        except Exception as e:
            print(f"❌ GUI (tkinter) test failed: {e}")
            return False
            
    elif current_platform == "Linux":
        # Check Linux-specific requirements
        shell_scripts = [
            'scripts/setup_ubuntu.sh',
            'scripts/run_gui.sh'
        ]
        
        for script in shell_scripts:
            script_path = Path(script)
            if script_path.exists():
                print(f"✅ Linux script {script} exists")
                if os.access(script_path, os.X_OK):
                    print(f"✅ Script {script} is executable")
                else:
                    print(f"⚠️ Script {script} is not executable (run: chmod +x {script})")
            else:
                print(f"⚠️ Linux script {script} is missing")
        
        # Check system dependencies
        try:
            import tkinter as tk
            print("✅ tkinter is available")
        except ImportError:
            print("❌ tkinter is not available. Install with: sudo apt install python3-tk")
            return False
    
    return True


def run_basic_tests():
    """Run basic functionality tests."""
    print("\n🧪 Running basic functionality tests...")
    
    try:
        # Test basic ML functionality
        from sklearn.datasets import make_classification

        from machine_learning_model.supervised.decision_tree import (
            DecisionTreeClassifier,
        )

        # Generate test data
        X, y = make_classification(n_samples=50, n_features=4, random_state=42)
        
        # Test Decision Tree
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X, y)
        predictions = clf.predict(X)
        
        if len(predictions) == len(y):
            print("✅ Decision Tree basic test passed")
        else:
            print("❌ Decision Tree basic test failed")
            return False
        
        # Test Random Forest if available
        try:
            from machine_learning_model.supervised.random_forest import (
                RandomForestClassifier,
            )
            rf = RandomForestClassifier(n_estimators=5, random_state=42)
            rf.fit(X, y)
            rf_predictions = rf.predict(X)
            
            if len(rf_predictions) == len(y):
                print("✅ Random Forest basic test passed")
            else:
                print("❌ Random Forest basic test failed")
                return False
        except ImportError:
            print("⚠️ Random Forest not available yet")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


def main():
    """Main validation function."""
    print("🔍 Machine Learning Framework Setup Validation")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Virtual Environment", check_virtual_environment),
        ("Required Packages", check_required_packages),
        ("Project Structure", check_project_structure),
        ("Platform Requirements", check_platform_specific),
        ("Basic Functionality", run_basic_tests)
    ]
    
    passed_checks = 0
    total_checks = len(checks)
    
    for check_name, check_function in checks:
        try:
            if check_function():
                passed_checks += 1
        except Exception as e:
            print(f"❌ {check_name} check failed with error: {e}")
    
    print("\n" + "=" * 50)
    print("📊 Validation Summary")
    print("=" * 50)
    print(f"Passed: {passed_checks}/{total_checks} checks")
    
    if passed_checks == total_checks:
        print("🎉 All checks passed! Your setup is ready to go.")
        print("\n🚀 Next steps:")
        print("   • Run the GUI: python run_gui.py")
        print("   • Run tests: python -m pytest tests/")
        print("   • Explore examples: check the examples/ directory")
        return True
    else:
        print("⚠️ Some checks failed. Please address the issues above.")
        print("\n🔧 Common solutions:")
        print("   • Activate virtual environment")
        print("   • Install missing packages: pip install -r requirements.txt")
        print("   • Check that you're in the correct project directory")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
