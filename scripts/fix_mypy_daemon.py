#!/usr/bin/env python3
"""
Fix MyPy Daemon Executable Issue

This script resolves the warning:
"The mypy daemon executable ('dmypy') was not found on your PATH. 
Please install mypy or adjust the mypy.dmypyExecutable setting."

It will:
1. Detect or create virtual environment
2. Install/upgrade mypy in the virtual environment
3. Verify dmypy is accessible
4. Create VS Code settings if needed
5. Test the installation
"""

import os
import sys
import subprocess
import json
import shutil
from pathlib import Path


def run_command(command, capture_output=True, check=True, cwd=None):
    """Run a shell command and return the result."""
    try:
        if isinstance(command, str):
            command = command.split()
        
        result = subprocess.run(
            command, 
            capture_output=capture_output, 
            text=True, 
            check=check,
            cwd=cwd
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running command: {' '.join(command)}")
        print(f"Error: {e}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"Stderr: {e.stderr}")
        return None


def find_virtual_environment():
    """Find existing virtual environment in common locations."""
    print("🔍 Looking for existing virtual environment...")
    
    # Common virtual environment names and locations
    common_venv_names = [
        "venv",
        ".venv", 
        "env",
        ".env",
        "virtualenv",
        ".virtualenv"
    ]
    
    project_root = Path.cwd()
    
    for venv_name in common_venv_names:
        venv_path = project_root / venv_name
        if venv_path.exists() and venv_path.is_dir():
            # Check if it's a valid virtual environment
            if os.name == 'nt':  # Windows
                python_exe = venv_path / "Scripts" / "python.exe"
                pip_exe = venv_path / "Scripts" / "pip.exe"
                activate_script = venv_path / "Scripts" / "activate.bat"
            else:  # Unix/Linux/macOS
                python_exe = venv_path / "bin" / "python"
                pip_exe = venv_path / "bin" / "pip"
                activate_script = venv_path / "bin" / "activate"
            
            if python_exe.exists():
                print(f"✅ Found virtual environment: {venv_path}")
                return str(venv_path), str(python_exe), str(pip_exe), str(activate_script)
    
    print("❌ No virtual environment found")
    return None, None, None, None


def create_virtual_environment():
    """Create a new virtual environment."""
    print("🐍 Creating new virtual environment...")
    
    project_root = Path.cwd()
    venv_path = project_root / "venv"
    
    # Create virtual environment
    result = run_command([sys.executable, "-m", "venv", str(venv_path)])
    
    if result and result.returncode == 0:
        print(f"✅ Virtual environment created: {venv_path}")
        
        # Get paths for the new virtual environment
        if os.name == 'nt':  # Windows
            python_exe = venv_path / "Scripts" / "python.exe"
            pip_exe = venv_path / "Scripts" / "pip.exe"
            activate_script = venv_path / "Scripts" / "activate.bat"
        else:  # Unix/Linux/macOS
            python_exe = venv_path / "bin" / "python"
            pip_exe = venv_path / "bin" / "pip"
            activate_script = venv_path / "bin" / "activate"
        
        # Upgrade pip in the new virtual environment
        print("📦 Upgrading pip in virtual environment...")
        upgrade_result = run_command([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"])
        
        if upgrade_result and upgrade_result.returncode == 0:
            print("✅ Pip upgraded successfully")
        
        return str(venv_path), str(python_exe), str(pip_exe), str(activate_script)
    else:
        print("❌ Failed to create virtual environment")
        return None, None, None, None


def get_virtual_environment():
    """Get or create virtual environment."""
    # First, check if we're already in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Already in a virtual environment")
        python_exe = sys.executable
        pip_exe = str(Path(python_exe).parent / "pip")
        if os.name == 'nt':
            pip_exe += ".exe"
        return None, python_exe, pip_exe, None
    
    # Look for existing virtual environment
    venv_path, python_exe, pip_exe, activate_script = find_virtual_environment()
    
    if venv_path:
        return venv_path, python_exe, pip_exe, activate_script
    
    # Create new virtual environment if none found
    return create_virtual_environment()


def check_python_executable(python_exe):
    """Check which Python executable we're using."""
    print(f"🐍 Using Python: {python_exe}")
    
    # Test if the Python executable works
    result = run_command([python_exe, "--version"])
    if result and result.returncode == 0:
        print(f"✅ Python version: {result.stdout.strip()}")
        return True
    else:
        print("❌ Python executable not working")
        return False


def install_mypy(python_exe, pip_exe):
    """Install or upgrade mypy."""
    print("📦 Installing/upgrading mypy...")
    
    if not check_python_executable(python_exe):
        return False
    
    # Try to install mypy
    result = run_command([pip_exe, "install", "--upgrade", "mypy"])
    
    if result and result.returncode == 0:
        print("✅ MyPy installed/upgraded successfully")
        return True
    else:
        print("❌ Failed to install mypy")
        # Try alternative installation method
        print("🔄 Trying alternative installation method...")
        alt_result = run_command([python_exe, "-m", "pip", "install", "--upgrade", "mypy"])
        
        if alt_result and alt_result.returncode == 0:
            print("✅ MyPy installed/upgraded successfully (alternative method)")
            return True
        else:
            print("❌ Failed to install mypy with alternative method")
            return False


def find_dmypy_executable(venv_path=None):
    """Find the dmypy executable."""
    print("🔍 Looking for dmypy executable...")
    
    # Check if dmypy is in PATH
    dmypy_path = shutil.which('dmypy')
    if dmypy_path:
        print(f"✅ Found dmypy in PATH: {dmypy_path}")
        return dmypy_path
    
    # Check in virtual environment first if provided
    if venv_path:
        venv_path = Path(venv_path)
        if os.name == 'nt':  # Windows
            venv_dmypy = venv_path / "Scripts" / "dmypy.exe"
        else:  # Unix/Linux/macOS
            venv_dmypy = venv_path / "bin" / "dmypy"
        
        if venv_dmypy.exists():
            print(f"✅ Found dmypy in virtual environment: {venv_dmypy}")
            return str(venv_dmypy)
    
    # Check in Python scripts directory
    python_exe = sys.executable
    python_dir = Path(python_exe).parent
    
    # Common locations for dmypy
    possible_paths = [
        python_dir / "dmypy",
        python_dir / "dmypy.exe",
        python_dir / "Scripts" / "dmypy",
        python_dir / "Scripts" / "dmypy.exe",
        Path.home() / ".local" / "bin" / "dmypy",
    ]
    
    for path in possible_paths:
        if path.exists():
            print(f"✅ Found dmypy at: {path}")
            return str(path)
    
    print("❌ dmypy executable not found")
    return None


def test_dmypy(dmypy_path=None):
    """Test if dmypy is working."""
    print("🧪 Testing dmypy...")
    
    if dmypy_path:
        result = run_command([dmypy_path, "--help"], check=False)
    else:
        result = run_command(["dmypy", "--help"], check=False)
    
    if result and result.returncode == 0:
        print("✅ dmypy is working correctly")
        return True
    else:
        print("❌ dmypy test failed")
        return False


def create_vscode_settings(dmypy_path=None, python_exe=None):
    """Create or update VS Code settings for mypy."""
    print("⚙️  Configuring VS Code settings...")
    
    vscode_dir = Path(".vscode")
    settings_file = vscode_dir / "settings.json"
    
    # Create .vscode directory if it doesn't exist
    vscode_dir.mkdir(exist_ok=True)
    
    # Load existing settings or create new
    settings = {}
    if settings_file.exists():
        try:
            with open(settings_file, 'r') as f:
                settings = json.load(f)
        except json.JSONDecodeError:
            print("⚠️  Invalid JSON in settings.json, creating new settings")
            settings = {}
    
    # Update mypy settings
    mypy_settings = {
        "python.linting.mypyEnabled": True,
        "python.linting.enabled": True,
        "mypy-type-checker.importStrategy": "fromEnvironment",
        "mypy-type-checker.args": [
            "--ignore-missing-imports",
            "--follow-imports=silent",
            "--show-column-numbers",
            "--strict-optional"
        ]
    }
    
    # Add dmypy executable path if found
    if dmypy_path:
        mypy_settings["mypy.dmypyExecutable"] = dmypy_path
    
    # Add Python interpreter path if in virtual environment
    if python_exe and "venv" in python_exe:
        mypy_settings["python.defaultInterpreterPath"] = python_exe
    
    settings.update(mypy_settings)
    
    # Write settings back
    with open(settings_file, 'w') as f:
        json.dump(settings, f, indent=2)
    
    print(f"✅ VS Code settings updated: {settings_file}")


def create_mypy_config():
    """Create a mypy.ini configuration file."""
    print("📝 Creating mypy configuration...")
    
    mypy_config = """[mypy]
# Global options
python_version = 3.8
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_incomplete_defs = True
check_untyped_defs = True
disallow_untyped_decorators = True
no_implicit_optional = True
warn_redundant_casts = True
warn_unused_ignores = True
warn_no_return = True
warn_unreachable = True
strict_equality = True

# Per-module options
[mypy-tests.*]
disallow_untyped_defs = False

[mypy-setup]
ignore_errors = True

# Third-party libraries without type hints
[mypy-numpy.*]
ignore_missing_imports = True

[mypy-pandas.*]
ignore_missing_imports = True

[mypy-matplotlib.*]
ignore_missing_imports = True

[mypy-sklearn.*]
ignore_missing_imports = True

[mypy-scipy.*]
ignore_missing_imports = True
"""
    
    with open("mypy.ini", "w") as f:
        f.write(mypy_config)
    
    print("✅ Created mypy.ini configuration file")


def add_to_requirements():
    """Add mypy to requirements-dev.txt or requirements.txt."""
    print("📋 Updating requirements files...")
    
    # Check for requirements-dev.txt first, then requirements.txt
    requirements_files = ["requirements-dev.txt", "requirements.txt"]
    
    for req_file in requirements_files:
        if Path(req_file).exists():
            with open(req_file, "r") as f:
                content = f.read()
            
            if "mypy" not in content:
                with open(req_file, "a") as f:
                    f.write("\n# Type checking\nmypy>=1.0.0\n")
                print(f"✅ Added mypy to {req_file}")
            else:
                print(f"✅ mypy already in {req_file}")
            break
    else:
        # Create requirements-dev.txt if no requirements file exists
        with open("requirements-dev.txt", "w") as f:
            f.write("# Development dependencies\nmypy>=1.0.0\npytest>=6.0.0\nblack>=21.0.0\nflake8>=3.9.0\n")
        print("✅ Created requirements-dev.txt with mypy")


def create_activation_script(venv_path, activate_script):
    """Create a convenient activation script."""
    if not venv_path or not activate_script:
        return
    
    print("📝 Creating activation helper script...")
    
    if os.name == 'nt':  # Windows
        script_content = f"""@echo off
echo Activating virtual environment...
call "{activate_script}"
echo ✅ Virtual environment activated!
echo To deactivate, run: deactivate
cmd /k
"""
        script_file = "activate_venv.bat"
    else:  # Unix/Linux/macOS
        script_content = f"""#!/bin/bash
echo "Activating virtual environment..."
source "{activate_script}"
echo "✅ Virtual environment activated!"
echo "To deactivate, run: deactivate"
exec "$SHELL"
"""
        script_file = "activate_venv.sh"
    
    with open(script_file, 'w') as f:
        f.write(script_content)
    
    if os.name != 'nt':
        # Make the script executable on Unix systems
        os.chmod(script_file, 0o755)
    
    print(f"✅ Created activation script: {script_file}")


def main():
    """Main function to fix mypy daemon issues."""
    print("🔧 MyPy Daemon Fix Script")
    print("=" * 40)
    
    # Step 1: Get or create virtual environment
    venv_path, python_exe, pip_exe, activate_script = get_virtual_environment()
    
    if not python_exe or not pip_exe:
        print("❌ Failed to set up virtual environment. Exiting.")
        sys.exit(1)
    
    # Step 2: Install mypy
    if not install_mypy(python_exe, pip_exe):
        print("❌ Failed to install mypy. Exiting.")
        sys.exit(1)
    
    # Step 3: Find dmypy executable
    dmypy_path = find_dmypy_executable(venv_path)
    
    # Step 4: Test dmypy
    dmypy_working = test_dmypy(dmypy_path)
    if not dmypy_working and dmypy_path:
        # If dmypy test fails but we found the path, try using the full path
        dmypy_working = test_dmypy(dmypy_path)
    
    # Step 5: Configure VS Code
    create_vscode_settings(dmypy_path, python_exe)
    
    # Step 6: Create mypy config
    create_mypy_config()
    
    # Step 7: Update requirements
    add_to_requirements()
    
    # Step 8: Create activation script if virtual environment was created
    if venv_path and activate_script:
        create_activation_script(venv_path, activate_script)
    
    print("\n🎉 MyPy daemon fix completed!")
    print("\n📋 Summary:")
    print("- Virtual environment set up")
    print("- MyPy installed/upgraded")
    print("- VS Code settings configured")
    print("- mypy.ini configuration created")
    print("- Requirements updated")
    
    if dmypy_path:
        print(f"- dmypy executable found at: {dmypy_path}")
    else:
        print("- ⚠️  dmypy executable not found in PATH")
    
    if venv_path:
        print(f"- Virtual environment: {venv_path}")
        if os.name == 'nt':
            print("- Activation script: activate_venv.bat")
        else:
            print("- Activation script: activate_venv.sh")
    
    print("\n🔄 Next steps:")
    if venv_path and not (hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)):
        if os.name == 'nt':
            print("1. Run: activate_venv.bat")
        else:
            print(f"1. Run: source {activate_script}")
            print("   Or: ./activate_venv.sh")
    
    print("2. Restart VS Code")
    print("3. Reload the Python extension")
    print("4. Try running: dmypy --help")
    
    if dmypy_working:
        print("\n✅ dmypy should now work correctly!")
    else:
        print("\n⚠️  dmypy may need manual configuration")


if __name__ == "__main__":
    main()
