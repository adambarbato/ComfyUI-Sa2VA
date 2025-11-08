#!/usr/bin/env python3
"""
ComfyUI-Sa2VA Installation Script

This script handles the installation and setup of ComfyUI-Sa2VA,
including dependency management and compatibility checks.
"""

import importlib
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, check=True, capture_output=False):
    """Run a shell command with proper error handling."""
    try:
        if capture_output:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, check=check
            )
            return result.stdout.strip()
        else:
            result = subprocess.run(cmd, shell=True, check=check)
            return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {cmd}")
        print(f"   Error: {e}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported")
        print("   ComfyUI-Sa2VA requires Python 3.8 or newer")
        return False

    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def check_package_version(package_name, min_version=None):
    """Check if a package is installed and meets version requirements."""
    try:
        module = importlib.import_module(package_name)
        if hasattr(module, "__version__"):
            current_version = module.__version__
            print(f"   {package_name}: {current_version}")

            if min_version:
                from packaging import version

                if version.parse(current_version) < version.parse(min_version):
                    print(f"   ⚠️  Version {current_version} < required {min_version}")
                    return False
            return True
        else:
            print(f"   {package_name}: installed (version unknown)")
            return True
    except ImportError:
        print(f"   {package_name}: not installed")
        return False
    except Exception as e:
        print(f"   {package_name}: error checking version - {e}")
        return False


def check_dependencies():
    """Check current dependency status."""
    print("\n📦 Checking dependencies...")

    dependencies = {
        "torch": "2.0.0",
        "transformers": "4.57.0",
        "accelerate": None,
        "PIL": None,  # Actually 'Pillow' but imports as 'PIL'
        "numpy": None,
        "qwen_vl_utils": None,
    }

    missing = []
    outdated = []

    for package, min_version in dependencies.items():
        if not check_package_version(package, min_version):
            if package == "PIL":
                missing.append("Pillow")  # Install name is different
            else:
                missing.append(package)
            if min_version:
                outdated.append(f"{package}>={min_version}")

    return missing, outdated


def check_gpu_support():
    """Check GPU availability and CUDA support."""
    print("\n🖥️  Checking GPU support...")

    try:
        import torch

        cuda_available = torch.cuda.is_available()

        if cuda_available:
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ CUDA GPU detected: {gpu_name}")
            print(f"   VRAM: {gpu_memory:.1f} GB")

            # Check VRAM recommendations
            if gpu_memory < 4:
                print("   ⚠️  Low VRAM: Consider using 2B models with float16")
            elif gpu_memory < 8:
                print("   💡 Recommended: 2B-3B models with float16")
            elif gpu_memory < 16:
                print("   💡 Recommended: 4B models with float16 or auto")
            else:
                print("   💡 Recommended: Any model with auto precision")
        else:
            print("⚠️  No CUDA GPU detected - using CPU")
            print("   💡 CPU inference is very slow, consider using GPU")

        return cuda_available

    except ImportError:
        print("❌ PyTorch not installed - cannot check GPU")
        return False


def install_dependencies():
    """Install required dependencies."""
    print("\n📥 Installing dependencies...")

    # Core requirements
    requirements = [
        "torch>=2.0.0",
        "transformers>=4.57.0",
        "accelerate",
        "Pillow",
        "numpy",
        "qwen_vl_utils",
        "packaging",  # For version checking
    ]

    # Install requirements
    cmd = f"{sys.executable} -m pip install " + " ".join(requirements)
    print(f"Running: {cmd}")

    if not run_command(cmd):
        print("❌ Failed to install dependencies")
        return False

    print("✅ Dependencies installed successfully")
    return True


def test_transformers_compatibility():
    """Test if transformers supports Sa2VA models."""
    print("\n🧪 Testing transformers compatibility...")

    try:
        from transformers import AutoConfig

        # Try to load Sa2VA model config
        model_name = "ByteDance/Sa2VA-Qwen3-VL-4B"
        print(f"   Testing config load for {model_name}...")

        config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True,
            # Don't actually download the model, just test config
        )

        print("✅ Transformers supports Sa2VA models")
        return True

    except Exception as e:
        print(f"❌ Transformers compatibility test failed: {e}")
        print("   This might indicate an incompatible transformers version")
        print("   Sa2VA requires transformers >= 4.57.0")
        return False


def setup_comfyui_integration():
    """Set up ComfyUI integration."""
    print("\n🔗 Setting up ComfyUI integration...")

    # Check if we're in the right directory structure
    current_dir = Path.cwd()

    if current_dir.name.lower() != "comfyui-sa2va":
        print("⚠️  Not in ComfyUI-Sa2VA directory")
        print(f"   Current directory: {current_dir}")
        return False

    # Check for ComfyUI directory structure
    comfyui_root = current_dir.parent.parent
    if not (comfyui_root / "main.py").exists():
        print("⚠️  ComfyUI root directory not found")
        print(f"   Expected ComfyUI at: {comfyui_root}")
        print("   Make sure this is installed in ComfyUI/custom_nodes/")
        return False

    print(f"✅ ComfyUI integration ready")
    print(f"   ComfyUI root: {comfyui_root}")
    return True


def run_basic_test():
    """Run a basic functionality test."""
    print("\n🧪 Running basic functionality test...")

    try:
        # Test node imports
        sys.path.append(str(Path.cwd()))

        from nodes.sa2va_node_tpl import Sa2VANodeTpl

        print("✅ Sa2VA node import successfully")

        # Test node instantiation
        sa2va_node = Sa2VANodeTpl()

        print("✅ Sa2VA node instantiated successfully")

        # Test basic functionality
        input_types = sa2va_node.INPUT_TYPES()
        if "required" in input_types and "model_name" in input_types["required"]:
            print("✅ Sa2VA node interface is correct")
        else:
            print("❌ Sa2VA node interface is incorrect")
            return False

        return True

    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def print_usage_info():
    """Print usage information and next steps."""
    print("\n" + "=" * 60)
    print("🎉 ComfyUI-Sa2VA Installation Complete!")
    print("=" * 60)
    print()
    print("📋 Next Steps:")
    print("1. Restart ComfyUI to load the new nodes")
    print("2. Look for 'Sa2VA' category in the node menu")
    print("3. Start with the basic workflow in example_workflows/")
    print()
    print("🎯 Available Node:")
    print(
        "   • Sa2VA Segmentation - Main processing node with integrated mask conversion"
    )
    print()
    print("📚 Documentation:")
    print("   • README.md - Quick start guide")
    print("   • SA2VA_README.md - Detailed documentation")
    print("   • TROUBLESHOOTING.md - Common issues and solutions")
    print()
    print("💡 Model Recommendations by VRAM:")
    print("   • 4-6GB:  Sa2VA-InternVL3-2B with float16")
    print("   • 6-8GB:  Sa2VA-Qwen2_5-VL-3B with float16")
    print("   • 8-12GB: Sa2VA-Qwen3-VL-4B with float16")
    print("   • 16GB+:  Any model with auto precision")
    print()
    print("🚨 If you encounter issues:")
    print("   • Check TROUBLESHOOTING.md")
    print("   • Run: python test_sa2va.py")
    print("   • Report issues on GitHub")


def main():
    """Main installation function."""
    print("🚀 ComfyUI-Sa2VA Installation Script")
    print("=" * 50)

    # Step 1: Check Python version
    if not check_python_version():
        sys.exit(1)

    # Step 2: Check current dependencies
    missing, outdated = check_dependencies()

    # Step 3: Install dependencies if needed
    if missing or outdated:
        print(f"\n📋 Need to install: {missing + outdated}")
        if not install_dependencies():
            sys.exit(1)
    else:
        print("✅ All dependencies are up to date")

    # Step 4: Check GPU support
    check_gpu_support()

    # Step 5: Test transformers compatibility
    if not test_transformers_compatibility():
        print("\n💡 Try upgrading transformers:")
        print("   pip install transformers>=4.57.0 --upgrade")
        sys.exit(1)

    # Step 6: Setup ComfyUI integration
    if not setup_comfyui_integration():
        print("\n💡 Make sure you're running this from:")
        print("   ComfyUI/custom_nodes/ComfyUI-Sa2VA/")
        sys.exit(1)

    # Step 7: Run basic functionality test
    if not run_basic_test():
        print("\n❌ Installation verification failed")
        print("   Check TROUBLESHOOTING.md for solutions")
        sys.exit(1)

    # Step 8: Print success information
    print_usage_info()


if __name__ == "__main__":
    main()
