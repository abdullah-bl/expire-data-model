#!/usr/bin/env python3
"""
Test script to verify project portability
Run this to ensure the project works when shared
"""

import os
import tempfile
import shutil
from pathlib import Path

def test_convert_script():
    """Test that convert.py works with dynamic paths"""

    print("🧪 Testing convert.py portability...")

    # Test import
    try:
        from convert import DateExpirationConverter
        print("✅ Import successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

    # Test with default (current directory)
    try:
        converter = DateExpirationConverter()
        print(f"✅ Default base_dir: {converter.base_dir}")

        # Check if paths exist
        print(f"   📁 Annotations file exists: {converter.annotations_file.exists()}")
        print(f"   📁 Images dir exists: {converter.images_dir.exists()}")

    except Exception as e:
        print(f"❌ Converter initialization failed: {e}")
        return False

    # Test with custom path
    try:
        custom_path = Path.cwd() / "test_path"
        converter2 = DateExpirationConverter(str(custom_path))
        print(f"✅ Custom base_dir: {converter2.base_dir}")

    except Exception as e:
        print(f"❌ Custom path failed: {e}")
        return False

    return True

def test_yolo_config():
    """Test that yolo_dataset/data.yaml uses relative paths"""

    print("\n🧪 Testing YOLO dataset configuration...")

    config_path = Path("yolo_dataset/data.yaml")
    if not config_path.exists():
        print("❌ data.yaml not found")
        return False

    try:
        with open(config_path, 'r') as f:
            content = f.read()

        # Check for relative path
        if "path: ../yolo_dataset" in content:
            print("✅ Uses relative path: ../yolo_dataset")
        elif "path: ./yolo_dataset" in content:
            print("✅ Uses relative path: ./yolo_dataset")
        elif "/Users/" in content:
            print("❌ Still contains hardcoded absolute path")
            return False
        else:
            print(f"⚠️  Path format: {content.split('path:')[1].split()[0]}")

        print("✅ YOLO config looks portable")

    except Exception as e:
        print(f"❌ Error reading config: {e}")
        return False

    return True

def test_download_scripts():
    """Test that download scripts don't have hardcoded paths"""

    print("\n🧪 Testing download scripts...")

    scripts = ["download_dataset.py", "download_dataset_simple.py"]

    for script in scripts:
        if not Path(script).exists():
            print(f"⚠️  {script} not found - skipping")
            continue

        try:
            with open(script, 'r') as f:
                content = f.read()

            if "/Users/abdullah" in content:
                print(f"❌ {script} contains hardcoded path")
                return False
            else:
                print(f"✅ {script} is portable")

        except Exception as e:
            print(f"❌ Error reading {script}: {e}")
            return False

    return True

def simulate_different_location():
    """Simulate running from a different directory"""

    print("\n🧪 Simulating different installation location...")

    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "custom-yolo-test"
        temp_path.mkdir()

        print(f"📁 Testing in: {temp_path}")

        # Copy essential files
        files_to_copy = ["convert.py", "train_yolo.py", "requirements.txt"]
        for file in files_to_copy:
            if Path(file).exists():
                shutil.copy(file, temp_path)

        # Copy dataset structure
        if Path("yolo_dataset").exists():
            shutil.copytree("yolo_dataset", temp_path / "yolo_dataset")

        # Test from temp directory
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_path)

            # Test import
            import sys
            sys.path.insert(0, str(temp_path))

            from convert import DateExpirationConverter

            converter = DateExpirationConverter()
            print(f"✅ Works from different location: {converter.base_dir}")

            return True

        except Exception as e:
            print(f"❌ Failed from different location: {e}")
            return False

        finally:
            os.chdir(original_cwd)

def main():
    """Main test function"""

    print("🚀 Testing Project Portability")
    print("=" * 40)

    tests = [
        ("Convert script", test_convert_script),
        ("YOLO config", test_yolo_config),
        ("Download scripts", test_download_scripts),
        ("Different location", simulate_different_location)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 30)

        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")

    print("\n" + "=" * 40)
    print("📊 PORTABILITY TEST RESULTS")
    print("=" * 40)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("🎉 Project is fully portable and shareable!")
        print("\n💡 You can now:")
        print("   - Share the project folder")
        print("   - Run on different machines")
        print("   - Deploy to different environments")
    else:
        print("⚠️  Some tests failed. Fix the issues before sharing.")

    print("=" * 40)

if __name__ == "__main__":
    main()