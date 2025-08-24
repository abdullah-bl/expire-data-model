#!/usr/bin/env python3
"""
Complete Date Expiration Detection Workflow
Run all components in sequence
"""

import os
import subprocess
import sys

def run_command(cmd, description):
    """Run command with error handling"""
    print(f"\n🔄 {description}")
    print(f"Command: {cmd}")
    print("-" * 50)

    try:
        result = subprocess.run(cmd, shell=True, check=True)
        print("✅ Completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def check_requirements():
    """Check if all requirements are installed"""
    print("🔍 Checking requirements...")

    try:
        import ultralytics
        import cv2
        import json
        print("✅ Basic requirements installed")
        return True
    except ImportError as e:
        print(f"❌ Missing requirement: {e}")
        print("Install: pip install -r requirements.txt")
        return False

def main():
    """Main workflow"""
    print("🚀 Date Expiration Detection - Complete Workflow")
    print("=" * 60)

    # Check requirements
    if not check_requirements():
        return

    steps = [
        {
            "cmd": "python download_dataset.py",
            "desc": "Download and setup dataset from Google Drive",
            "required": False
        },
        {
            "cmd": "python convert.py",
            "desc": "Convert annotations to YOLO and OCR datasets",
            "required": True
        },
        {
            "cmd": "python train_yolo.py --quick",
            "desc": "Train YOLOv11 model (quick mode)",
            "required": False
        },
        {
            "cmd": "python train_ocr.py",
            "desc": "Setup OCR training",
            "required": False
        },
        {
            "cmd": "python inference.py --image datasets/images/img_00001.jpg --visualize",
            "desc": "Test inference on sample image",
            "required": False
        }
    ]

    completed = 0
    for step in steps:
        if step["required"] or input(f"Run: {step['desc']}? (y/n): ").lower() == 'y':
            if run_command(step["cmd"], step["desc"]):
                completed += 1
            elif step["required"]:
                print("❌ Required step failed. Exiting.")
                return

    print("\n" + "=" * 60)
    print(f"🎉 Workflow completed! {completed}/{len(steps)} steps successful")
    print("=" * 60)

    if completed == len(steps):
        print("✅ All steps completed successfully!")
        print("\n📁 Your trained models are ready for deployment.")
    else:
        print("⚠️  Some steps were skipped or failed.")
        print("You can run individual components manually:")
        print("  python train_yolo.py --help")
        print("  python train_ocr.py --help")
        print("  python inference.py --help")

if __name__ == "__main__":
    main()
