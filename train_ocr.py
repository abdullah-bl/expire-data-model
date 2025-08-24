#!/usr/bin/env python3
"""
OCR Training Script for Date Text Recognition
Simple and optimized for edge deployment
"""

import os
import json
import argparse
from pathlib import Path

def load_ocr_data(json_path):
    """Load OCR training data"""
    if not os.path.exists(json_path):
        print(f"❌ OCR data not found: {json_path}")
        return []

    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_ocr_data():
    """Analyze OCR dataset and show statistics"""
    print("🔍 Analyzing OCR Dataset...")

    train_data = load_ocr_data("ocr_dataset/annotations/train_ocr.json")
    val_data = load_ocr_data("ocr_dataset/annotations/val_ocr.json")

    if not train_data:
        print("❌ No OCR data found. Run convert.py first.")
        return False

    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

    # Show sample texts
    print("\n📝 Sample texts:")
    for i, item in enumerate(train_data[:5]):
        print(f"  {i+1}. '{item['text']}' ({item['class']})")

    return True

def train_with_easyocr():
    """Train OCR using EasyOCR"""
    try:
        import easyocr
        print("🚀 Training with EasyOCR...")

        # Initialize EasyOCR
        reader = easyocr.Reader(['en'])

        print("✅ EasyOCR model ready")
        print("📝 For custom training, you would need to:")
        print("   1. Fine-tune on your specific date formats")
        print("   2. Use the training data in ocr_dataset/")
        print("   3. Consider using Tesseract for simpler deployment")

        return True

    except ImportError:
        print("❌ EasyOCR not installed")
        print("Install: pip install easyocr")
        return False

def train_with_tesseract():
    """Simple Tesseract OCR setup"""
    try:
        import pytesseract
        print("🚀 Setting up Tesseract OCR...")

        # Check if Tesseract is installed
        try:
            version = pytesseract.get_tesseract_version()
            print(f"✅ Tesseract version: {version}")
        except Exception as e:
            print("❌ Tesseract not found. Install from: https://github.com/UB-Mannheim/tesseract/wiki")
            return False

        print("✅ Tesseract OCR ready")
        print("💡 For date recognition, Tesseract works well with:")
        print("   - Clear text")
        print("   - Consistent formatting")
        print("   - Preprocessing (grayscale, threshold)")

        return True

    except ImportError:
        print("❌ pytesseract not installed")
        print("Install: pip install pytesseract")
        return False

def create_ocr_training_script():
    """Create a simple OCR training template"""
    script_content = '''#!/usr/bin/env python3
"""
Custom OCR Training Template
Modify this to train your own date recognition model
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path

def load_training_data():
    """Load OCR training data"""
    with open("ocr_dataset/annotations/train_ocr.json", "r") as f:
        return json.load(f)

def preprocess_image(image_path, bbox):
    """Extract and preprocess text region"""
    image = cv2.imread(image_path)
    if image is None:
        return None

    x1, y1, x2, y2 = bbox
    roi = image[y1:y2, x1:x2]

    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply threshold for better OCR
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh

def train_custom_ocr():
    """Implement your custom OCR training here"""
    print("🎯 Custom OCR Training Template")

    # Load data
    data = load_training_data()
    print(f"Loaded {len(data)} training samples")

    # TODO: Implement your OCR training logic
    # Options:
    # 1. Fine-tune existing OCR model
    # 2. Train character recognition model
    # 3. Use transfer learning
    # 4. Implement sequence-to-sequence model

    print("📝 Training steps to implement:")
    print("   1. Data preprocessing")
    print("   2. Character segmentation")
    print("   3. Feature extraction")
    print("   4. Model training")
    print("   5. Evaluation")

if __name__ == "__main__":
    train_custom_ocr()
'''

    with open("custom_ocr_training.py", "w") as f:
        f.write(script_content)

    print("✅ Created custom_ocr_training.py template")

def main():
    """Main OCR training function"""
    print("🔤 OCR Training for Date Recognition")
    print("=" * 40)

    # Analyze dataset
    if not analyze_ocr_data():
        return

    print("\n📚 Available OCR Methods:")
    print("1. EasyOCR (Deep Learning)")
    print("2. Tesseract (Traditional OCR)")
    print("3. Custom Training Template")

    choice = input("\nChoose method (1-3) or press Enter for Tesseract: ").strip()

    if choice == "1":
        train_with_easyocr()
    elif choice == "3":
        create_ocr_training_script()
        print("📝 Check custom_ocr_training.py for training template")
    else:
        train_with_tesseract()

if __name__ == "__main__":
    main()
