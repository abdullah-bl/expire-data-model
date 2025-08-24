# Date Expiration Model - YOLO + Custom OCR

This project converts annotation data into datasets suitable for training a date expiration detection model using YOLO for object detection and custom OCR for text recognition.

## Dataset Overview

- **Total Images**: 11,858
- **Training Set**: 9,486 images (80%)
- **Validation Set**: 2,372 images (20%)
- **Classes**: `prod` (0), `date` (1), `due` (2), `code` (3)
- **OCR Samples**: 13,349 training + 3,325 validation text annotations

## Directory Structure

```
custom-yolo/
├── convert.py              # ✅ Convert annotations to datasets
├── train_yolo.py           # 🚀 Train YOLOv11 model
├── train_ocr.py            # 🔤 Setup OCR training
├── inference.py            # 🔍 Run detection & recognition
├── run_all.py              # ⚡ Complete workflow
├── download_dataset.py     # 📥 Download dataset (gdown)
├── download_dataset_simple.py # 📥 Download dataset (requests)
├── test_portability.py     # 🧪 Test project portability
├── requirements.txt        # 📦 Dependencies
├── README.md               # 📖 Documentation
├── datasets/               # 📂 Original data (11,858 images)
│   ├── annotations.json    # Original annotations
│   └── images/            # Original images
├── yolo_dataset/           # 🟢 YOLO Training Data
│   ├── data.yaml          # YOLO configuration
│   ├── images/
│   │   ├── train/         # 9,486 training images
│   │   └── val/           # 2,372 validation images
│   └── labels/
│       ├── train/         # YOLO format labels
│       └── val/           # Validation labels
└── ocr_dataset/            # 🔵 OCR Training Data
    ├── annotations/
    │   ├── train_ocr.json  # 13,349 training samples
    │   └── val_ocr.json    # 3,325 validation samples
    └── images/
        ├── train/          # Training images (same as YOLO)
        └── val/            # Validation images
```

## YOLO Dataset Format

### Label Files (.txt)
Each label file corresponds to an image and contains lines in YOLO format:
```
class_id x_center y_center width height
```

Where:
- `class_id`: 0=prod, 1=date, 2=due, 3=code
- Coordinates are normalized to [0, 1]

Example:
```
0 0.1614785992217899 0.09536784741144415 0.15953307392996108 0.027247956403269755
1 0.377431906614786 0.09400544959128065 0.2178988326848249 0.02997275204359673
```

### Data Configuration (data.yaml)
```yaml
path: ./yolo_dataset  # Path to your yolo_dataset directory
train: images/train
val: images/val
nc: 4
names:
  - prod
  - date
  - due
  - code
```

## OCR Dataset Format

### Annotations (JSON)
Each OCR annotation contains:
```json
{
  "image": "img_00001.jpg",
  "bbox": [138, 58, 250, 80],
  "text": "11.2024",
  "class": "date"
}
```

Where:
- `image`: Image filename
- `bbox`: [x1, y1, x2, y2] in absolute pixel coordinates
- `text`: The transcribed text
- `class`: The object class

## Usage

### 🚀 Complete Workflow (Recommended)
```bash
# Run the complete pipeline
python run_all.py

# Or run individual steps:
python convert.py              # Convert data
python train_yolo.py --quick   # Train YOLOv11
python train_ocr.py            # Setup OCR
python inference.py --image test.jpg --visualize  # Test detection
```

### 1. Convert Data
```bash
python convert.py
# Creates yolo_dataset/ and ocr_dataset/
```

### 2. Train YOLOv11
```bash
# Quick training (recommended)
python train_yolo.py --quick

# Custom training
python train_yolo.py --model n --epochs 50 --batch 8

# Advanced training
python train_yolo.py --model m --imgsz 640 --epochs 100
```

### 3. Train OCR
```bash
python train_ocr.py
# Choose OCR method: EasyOCR, Tesseract, or Custom
```

### 4. Run Inference
```bash
# Single image
python inference.py --image test.jpg --visualize

# Batch processing
python inference.py --dir test_images/

# With trained model
python inference.py --model runs/train/weights/best.pt --image test.jpg
```

## 📊 Model Information

### YOLOv11 Classes
- **prod** (0): Product labels
- **date** (1): Date text (with OCR)
- **due** (2): Due date text (with OCR)
- **code** (3): Barcodes/codes

### Dataset Statistics
- **Training images**: 9,486
- **Validation images**: 2,372
- **OCR training samples**: 13,349
- **OCR validation samples**: 3,325

## 🔧 Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- `ultralytics>=8.3.0` - YOLOv11
- `opencv-python` - Image processing
- `pytesseract` - OCR (optional)
- `easyocr` - Advanced OCR (optional)

## 🎯 Quick Start

### Option 1: Download Dataset Automatically
```bash
# Install dependencies
pip install -r requirements.txt

# Download and setup dataset
python download_dataset.py

# Run complete workflow
python run_all.py

# That's it! Your model is ready.
```

### Option 2: Manual Dataset Setup
```bash
# Download dataset manually from:
# https://drive.google.com/file/d/1Ewnsqjm2SqdZvAN0EhVS6PulfoE59giX/view

# Extract to 'datasets' folder
# Then run:
python convert.py
python train_yolo.py --quick
```

### Option 3: Simple Download (No gdown)
```bash
# If gdown fails, use simple version
python download_dataset_simple.py
```

## 📁 Dataset Download

Two download scripts are provided:

1. **`download_dataset.py`** - Uses `gdown` library (recommended)
2. **`download_dataset_simple.py`** - Uses `requests` only (fallback)

Both scripts will:
- ✅ Download dataset from Google Drive
- ✅ Extract ZIP file automatically
- ✅ Rename folder to `datasets`
- ✅ Clean up temporary files

## 🧪 Testing Portability

Before sharing, test that the project works on different systems:

```bash
python test_portability.py
```

This will verify:
- ✅ No hardcoded paths
- ✅ Works from different directories
- ✅ All scripts use dynamic paths
- ✅ Ready for sharing

## 📝 Notes

- **YOLOv11**: Latest version with improved accuracy and speed
- **Portable**: No hardcoded paths - works on any system
- **Optimized**: Reduced image sizes and batch sizes for edge deployment
- **OCR Ready**: Dataset prepared for text recognition training
- **Minimal**: Streamlined codebase with only essential files
- **Shareable**: Tested and verified to work when copied to different locations
