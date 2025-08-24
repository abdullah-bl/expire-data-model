#!/usr/bin/env python3
"""
Date Expiration Model - Data Conversion Script
Converts annotations.json to YOLO format and custom OCR dataset
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any
import argparse

class DateExpirationConverter:
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            self.base_dir = Path.cwd()
        else:
            self.base_dir = Path(base_dir)
        self.annotations_file = self.base_dir / "datasets" / "annotations.json"
        self.images_dir = self.base_dir / "datasets" / "images"

        # YOLO class mapping
        self.classes = {
            "prod": 0,  # Product
            "date": 1,  # Date (for YOLO detection)
            "due": 2,   # Due date
            "code": 3   # Code/Barcode
        }

        # Output directories
        self.yolo_dataset_dir = self.base_dir / "yolo_dataset"
        self.ocr_dataset_dir = self.base_dir / "ocr_dataset"

    def setup_directories(self):
        """Create necessary directory structure for YOLO and OCR datasets"""
        print("Setting up directory structure...")

        # YOLO directories
        yolo_dirs = [
            self.yolo_dataset_dir / "images" / "train",
            self.yolo_dataset_dir / "images" / "val",
            self.yolo_dataset_dir / "labels" / "train",
            self.yolo_dataset_dir / "labels" / "val"
        ]

        # OCR directories
        ocr_dirs = [
            self.ocr_dataset_dir / "images" / "train",
            self.ocr_dataset_dir / "images" / "val",
            self.ocr_dataset_dir / "annotations"
        ]

        # Create all directories
        for dir_path in yolo_dirs + ocr_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

        print(f"Created directories for YOLO dataset at: {self.yolo_dataset_dir}")
        print(f"Created directories for OCR dataset at: {self.ocr_dataset_dir}")

    def convert_bbox_to_yolo(self, bbox: List[int], img_width: int, img_height: int) -> List[float]:
        """
        Convert bounding box from [x1, y1, x2, y2] to YOLO format [x_center, y_center, width, height]
        All values normalized to [0, 1]
        """
        x1, y1, x2, y2 = bbox

        # Calculate center coordinates
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0

        # Calculate width and height
        width = x2 - x1
        height = y2 - y1

        # Normalize to [0, 1]
        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height

        return [x_center, y_center, width, height]

    def create_yolo_label_file(self, img_name: str, annotations: List[Dict], img_width: int, img_height: int, split: str = "train"):
        """Create YOLO format label file for an image"""
        label_file = self.yolo_dataset_dir / "labels" / split / f"{img_name.replace('.jpg', '.txt')}"

        with open(label_file, 'w') as f:
            for ann in annotations:
                class_name = ann['cls']
                if class_name in self.classes:
                    class_id = self.classes[class_name]
                    bbox = ann['bbox']

                    # Convert bbox to YOLO format
                    yolo_bbox = self.convert_bbox_to_yolo(bbox, img_width, img_height)

                    # Write in YOLO format: class_id x_center y_center width height
                    line = f"{class_id} {' '.join(map(str, yolo_bbox))}\n"
                    f.write(line)

    def extract_ocr_data(self, img_name: str, annotations: List[Dict], split: str = "train") -> List[Dict]:
        """Extract OCR-relevant data (text annotations)"""
        ocr_entries = []

        for ann in annotations:
            if 'transcription' in ann:
                ocr_entries.append({
                    'image': img_name,
                    'bbox': ann['bbox'],
                    'text': ann['transcription'],
                    'class': ann['cls']
                })

        return ocr_entries

    def copy_image_to_split(self, img_name: str, split: str):
        """Copy image to appropriate train/val split directory"""
        src_path = self.images_dir / img_name
        dst_path = self.yolo_dataset_dir / "images" / split / img_name
        shutil.copy2(src_path, dst_path)

        # Also copy to OCR dataset if it has text annotations
        ocr_dst_path = self.ocr_dataset_dir / "images" / split / img_name
        shutil.copy2(src_path, ocr_dst_path)

    def create_data_yaml(self):
        """Create data.yaml file for YOLO training"""
        data_yaml = {
            'path': str(self.yolo_dataset_dir),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(self.classes),
            'names': list(self.classes.keys())
        }

        yaml_file = self.yolo_dataset_dir / "data.yaml"
        with open(yaml_file, 'w') as f:
            f.write("# YOLO Dataset Configuration\n")
            f.write(f"path: {data_yaml['path']}\n")
            f.write(f"train: {data_yaml['train']}\n")
            f.write(f"val: {data_yaml['val']}\n")
            f.write(f"nc: {data_yaml['nc']}\n")
            f.write("names:\n")
            for name in data_yaml['names']:
                f.write(f"  - {name}\n")

        print(f"Created data.yaml at: {yaml_file}")

    def create_ocr_annotations_file(self, ocr_data: List[Dict], split: str = "train"):
        """Create OCR annotations file"""
        ocr_file = self.ocr_dataset_dir / "annotations" / f"{split}_ocr.json"

        with open(ocr_file, 'w', encoding='utf-8') as f:
            json.dump(ocr_data, f, indent=2, ensure_ascii=False)

        print(f"Created OCR annotations at: {ocr_file}")

    def convert_dataset(self, train_split: float = 0.8):
        """Main conversion function"""
        print("Loading annotations...")
        with open(self.annotations_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)

        print(f"Found {len(annotations)} images in dataset")

        # Setup directories
        self.setup_directories()

        # Split data into train/val
        img_names = list(annotations.keys())
        train_size = int(len(img_names) * train_split)

        train_images = img_names[:train_size]
        val_images = img_names[train_size:]

        print(f"Train set: {len(train_images)} images")
        print(f"Validation set: {len(val_images)} images")

        # Process train set
        print("\nProcessing training set...")
        train_ocr_data = []
        for i, img_name in enumerate(train_images):
            if i % 100 == 0:
                print(f"  Processed {i}/{len(train_images)} images")

            img_data = annotations[img_name]
            img_width = img_data['width']
            img_height = img_data['height']
            img_annotations = img_data['ann']

            # Create YOLO label file
            self.create_yolo_label_file(img_name, img_annotations, img_width, img_height, "train")

            # Copy image
            self.copy_image_to_split(img_name, "train")

            # Extract OCR data
            ocr_entries = self.extract_ocr_data(img_name, img_annotations, "train")
            train_ocr_data.extend(ocr_entries)

        # Process validation set
        print("\nProcessing validation set...")
        val_ocr_data = []
        for i, img_name in enumerate(val_images):
            if i % 100 == 0:
                print(f"  Processed {i}/{len(val_images)} images")

            img_data = annotations[img_name]
            img_width = img_data['width']
            img_height = img_data['height']
            img_annotations = img_data['ann']

            # Create YOLO label file
            self.create_yolo_label_file(img_name, img_annotations, img_width, img_height, "val")

            # Copy image
            self.copy_image_to_split(img_name, "val")

            # Extract OCR data
            ocr_entries = self.extract_ocr_data(img_name, img_annotations, "val")
            val_ocr_data.extend(ocr_entries)

        # Create configuration files
        print("\nCreating configuration files...")
        self.create_data_yaml()
        self.create_ocr_annotations_file(train_ocr_data, "train")
        self.create_ocr_annotations_file(val_ocr_data, "val")

        # Print summary
        print("\n" + "="*50)
        print("CONVERSION COMPLETE!")
        print("="*50)
        print(f"YOLO dataset created at: {self.yolo_dataset_dir}")
        print(f"OCR dataset created at: {self.ocr_dataset_dir}")
        print(f"Classes: {self.classes}")
        print(f"Total training images: {len(train_images)}")
        print(f"Total validation images: {len(val_images)}")
        print(f"Total OCR training samples: {len(train_ocr_data)}")
        print(f"Total OCR validation samples: {len(val_ocr_data)}")
        print("="*50)

def main():
    parser = argparse.ArgumentParser(description="Convert annotations.json to YOLO and OCR datasets")
    parser.add_argument("--train-split", type=float, default=0.8, help="Train/validation split ratio")
    parser.add_argument("--base-dir", type=str, default=None, help="Base directory path (default: current working directory)")

    args = parser.parse_args()

    converter = DateExpirationConverter(args.base_dir)
    converter.convert_dataset(args.train_split)

if __name__ == "__main__":
    main()
