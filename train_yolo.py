#!/usr/bin/env python3
"""
YOLOv11 Training Script for Date Expiration Detection
Minimal and optimized for edge deployment
"""

import os
import argparse
from ultralytics import YOLO

def train_yolo(
    data_yaml="yolo_dataset/data.yaml",
    model_size="n",
    epochs=50,
    batch_size=8,
    img_size=416,
    device="auto"
):
    """Train YOLOv11 model with optimized settings"""

    print(f"🚀 Training YOLOv11{model_size} for Date Expiration Detection")
    print("=" * 50)
    print(f"Data: {data_yaml}")
    print(f"Epochs: {epochs}, Batch: {batch_size}, Image: {img_size}")
    print(f"Device: {device}")
    print("=" * 50)

    # Load YOLOv11 model
    model_name = f"yolo11{model_size}"
    print(f"Loading {model_name}...")

    try:
        model = YOLO(model_name)
        print(f"✅ Successfully loaded {model_name} model")
    except Exception as e:
        print(f"❌ Error loading {model_name}: {e}")
        print("Make sure you have the latest ultralytics installed: pip install ultralytics")
        return None

    # Check dataset
    if not os.path.exists(data_yaml):
        print(f"❌ Dataset not found: {data_yaml}")
        print("Run: python convert.py")
        return None

    # Train with optimized settings
    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            device=device,
            save=True,
            save_period=10,
            project="yolo11_date_detection",
            name=f"yolov11{model_size}",
            exist_ok=True,
            pretrained=True,
            optimizer="auto",
            verbose=True,
            val=True,
            plots=True,
            # Performance optimizations
            conf=0.25,
            iou=0.45,
            max_det=300,
            patience=20,
            cos_lr=True,
            amp=True
        )

        print(f"\n✅ Training completed! Model saved at: {results.save_dir}")
        return results

    except Exception as e:
        print(f"❌ Training error: {e}")
        return None

def quick_train(device="auto"):
    """Quick training with optimal settings"""
    print("⚡ Quick Training Mode")
    return train_yolo(
        model_size="n",
        epochs=30,
        batch_size=4,
        img_size=320,
        device=device
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv11 for Date Detection")
    parser.add_argument("--data", type=str, default="yolo_dataset/data.yaml", help="Dataset config")
    parser.add_argument("--model", type=str, default="n", choices=["n", "s", "m", "l", "x"], help="Model size")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=416, help="Image size")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--quick", action="store_true", help="Quick training mode")

    args = parser.parse_args()

    if args.quick:
        results = quick_train(device=args.device)
    else:
        results = train_yolo(
            data_yaml=args.data,
            model_size=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.imgsz,
            device=args.device
        )

    if results:
        print("\n📊 Results saved to:", results.save_dir)
