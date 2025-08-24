#!/usr/bin/env python3
"""
YOLOv11 + OCR Inference for Date Expiration Detection
Run detection and text recognition on images
"""

import os
import cv2
import json
import argparse
from pathlib import Path
from ultralytics import YOLO

class DateExpirationDetector:
    def __init__(self, yolo_model_path=None, device="auto"):
        """Initialize the detector"""
        self.device = device
        self.classes = {0: "prod", 1: "date", 2: "due", 3: "code"}

        # Load YOLOv11 model
        if yolo_model_path and os.path.exists(yolo_model_path):
            self.model = YOLO(yolo_model_path)
        else:
            # Use default YOLOv11n
            self.model = YOLO("yolov11n.pt")
            print("⚠️  Using default YOLOv11n model. Train your own for better results.")

        # Initialize OCR
        self.ocr_available = self._init_ocr()

    def _init_ocr(self):
        """Initialize OCR engine"""
        try:
            import pytesseract
            return True
        except ImportError:
            try:
                import easyocr
                self.reader = easyocr.Reader(['en'])
                return True
            except ImportError:
                print("⚠️  No OCR engine found. Install: pip install pytesseract easyocr")
                return False

    def detect_dates(self, image_path, conf=0.25, save_results=True):
        """Detect dates and objects in image"""
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return None

        print(f"🔍 Analyzing: {image_path}")

        # Run YOLO detection
        results = self.model(
            source=image_path,
            conf=conf,
            save=save_results,
            project="inference_results",
            name="date_detection",
            exist_ok=True
        )

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get detection info
                bbox = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.classes.get(class_id, "unknown")

                detection = {
                    "bbox": bbox.tolist(),
                    "confidence": confidence,
                    "class": class_name,
                    "class_id": class_id
                }

                # Extract text if OCR is available and it's a text class
                if self.ocr_available and class_name in ["date", "due", "code"]:
                    text = self._extract_text(image_path, bbox)
                    detection["text"] = text

                detections.append(detection)

        return detections

    def _extract_text(self, image_path, bbox):
        """Extract text from bounding box"""
        try:
            image = cv2.imread(image_path)
            x1, y1, x2, y2 = map(int, bbox)
            roi = image[y1:y2, x1:x2]

            # Preprocess for OCR
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Try pytesseract first
            try:
                import pytesseract
                text = pytesseract.image_to_string(thresh, config='--psm 6')
                return text.strip()
            except:
                # Fallback to EasyOCR
                try:
                    results = self.reader.readtext(thresh)
                    if results:
                        return results[0][1]  # Return text with highest confidence
                except:
                    pass

        except Exception as e:
            print(f"OCR error: {e}")

        return ""

    def batch_detect(self, image_dir, output_json="inference_results.json"):
        """Process multiple images"""
        image_dir = Path(image_dir)
        if not image_dir.exists():
            print(f"❌ Directory not found: {image_dir}")
            return

        # Find images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [f for f in image_dir.iterdir() if f.suffix.lower() in image_extensions]

        if not image_files:
            print(f"❌ No images found in {image_dir}")
            return

        print(f"📁 Processing {len(image_files)} images...")

        results = {}
        for img_path in image_files:
            detections = self.detect_dates(str(img_path), save_results=False)
            if detections:
                results[img_path.name] = detections

        # Save results
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"✅ Results saved to: {output_json}")
        return results

    def visualize_detections(self, image_path, detections=None):
        """Show detections on image"""
        if detections is None:
            detections = self.detect_dates(image_path, save_results=False)

        if not detections:
            print("No detections found")
            return

        image = cv2.imread(image_path)
        colors = {
            "prod": (0, 255, 0),    # Green
            "date": (255, 0, 0),    # Blue
            "due": (0, 0, 255),     # Red
            "code": (255, 255, 0)   # Yellow
        }

        for det in detections:
            bbox = det["bbox"]
            class_name = det["class"]
            confidence = det["confidence"]

            x1, y1, x2, y2 = map(int, bbox)
            color = colors.get(class_name, (255, 255, 255))

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = ".2f"
            if "text" in det and det["text"]:
                label += f" | {det['text']}"

            cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Show image
        cv2.imshow("Date Detection Results", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Date Expiration Detection Inference")
    parser.add_argument("--image", type=str, help="Single image path")
    parser.add_argument("--dir", type=str, help="Directory of images")
    parser.add_argument("--model", type=str, help="Path to trained YOLO model")
    parser.add_argument("--output", type=str, default="inference_results.json", help="Output JSON file")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--visualize", action="store_true", help="Show results with OpenCV")

    args = parser.parse_args()

    # Initialize detector
    detector = DateExpirationDetector(yolo_model_path=args.model)

    if args.image:
        # Single image
        detections = detector.detect_dates(args.image, conf=args.conf)

        if detections:
            print(f"\n📊 Detections in {args.image}:")
            for i, det in enumerate(detections, 1):
                print(f"  {i}. {det['class']} (confidence: {det['confidence']:.2f})")
                if "text" in det and det["text"]:
                    print(f"     Text: '{det['text']}'")

        if args.visualize:
            detector.visualize_detections(args.image, detections)

    elif args.dir:
        # Batch processing
        detector.batch_detect(args.dir, args.output)

    else:
        print("❌ Please specify --image or --dir")
        print("Usage examples:")
        print("  python inference.py --image test.jpg --visualize")
        print("  python inference.py --dir test_images/ --output results.json")

if __name__ == "__main__":
    main()
