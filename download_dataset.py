#!/usr/bin/env python3
"""
Download and setup dataset from Google Drive
"""

import os
import zipfile
import shutil
from pathlib import Path

def download_dataset():
    """Download dataset from Google Drive and setup"""

    print("🚀 Downloading Dataset from Google Drive...")
    print("=" * 50)

    # Check if gdown is available
    try:
        import gdown
    except ImportError:
        print("❌ gdown not installed. Installing...")
        os.system("pip install gdown")
        import gdown

    # Google Drive file ID
    file_id = "1Ewnsqjm2SqdZvAN0EhVS6PulfoE59giX"
    download_url = f"https://drive.google.com/uc?id={file_id}"

    # Download the file
    print(f"📥 Downloading from: {download_url}")

    try:
        output_path = gdown.download(download_url, "dataset.zip", quiet=False)
        print("✅ Download completed!")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("💡 Try downloading manually from:")
        print("   https://drive.google.com/file/d/1Ewnsqjm2SqdZvAN0EhVS6PulfoE59giX/view")
        return False

    # Extract the zip file
    print("📦 Extracting dataset...")

    try:
        with zipfile.ZipFile("dataset.zip", 'r') as zip_ref:
            zip_ref.extractall("temp_dataset")
        print("✅ Extraction completed!")
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False

    # Find the extracted folder and rename it
    temp_dir = Path("temp_dataset")

    if not temp_dir.exists():
        print("❌ Extracted folder not found")
        return False

    # Look for the main dataset folder (could be nested)
    dataset_folders = list(temp_dir.glob("*"))
    main_folder = None

    # Find folder containing images or annotations
    for folder in dataset_folders:
        if folder.is_dir():
            # Check if it contains images or annotations
            if any(folder.glob("*.jpg")) or any(folder.glob("*.json")) or "images" in str(folder).lower():
                main_folder = folder
                break

    # If no specific folder found, use the first directory
    if not main_folder and dataset_folders:
        main_folder = dataset_folders[0]

    if main_folder:
        print(f"📁 Found dataset folder: {main_folder}")

        # Remove existing datasets folder if it exists
        if Path("datasets").exists():
            print("🗑️  Removing existing datasets folder...")
            shutil.rmtree("datasets")

        # Rename to datasets
        shutil.move(str(main_folder), "datasets")
        print("✅ Renamed to 'datasets'")

        # Clean up temporary files
        print("🧹 Cleaning up temporary files...")
        shutil.rmtree("temp_dataset")
        os.remove("dataset.zip")
        print("✅ Cleanup completed!")

        # Verify the dataset
        datasets_path = Path("datasets")
        if datasets_path.exists():
            # Count files
            image_count = len(list(datasets_path.rglob("*.jpg")))
            json_count = len(list(datasets_path.rglob("*.json")))

            print("📊 Dataset Summary:")
            print(f"   📁 Location: {datasets_path.absolute()}")
            print(f"   🖼️  Images: {image_count}")
            print(f"   📄 JSON files: {json_count}")

            return True
        else:
            print("❌ Dataset folder not found after rename")
            return False
    else:
        print("❌ No dataset folder found in extracted files")
        return False

def main():
    """Main function"""
    print("📥 Dataset Download & Setup")
    print("=" * 30)

    # Check if datasets already exists
    if Path("datasets").exists():
        choice = input("⚠️  'datasets' folder already exists. Overwrite? (y/n): ").lower()
        if choice != 'y':
            print("❌ Operation cancelled")
            return

    success = download_dataset()

    if success:
        print("\n" + "=" * 50)
        print("🎉 Dataset setup completed successfully!")
        print("=" * 50)
        print("📁 Dataset ready at: datasets/")
        print("\n🚀 Next steps:")
        print("   python convert.py              # Convert to YOLO format")
        print("   python train_yolo.py --quick   # Train YOLOv11")
        print("=" * 50)
    else:
        print("\n❌ Dataset setup failed")
        print("💡 Try downloading manually and extracting to 'datasets' folder")

if __name__ == "__main__":
    main()
