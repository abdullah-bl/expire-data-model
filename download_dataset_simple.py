#!/usr/bin/env python3
"""
Simple dataset download from Google Drive (no gdown dependency)
"""

import os
import zipfile
import shutil
import requests
from pathlib import Path

def download_file_from_google_drive(file_id, destination):
    """Download file from Google Drive"""

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def download_dataset_simple():
    """Download and setup dataset using requests"""

    print("🚀 Downloading Dataset from Google Drive...")
    print("=" * 50)

    # Google Drive file ID
    file_id = "1Ewnsqjm2SqdZvAN0EhVS6PulfoE59giX"

    # Download the file
    print(f"📥 Downloading file ID: {file_id}")

    try:
        download_file_from_google_drive(file_id, "dataset.zip")
        print("✅ Download completed!")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("\n💡 Try using the gdown version:")
        print("   python download_dataset.py")
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

    # Find and rename the dataset folder
    temp_dir = Path("temp_dataset")

    if not temp_dir.exists():
        print("❌ Extracted folder not found")
        return False

    # Look for the main dataset folder
    dataset_folders = list(temp_dir.glob("*"))
    main_folder = None

    for folder in dataset_folders:
        if folder.is_dir():
            if any(folder.glob("*.jpg")) or any(folder.glob("*.json")):
                main_folder = folder
                break

    if not main_folder and dataset_folders:
        main_folder = dataset_folders[0]

    if main_folder:
        print(f"📁 Found dataset folder: {main_folder}")

        # Remove existing datasets folder
        if Path("datasets").exists():
            print("🗑️  Removing existing datasets folder...")
            shutil.rmtree("datasets")

        # Rename to datasets
        shutil.move(str(main_folder), "datasets")
        print("✅ Renamed to 'datasets'")

        # Clean up
        print("🧹 Cleaning up temporary files...")
        shutil.rmtree("temp_dataset")
        os.remove("dataset.zip")
        print("✅ Cleanup completed!")

        return True
    else:
        print("❌ No dataset folder found")
        return False

def main():
    """Main function"""
    print("📥 Simple Dataset Download & Setup")
    print("=" * 40)

    if Path("datasets").exists():
        choice = input("⚠️  'datasets' folder exists. Overwrite? (y/n): ").lower()
        if choice != 'y':
            print("❌ Operation cancelled")
            return

    success = download_dataset_simple()

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

if __name__ == "__main__":
    main()
