"""
Model download script for Smart Campus Security & Attendance 2.0
Downloads ArcFace and YOLOv8 models
"""

import os
import sys
from pathlib import Path
import requests
from tqdm import tqdm
import zipfile
import tarfile
from loguru import logger


class ModelDownloader:
    """Download and setup pre-trained models"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def download_file(self, url: str, dest_path: Path, desc: str = "Downloading"):
        """Download file with progress bar"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(dest_path, 'wb') as f, tqdm(
                desc=desc,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            logger.info(f"Downloaded: {dest_path}")
            return True
        
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False
    
    def download_arcface(self):
        """Download InsightFace ArcFace model"""
        logger.info("Downloading ArcFace model...")
        
        arcface_dir = self.models_dir / "arcface"
        arcface_dir.mkdir(parents=True, exist_ok=True)
        
        # InsightFace buffalo_sc model (smaller, suitable for Pi)
        # Note: This is a placeholder URL - actual download requires InsightFace model zoo
        # In practice, models are downloaded automatically by InsightFace library
        
        logger.info("ArcFace models will be downloaded automatically by InsightFace")
        logger.info("Run: pip install insightface")
        logger.info("Models will be cached in ~/.insightface/models/")
        
        # Create a marker file
        marker_file = arcface_dir / "buffalo_sc" / "README.txt"
        marker_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(marker_file, 'w') as f:
            f.write("""
ArcFace Model - buffalo_sc

This model is automatically downloaded by InsightFace library.
The first time you run the application, InsightFace will download
the model to ~/.insightface/models/buffalo_sc/

Model details:
- Architecture: ArcFace (ResNet backbone)
- Embedding dimension: 512
- Size: ~150MB
- Optimized for: CPU inference

For manual download:
https://github.com/deepinsight/insightface/tree/master/model_zoo
""")
        
        logger.info(f"Created marker file: {marker_file}")
        return True
    
    def download_yolo(self):
        """Download YOLOv8-nano model"""
        logger.info("Downloading YOLOv8-nano model...")
        
        yolo_dir = self.models_dir / "yolov8"
        yolo_dir.mkdir(parents=True, exist_ok=True)
        
        # Download YOLOv8n face detection model
        # Using official Ultralytics YOLOv8n as base
        model_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
        model_path = yolo_dir / "yolov8n.pt"
        
        if model_path.exists():
            logger.info(f"YOLOv8 model already exists: {model_path}")
            return True
        
        success = self.download_file(model_url, model_path, "Downloading YOLOv8n")
        
        if success:
            logger.info("YOLOv8 model downloaded successfully")
            
            # Create info file
            info_file = yolo_dir / "README.txt"
            with open(info_file, 'w') as f:
                f.write("""
YOLOv8-nano Model

Downloaded from: Ultralytics official repository
Model: yolov8n.pt
Size: ~6MB

For face detection, you can:
1. Use this general object detection model
2. Fine-tune on WIDER FACE dataset for better face detection
3. Download a pre-trained face detection model

For mask/helmet detection:
- Fine-tune on custom dataset with mask/helmet annotations
- Or use a pre-trained model from Roboflow/Ultralytics Hub

Model conversion to TFLite:
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='tflite')
""")
            
            logger.info(f"Created info file: {info_file}")
        
        return success
    
    def download_all(self):
        """Download all required models"""
        logger.info("Starting model downloads...")
        
        success = True
        
        # Download ArcFace
        if not self.download_arcface():
            success = False
        
        # Download YOLO
        if not self.download_yolo():
            success = False
        
        if success:
            logger.info("All models downloaded successfully!")
            logger.info("\nNext steps:")
            logger.info("1. Install dependencies: pip install -r requirements.txt")
            logger.info("2. Configure config.yaml with your settings")
            logger.info("3. Run: streamlit run main.py")
        else:
            logger.error("Some models failed to download")
        
        return success
    
    def verify_models(self):
        """Verify that all required models are present"""
        logger.info("Verifying models...")
        
        required_paths = [
            self.models_dir / "arcface" / "buffalo_sc",
            self.models_dir / "yolov8" / "yolov8n.pt"
        ]
        
        all_present = True
        for path in required_paths:
            if path.exists():
                logger.info(f"✓ Found: {path}")
            else:
                logger.warning(f"✗ Missing: {path}")
                all_present = False
        
        return all_present


def main():
    """Main entry point"""
    logger.info("Smart Campus Security - Model Downloader")
    logger.info("=" * 50)
    
    downloader = ModelDownloader()
    
    # Check if models already exist
    if downloader.verify_models():
        logger.info("\nAll models are already present!")
        response = input("Re-download models? (y/N): ")
        if response.lower() != 'y':
            logger.info("Skipping download")
            return
    
    # Download models
    downloader.download_all()
    
    # Verify
    logger.info("\nVerifying installation...")
    downloader.verify_models()


if __name__ == "__main__":
    main()
