"""
Quick setup script for Smart Campus Security & Attendance 2.0
Automates installation and initial configuration
"""

import subprocess
import sys
from pathlib import Path
from loguru import logger

def run_command(cmd, description):
    """Run shell command with error handling"""
    logger.info(f"{description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✓ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed: {e.stderr}")
        return False

def main():
    """Main setup process"""
    logger.info("=" * 60)
    logger.info("Smart Campus Security & Attendance 2.0 - Setup")
    logger.info("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 9):
        logger.error("Python 3.9+ required")
        return False
    
    logger.info(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    logger.info("\n[1/5] Installing Python dependencies...")
    if not run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing packages"
    ):
        logger.warning("Some packages may have failed to install")
    
    # Download models
    logger.info("\n[2/5] Downloading models...")
    if not run_command(
        f"{sys.executable} models/download_models.py",
        "Downloading AI models"
    ):
        logger.warning("Model download incomplete - will download on first run")
    
    # Generate alert sound
    logger.info("\n[3/5] Generating alert sound...")
    try:
        from assets.generate_sound import generate_alert_sound
        generate_alert_sound()
        logger.info("✓ Alert sound generated")
    except Exception as e:
        logger.warning(f"Alert sound generation failed: {e}")
    
    # Initialize database
    logger.info("\n[4/5] Initializing database...")
    try:
        from db import Database
        db = Database()
        logger.info("✓ Database initialized")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False
    
    # Seed demo data
    logger.info("\n[5/5] Seeding demo data...")
    response = input("Generate demo data? (Y/n): ")
    if response.lower() != 'n':
        if run_command(
            f"{sys.executable} tests/seed_data.py",
            "Generating demo data"
        ):
            logger.info("✓ Demo data created")
    
    # Setup complete
    logger.info("\n" + "=" * 60)
    logger.info("✓ Setup Complete!")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("1. Configure config.yaml with your settings")
    logger.info("2. Run: streamlit run main.py")
    logger.info("3. Access dashboard at http://localhost:8501")
    logger.info("\nFor Docker deployment:")
    logger.info("  docker-compose up -d")
    logger.info("\n" + "=" * 60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
