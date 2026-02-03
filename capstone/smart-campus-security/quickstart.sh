#!/bin/bash
# Quick start script for Smart Campus Security & Attendance 2.0
# Linux/Mac bash script

echo "========================================"
echo "Smart Campus Security - Quick Start"
echo "========================================"
echo ""

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python not found. Please install Python 3.9+"
    exit 1
fi

echo "[1/4] Installing dependencies..."
python3 -m pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "WARNING: Some packages may have failed to install"
fi

echo ""
echo "[2/4] Generating alert sound..."
python3 assets/generate_sound.py
if [ $? -ne 0 ]; then
    echo "WARNING: Alert sound generation failed"
fi

echo ""
echo "[3/4] Initializing database..."
python3 -c "from db import Database; Database()"
if [ $? -ne 0 ]; then
    echo "ERROR: Database initialization failed"
    exit 1
fi

echo ""
echo "[4/4] Seeding demo data..."
read -p "Generate demo data? (Y/n): " seed
if [ "$seed" != "n" ] && [ "$seed" != "N" ]; then
    python3 tests/seed_data.py
fi

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "To start the application, run:"
echo "  streamlit run main.py"
echo ""
echo "Then open: http://localhost:8501"
echo ""
