#!/bin/bash
# Quick setup script for Venipuncture AR Training System

echo "================================"
echo "Venipuncture AR System - Setup"
echo "================================"
echo ""

# Create virtual environment
echo "[1/3] Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "[2/3] Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "[3/3] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✓ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Generate ChArUco board:"
echo "   python charuco_board_generator.py"
echo ""
echo "2. Print the generated charuco_board.png at 100% scale"
echo ""
echo "3. Run the detection pipeline:"
echo "   python main.py --camera webcam --output-video"
echo ""
echo "For more options, see: python main.py --help"
echo ""
