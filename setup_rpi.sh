#!/usr/bin/env bash
# ==============================================================================
# Real-Time Vision OCR - Setup Script for Raspberry Pi 5 (Raspberry Pi OS 64-bit)
# ==============================================================================

set -e

echo "=== [1/5] Updating APT repositories & installing system dependencies ==="
sudo apt-get update -y
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    libportaudio2 \
    portaudio19-dev \
    libasound2-dev \
    alsa-utils \
    v4l-utils \
    fonts-dejavu-core \
    fonts-freefont-ttf \
    curl \
    wget

echo "=== [2/5] Creating Python virtual environment with system site packages ==="
if [ ! -d ".venv" ]; then
    python3 -m venv .venv --system-site-packages
    echo "Virtual environment created at .venv"
else
    echo "Virtual environment already exists at .venv"
fi

# Activate venv
source .venv/bin/activate

echo "=== [3/5] Upgrading pip & installing Python dependencies ==="
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo "=== [4/5] Setting up configuration ==="
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "Created .env from .env.example. Please edit .env with your API keys:"
        echo "  nano .env"
    fi
else
    echo ".env file already exists."
fi

echo "=== [5/5] Pre-caching Piper TTS Voice (English offline fallback) ==="
mkdir -p piper_voices
if [ ! -f "piper_voices/en_US-lessac-medium.onnx" ]; then
    echo "Downloading en_US-lessac-medium Piper voice for offline TTS..."
    curl -L -o piper_voices/en_US-lessac-medium.onnx \
        https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx || true
    curl -L -o piper_voices/en_US-lessac-medium.onnx.json \
        https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json || true
    echo "Piper voice downloaded to piper_voices/"
fi

echo ""
echo "=============================================================================="
echo "🎉 Setup complete! You are ready to run Real-Time Vision OCR on Raspberry Pi 5."
echo "=============================================================================="
echo ""
echo "Quick Start Commands:"
echo "  1. Activate virtual environment:"
echo "     source .venv/bin/activate"
echo ""
echo "  2. Test your connected cameras:"
echo "     python main_webcam.py --list-cameras"
echo ""
echo "  3. Run live OCR with desktop preview window:"
echo "     python main_webcam.py"
echo ""
echo "  4. Run headless (via SSH / without a monitor):"
echo "     python main_webcam.py --headless"
echo ""
echo "  5. Run FastAPI Mobile / Web Backend Server:"
echo "     uvicorn app_server:app --host 0.0.0.0 --port 8000"
echo "=============================================================================="
