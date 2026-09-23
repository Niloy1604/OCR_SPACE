# 🍓 Raspberry Pi 5 Deployment Guide: Real-Time Vision OCR & TTS

This guide walks you through deploying and running this **Real-Time Multilingual Vision OCR & Multi-Engine TTS** system on a **Raspberry Pi 5**.

---

## 📋 Hardware Requirements & Recommendations

| Component | Recommendation & Notes |
| :--- | :--- |
| **Board** | **Raspberry Pi 5** (4GB or 8GB RAM recommended) |
| **OS** | **Raspberry Pi OS (64-bit)** (Debian 12 "Bookworm") |
| **Power Supply** | **Official 27W USB-C Power Supply** (avoids throttling when powering USB webcams) |
| **Camera** | **Option A (Plug-and-play):** Standard USB Webcam (plugged into blue USB 3.0 port).<br>**Option B (CSI Ribbon):** Raspberry Pi Camera Module 3 or HQ Camera (requires standard 22-pin Pi 5 camera ribbon). |
| **Audio Output** | ⚠️ **Note:** Raspberry Pi 5 **does NOT have a 3.5mm analog headphone jack**.<br>Audio output options:<br>1. **USB Audio / USB Speaker** (e.g. USB soundcard dongle or USB desktop speaker).<br>2. **HDMI Audio** (TV or monitor with built-in speakers).<br>3. **Bluetooth Speaker** (paired via `bluetoothctl`).<br>4. **I2S DAC / Speaker HAT** (e.g., MAX98357A, HiFiBerry). |

---

## ⚡ Quick Start (Automated Script)

On your Raspberry Pi 5 terminal, clone the repository and run the setup script:

```bash
cd ~
git clone <your-repository-url> OCRSPACE
cd OCRSPACE

# Make the setup script executable and run it
chmod +x setup_rpi.sh
./setup_rpi.sh
```

The script automatically:
1. Installs all required APT system packages (`libportaudio2`, `alsa-utils`, `v4l-utils`, OpenGL libraries).
2. Creates a Python 3 virtual environment (`.venv`) with `--system-site-packages`.
3. Installs all Python dependencies.
4. Pre-downloads the offline **Piper TTS neural voice model** (`en_US-lessac-medium.onnx`) for instant low-latency speech on the Pi 5 CPU.
5. Prepares `.env` from `.env.example`.

---

## 🛠️ Step-by-Step Manual Setup

If you prefer manual installation:

### 1. Update system & install OS libraries

```bash
sudo apt-get update && sudo apt-get install -y \
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
    fonts-freefont-ttf
```

### 2. Create and activate a Virtual Environment

Debian 12 (Bookworm) uses PEP 668 to protect system packages. Always use a virtual environment:

```bash
cd ~/OCRSPACE
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 4. Configure your `.env`

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
nano .env
```

Set your keys:
```env
OCRSPACE_API_KEY=your_ocr_space_api_key_here
GOOGLE_API_KEY=your_gemini_api_key_here
LANGUAGE_HINTS=en,hi,bn,gu,kn,ml,mr,ne,ur
DETECTION_INTERVAL_SECONDS=2.0
```

---

## 📷 Camera Setup & Verification

### Using a USB Webcam (Recommended)
1. Plug your webcam into one of the blue **USB 3.0** ports on the Pi 5.
2. List available video devices:
   ```bash
   v4l2-ctl --list-devices
   ```
   Or use the built-in scanner:
   ```bash
   python main_webcam.py --list-cameras
   ```

### Using a Raspberry Pi CSI Camera Module (Module 3 / HQ)
1. Connect the ribbon cable to either `CAM/DISP0` or `CAM/DISP1` (make sure contacts face the correct direction).
2. Test camera hardware:
   ```bash
   rpicam-hello -t 3000
   ```
3. Because Pi 5 uses modern `libcamera`, to expose the CSI camera as a standard V4L2 device for OpenCV, run using `libcamerify`:
   ```bash
   libcamerify python main_webcam.py
   ```

---

## 🔊 Audio Output Verification

Check detected audio devices:
```bash
aplay -l
```

Test speaker output with a test tone:
```bash
speaker-test -t wav -c 2
```

Test Piper & Gemini TTS directly:
```bash
python tts.py "Hello, text to speech is operational on Raspberry Pi 5."
```

---

## 🚀 Running the Application

### Mode 1: Desktop Mode (With HDMI / Monitor connected)
Displays the live bounding box video feed, OCR progress monitor, and speaks detected text:
```bash
source .venv/bin/activate
python main_webcam.py
```

### Mode 2: Headless Mode (Over SSH / No monitor connected)
Runs the continuous detection and speech synthesis loop in the terminal without opening GUI windows:
```bash
source .venv/bin/activate
python main_webcam.py --headless
```

### Mode 3: Lower Resolution for Maximum FPS
If using high-resolution webcams (4K or 1080p), running at 720p or 480p gives maximum responsiveness on Pi 5:
```bash
python main_webcam.py --width 640 --height 480
```

### Mode 4: Mobile & Web REST API Backend
Host the FastAPI server so mobile apps or other computers on your local Wi-Fi can send images to the Pi:
```bash
source .venv/bin/activate
uvicorn app_server:app --host 0.0.0.0 --port 8000
```
Open `http://<pi-ip-address>:8000/docs` in your browser to access the interactive Swagger documentation.

---

## 🔄 Auto-Start as a Background Service (systemd)

To make the Raspberry Pi 5 automatically launch the OCR & TTS system whenever it powers on:

1. Create a service file:
   ```bash
   sudo nano /etc/systemd/system/vision-ocr.service
   ```

2. Paste the following configuration (replace `pi` with your username if different):
   ```ini
   [Unit]
   Description=Real-Time Vision OCR & TTS Service
   After=network.target sound.target

   [Service]
   Type=simple
   User=pi
   WorkingDirectory=/home/pi/OCRSPACE
   ExecStart=/home/pi/OCRSPACE/.venv/bin/python /home/pi/OCRSPACE/main_webcam.py --headless
   Restart=always
   RestartSec=5
   Environment=PYTHONUNBUFFERED=1

   [Install]
   WantedBy=multi-user.target
   ```

3. Enable and start the service:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable vision-ocr.service
   sudo systemctl start vision-ocr.service
   ```

4. Check live status and logs:
   ```bash
   sudo systemctl status vision-ocr.service
   journalctl -u vision-ocr.service -f
   ```

---

## 💡 Performance & Optimization Tips on Pi 5

1. **Active Cooler**: The Raspberry Pi 5 runs significantly faster when equipped with the official Raspberry Pi Active Cooler, preventing thermal throttling during prolonged video processing.
2. **Offline Piper TTS**: Piper runs ONNX models directly on the Pi 5's Cortex-A76 cores with a real-time factor under 0.1x (sub-50ms synthesis), providing zero-cloud-cost offline speech.
3. **Motion Stability**: The system automatically avoids sending blurry frames during rapid movement, preserving your OCR.space API rate limits.
