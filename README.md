# 👁️ Real-Time Vision OCR

> **Real-time multilingual OCR using Python, OpenCV, and OCR.space API.**

A lightweight computer vision application that captures text from a webcam and extracts it using the **OCR.space Cloud OCR API**. It also provides a **FastAPI REST API** for integrating OCR with mobile or web applications.

---

## ✨ Features

* 📷 **Real-time webcam OCR** using OpenCV
* ☁️ **OCR.space Engine 3** for text recognition
* 🌍 **Multilingual OCR** with automatic language detection
* 🇮🇳 Supports multiple Indian languages
* 📦 Simple `.env` configuration
* ⚡ FastAPI backend for external applications
* 🔍 Text detection with bounding-box visualization

---

## 🌍 Supported Languages

The application supports:

`English` · `Hindi` · `Bengali` · `Gujarati` · `Kannada` · `Malayalam` · `Marathi` · `Nepali` · `Urdu`

When multiple languages are configured, OCR.space uses automatic language detection.

---

## 🔑 Get an OCR.space API Key

You can get a free API key from:

https://ocr.space/ocrapi/freekey

Add the key to your `.env` file:


---

## 🚀 Quick Start

### 1. Clone the project

```bash
git clone <your-repository-url>
cd vision_ocr_realtime
```

### 2. Create environment

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure `.env`

```env
OCRSPACE_API_KEY=your_api_key_here
LANGUAGE_HINTS=en,hi,bn
```

### 5. Run webcam OCR

By default, the application auto-detects connected cameras and starts with your default camera:

```bash
python main_webcam.py
```

#### Multi-Camera & External Webcam Options

* **List all connected webcams (laptop & external USB):**
  ```bash
  python main_webcam.py --list-cameras
  ```

* **Launch with an external webcam:**
  ```bash
  python main_webcam.py --camera 1
  ```

* **Auto-prioritize external webcam when plugged in:**
  ```bash
  python main_webcam.py --prefer-external
  ```

* **Live Hotkeys (while video feed is active):**
  * Press **`w`**: Switch seamlessly between laptop webcam and external webcam without restarting
  * Press **`0` - `9`**: Jump directly to Camera index 0, 1, 2, etc.
  * Press **`s`**: Force instant OCR trigger
  * Press **`m`**: Mute / Unmute Text-to-Speech (TTS)
  * Press **`c`**: Clear detected text
  * Press **`q`**: Quit

### 6. Run FastAPI server

```bash
uvicorn app_server:app --reload
```

---

## 📁 Project Structure

```text
vision_ocr_realtime/
│
├── .env
├── config.py
├── vision_ocr.py
├── main_webcam.py
├── app_server.py
├── requirements.txt
└── README.md
```

| File               | Purpose                             |
| ------------------ | ----------------------------------- |
| `config.py`        | Loads application configuration     |
| `vision_ocr.py`    | OCR.space API integration           |
| `main_webcam.py`   | Real-time OpenCV webcam application |
| `tts.py`           | Text-to-Speech (Instant Windows & Cloud AI) |
| `app_server.py`    | FastAPI REST API                    |
| `requirements.txt` | Python dependencies                 |

---

## 🛠️ Tech Stack

**Python** · **OpenCV** · **OCR.space API** · **FastAPI** · **Uvicorn** · **Requests** · **python-dotenv**

---
#
