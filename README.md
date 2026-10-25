<div align="center">

<h1>Aether</h1>

**AI-powered fall detection from video**

Upload a video. Analyze movement. Review potential falls.

[Run locally](#run-locally-windows) · [How to use](#how-to-use) · [License](LICENSE)

</div>

---

## About the project

Aether analyzes recorded videos to identify potential falls using human pose estimation and a trained motion-classification model. It displays an annotated video, a detection summary, and a timestamped analysis timeline in a simple web dashboard.

Built for demonstrations and experimentation, Aether includes four sample videos, looping result playback, downloadable results, and an information panel for presenting the project.

## How it works

**Video → Body landmarks → Motion analysis → Fall prediction → Annotated results**

MediaPipe extracts body landmarks. A TensorFlow Lite model analyzes sequences of 30 frames and flags fall predictions at a 90% threshold. OpenCV draws the annotations, and FFmpeg prepares the video for browser playback.

| Technology | Role |
| --- | --- |
| Python, FastAPI, Uvicorn | Application backend and local server |
| MediaPipe | Human pose estimation |
| TensorFlow Lite, NumPy | Motion classification and feature processing |
| OpenCV, FFmpeg | Video processing and output encoding |
| HTML, JavaScript, Tailwind CSS | Main dashboard |
| Gradio | Alternate interface at `/api` |

The model and sample videos are included. **No GPU, Node.js, or npm build is required.**

## Run locally (Windows)

Use **PowerShell** for the commands below. Complete these steps in order.

### 1. Install the prerequisites

Install **64-bit Python 3.12**, **Git**, and **FFmpeg**. With Windows Package Manager (`winget`), run:

```powershell
winget install --id Python.Python.3.12 --exact
winget install --id Git.Git --exact
winget install --id Gyan.FFmpeg --exact
```

Close PowerShell and open a new window so the installed programs are available. Verify them:

```powershell
py -3.12 --version
git --version
ffmpeg -version
```

Each command should print version information before you continue.

### 2. Download the project

```powershell
cd "$HOME\Documents"
git clone https://github.com/tony19053000/Aether.git
cd Aether
```

If you already cloned the project, open its existing `Aether` folder instead.

### 3. Create an isolated Python environment

```powershell
py -3.12 -m venv .venv
```

The following commands use that environment directly; activation is not required.

### 4. Install dependencies

Run both commands from the `Aether` folder. Keep the second command on one line.

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt "mediapipe==0.10.21" "tensorflow-cpu==2.19.1" "numpy==1.26.4" "opencv-python-headless==4.11.0.86" "opencv-contrib-python==4.11.0.86" "gradio==6.30.0"
```

These version overrides match the key packages in the working Ubuntu setup. Windows installation has not yet been verified.

### 5. Start Aether

```powershell
.\.venv\Scripts\python.exe app.py
```

Wait for the server startup message, then open **[http://127.0.0.1:7860](http://127.0.0.1:7860)** in your browser.

Keep PowerShell open while using Aether. Press **Ctrl+C** to stop it. Internet access is needed for installation, the first pose-model download, and externally hosted dashboard styles and fonts.

### Start it again later

You only need these commands; do not repeat installation:

```powershell
cd "$HOME\Documents\Aether"
.\.venv\Scripts\python.exe app.py
```

<details>
<summary><strong>Ubuntu 24.04 setup</strong></summary>

Run these commands in a terminal, in order:

```bash
sudo apt update
sudo apt install -y git python3.12-venv ffmpeg libgl1 libglib2.0-0
git clone https://github.com/tony19053000/Aether.git
cd Aether
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt "mediapipe==0.10.21" "tensorflow-cpu==2.19.1" "numpy==1.26.4" "opencv-python-headless==4.11.0.86" "opencv-contrib-python==4.11.0.86" "gradio==6.30.0"
.venv/bin/python app.py
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860). For later runs, open the `Aether` directory and run `.venv/bin/python app.py`.

</details>

## How to use

1. Upload a short MP4 or select a sample video.
2. Click **Analyze Video** to begin processing.
3. Review the annotated video, detection summary, and timeline. Playback loops automatically; use **Stop Video** to stop it.
4. Click **Download Results** to save the text summary, or **Reset** to begin again.

Use the **ⓘ** icon in the header to open the project explanation during a presentation.

## Quick troubleshooting

- **Page cannot be reached:** keep the server terminal open, check for errors, and use the address above.
- **Model file not found:** run the start command from the `Aether` folder.
- **Video conversion fails:** verify that `ffmpeg -version` works in a new terminal.
- **Dependency errors:** use Python 3.12 and the full installation command above.

## Project scope

Aether is a prototype for uploaded videos, not a validated emergency-alert system. It can miss falls or produce false positives; confidence scores are not accuracy measurements. The timeline and export contain only the latest 15 status entries, so review the annotated video for the full clip. Start with short videos because memory usage increases with video length.

## License

Source code is available under the [MIT License](LICENSE), with the original copyright notice preserved. Bundled third-party sample footage may have separate usage rights.
