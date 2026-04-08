<div align="center">

```
    █████╗ ███████╗████████╗██╗  ██╗███████╗██████╗
   ██╔══██╗██╔════╝╚══██╔══╝██║  ██║██╔════╝██╔══██╗
   ███████║█████╗     ██║   ███████║█████╗  ██████╔╝
   ██╔══██║██╔══╝     ██║   ██╔══██║██╔══╝  ██╔══██╗
   ██║  ██║███████╗   ██║   ██║  ██║███████╗██║  ██║
   ╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝
```

### AI-powered fall detection and safety monitoring for real-world video analysis.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev)
[![TensorFlow Lite](https://img.shields.io/badge/TFLite-inference-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org/lite)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

[📦 Source Code](https://github.com/tony19053000/Aether) · [🤗 Hugging Face Space](https://huggingface.co)

---

*Upload a video. Detect critical fall events. Get structured visual insights back.*

</div>

---

## The Problem

Most video-based safety systems look impressive in demos but break down in real workflows.

A camera records a hallway. A fall incident may be hidden somewhere inside that footage. But before any response can happen, someone has to manually scrub through the timeline, frame by frame, to find out whether something important actually happened.

**That is the bottleneck.**

The problem isn't only pose detection — it's turning raw footage into something *actionable* without forcing a human to inspect every second by hand.

> A clip comes in from a room monitor.  
> A care facility needs faster review of potential incidents.  
> A safety system should flag what matters, not dump hours of footage on a screen.

**Aether attacks that first bottleneck directly.** Upload a video, and the system extracts pose data frame-by-frame, runs temporal fall detection, and returns a processed output video alongside a readable detection summary — structured safety insight instead of raw footage.

---

## How It Works

```
┌───────────────────────┐
│  Upload Video  /       │
│  Select Sample Case    │
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│  Load Video Source    │
│  & Prepare Frames     │
└──────────┬────────────┘
           │
           ▼
┌────────────────────────────────┐
│  Pose Extraction               │
│  → body landmarks per frame    │
│  → normalized keypoints        │
└──────────┬─────────────────────┘
           │
           ▼
┌────────────────────────────────┐
│  Sequence-Based AI Analysis    │
│  → temporal motion patterns    │
│  → fall / no-fall inference    │
│  → confidence estimation       │
└──────────┬─────────────────────┘
           │
           ▼
┌────────────────────────────────┐
│  Return Results                │
│  → processed output video      │
│  → detection summary           │
│  → scenario review panel       │
└────────────────────────────────┘
```

The core loop is intentionally simple:

**Upload → Analyze → Review → Act**

Every design decision in Aether supports that workflow.

---

## Features

| Feature | Description |
|---|---|
| 🧠 **AI Fall Detection** | Detects fall events from video using pose-based temporal analysis — not just raw frame appearance |
| 🦴 **Pose-Based Understanding** | Extracts body landmarks per frame and converts motion into structured sequences for robust, interpretable inference |
| 🎬 **Processed Output Video** | Returns an annotated visual stream so you can inspect exactly what the model saw and where detection occurred |
| 📋 **Detection Summary Panel** | Produces a human-readable result summary — no need to rely solely on reviewing output video |
| 🎯 **Sample Scenario Testing** | Preloaded test cases let you demonstrate the full workflow without uploading a new video every time |
| 🖥️ **Unified Web Interface** | Input, output, controls, and detection summary all live in one integrated product experience |
| 🐳 **Deployment-Ready Architecture** | Built as a containerized, deployable application — not an isolated notebook |
| 🔌 **Extensible Safety Monitoring Base** | Designed to serve as a foundation for alerting, event logging, or broader monitoring workflows |

---

## Tech Stack

```
┌────────────────────────────────────────────────────┐
│                   Aether Stack                     │
├──────────────┬─────────────────────────────────────┤
│ Frontend     │ React · Vite · JavaScript           │
│ Backend      │ Python · FastAPI · Gradio           │
│ CV Pipeline  │ MediaPipe Pose · OpenCV             │
│ Model        │ TensorFlow Lite                     │
│ Styling      │ Tailwind CSS                        │
│ Deployment   │ Docker · Hugging Face Spaces        │
└──────────────┴─────────────────────────────────────┘
```

Aether is a unified full-stack application. The frontend delivers the monitoring dashboard and analysis flow; the backend handles model loading, pose extraction, inference, and result delivery. Structured for deployment as a single integrated product, not a loose collection of scripts.

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+ *(for frontend development)*

### Installation

```bash
# Clone the repository
git clone https://github.com/tony19053000/Aether.git
cd Aether

# Install Python dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

Then open your browser at:

```
http://localhost:7860
```

> If runtime assets are not available locally, the app can be configured to fetch required files from the Hugging Face asset repository.

---

## Who It's For

Aether fits anywhere video-based incident review is slow, manual, or repetitive:

- **Care and assisted-living monitoring** — faster review of possible fall incidents
- **Safety-oriented computer vision demos** — end-to-end deployable system, not a notebook demo
- **Research and portfolio work** — demonstrates temporal video understanding with a full product interface
- **Edge-to-cloud fall detection workflows** — deployable backend with a usable frontend layer
- **Anyone who needs a human-motion analysis interface** beyond a Jupyter notebook

---

## Why This Matters

A lot of AI video projects stop at the model stage. They show predictions in a notebook, save a few annotated frames, and call it done.

Real usefulness starts one layer above that.

A practical system must:
- accept real input,
- process it reliably,
- return interpretable output,
- and present results through a usable workflow.

Aether turns detection capability into a complete product loop:

**source video in → structured result out → reviewable by a human → deployable beyond the notebook.**

That makes it meaningful both as a real application concept and as a full-stack AI portfolio project demonstrating computer vision, temporal modeling, backend integration, deployment thinking, and product design.

---

## Product Direction

The long-term direction is not just to classify falls — it's to make safety-event review faster, clearer, and less manual.

**The goal: turn passive footage into actionable safety intelligence.**

---

## License & Attribution

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

Aether is a customized adaptation of an earlier MIT-licensed open-source project. Required license and attribution terms have been preserved in this repository.

---

<div align="center">

*Built with a focus on real-world utility — not just demo performance.*

</div>
