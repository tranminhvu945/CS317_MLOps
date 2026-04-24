# UIT-MedSeg MLOps — Helmet Violation Detection

> **UIT-MedSeg MLOps** — Real-time helmet / PPE violation detection for motorcycles, powered by NVIDIA DeepStream 6.4 and YOLOv8.
>
> Vietnamese: **Hệ thống MLOps phát hiện vi phạm không đội mũ bảo hiểm** — được phát triển bởi UIT MMLab.

[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-blue?logo=github-actions)](https://github.com)
[![Python](https://img.shields.io/badge/Python-3.8+-green?logo=python)](https://www.python.org/)
[![DeepStream](https://img.shields.io/badge/DeepStream-6.4-orange?logo=nvidia)](https://developer.nvidia.com/deepstream-sdk)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [Project Structure](#project-structure)
5. [Configuration Guide](#configuration-guide)
6. [Output Format — events.jsonl](#output-format--eventsjsonl)
7. [RTSP Streaming](#rtsp-streaming)
8. [Docker Deployment](#docker-deployment)
9. [Development](#development)
10. [Authors & License](#authors--license)

---

## Overview

UIT-MedSeg MLOps is a production-ready, GPU-accelerated video analytics pipeline that:

- Ingests **live RTSP streams** or **video files** (MP4, HLS).
- Runs **real-time object detection** using YOLOv8 / NVIDIA PeopleNet via DeepStream 6.4.
- Tracks **multiple objects** across frames with NvDCF multi-object tracker.
- Detects **helmet/PPE violations** using rule-based logic on detected `person` objects.
- Captures **snapshot images** and **video clips** as evidence for each violation.
- Writes **structured event logs** to `events.jsonl` for downstream SIEM / dashboards.
- Exposes a **FastAPI health endpoint** and a **RTSP output stream** for integration.

### Key Features

| Feature | Detail |
|---|---|
| **Framework** | NVIDIA DeepStream 6.4 + Python 3 (pyds 1.1.10) |
| **Detection** | YOLOv8 Helmet / PeopleNet INT8 ONNX |
| **Tracking** | NvDCF multi-object tracker (GPU-accelerated) |
| **OSD** | GStreamer python-ds examples overlay (clock, bbox, labels) |
| **Evidence** | Per-violation JPEG snapshots + MP4 clips |
| **Output** | `events.jsonl` (per-event) + `events.jsonl` (per-window summary) |
| **Streaming** | RTSP output over TCP/UDP (rtsp-simple-server) |
| **API** | FastAPI health + metrics endpoint |
| **CI/CD** | Docker Compose with NVIDIA GPU runtime |

---

## Architecture

```
                              ┌─────────────────────────────────────────┐
                              │          UIT-MedSeg MLOps               │
                              │           (Docker Compose)              │
                              └──────────────────┬──────────────────────┘
                                                 │
          ┌──────────────────────────────────────┼──────────────────────────────┐
          │                                      │                              │
          ▼                                      ▼                              ▼
  ┌───────────────┐                    ┌─────────────────┐             ┌─────────────────┐
  │   Camera /    │                    │  rtsp-simple-   │             │   backend-api   │
  │   Video File  │──────────────────▶│     server      │             │   (FastAPI)     │
  │               │   RTSP / HLS       │                 │             │                 │
  └───────────────┘                    └────────┬────────┘             │  :8080 /health  │
                                                 │                        └────────┬────────┘
                                                 │ (RTSP pull)                     │
                                                 ▼                                 │
  ┌────────────────────────────────────────────────────────────────────────────────┐
  │                        vision-service  (DeepStream 6.4)                       │
  │  ┌──────────┐   ┌────────┐   ┌──────────┐   ┌───────────┐   ┌──────────┐  │
  │  │  Source  │──▶│StreamMux│──▶│ Infer    │──▶│  Tracker  │──▶│   OSD    │  │
  │  │ (RTSP/   │   │(batching)│  │(YOLOv8/  │   │(NvDCF GPU)│   │(bbox,    │  │
  │  │  file)   │   │        │   │ PeopleNet)│   │           │   │ text)    │  │
  │  └──────────┘   └────────┘   └────┬─────┘   └───────────┘   └─────┬────┘  │
  │                                    │                                  │       │
  │                         ┌──────────▼──────────┐                     │       │
  │                         │  ViolationProbe    │                     │       │
  │                         │  (rule engine)      │                     │       │
  │                         │  → snapshot_service │                     │       │
  │                         │  → clip_service     │                     │       │
  │                         │  → event_publisher   │                     │       │
  │                         └──────────┬──────────┘                     │       │
  └─────────────────────────────────────┼─────────────────────────────────┼───────┘
                                        │                                 │
                                        ▼                                 ▼
                               storage/logs/                        storage/snapshots/
                               events.jsonl                         evidence/
                               storage/rolling/                    storage/clips/
```

---

## Quick Start

### Prerequisites

| Component | Version |
|---|---|
| Ubuntu | 20.04 LTS |
| Docker | ≥ 20.10 |
| NVIDIA Driver | ≥ 525 |
| NVIDIA Container Toolkit | configured |
| Python | ≥ 3.8 |
| GPU | NVIDIA GPU (Pascal or newer) |

### 1 — Clone & prepare environment

```bash
git clone https://github.com/your-org/uit-medseg.git
cd uit-medseg/MLOps

# Copy environment variables
cp .env.example .env
# Edit .env with your GPU_ID, RTSP URLs, etc.
```

### 2 — Install Python dependencies

```bash
# Install the pre-built pyds wheel (GPU only, ships with DeepStream bindings)
pip install pyds-1.1.10-py3-none-linux_x86_64.whl

# Install remaining dependencies
pip install -r requirements.txt
```

> **Note:** `pyds` is not published on PyPI. A compatible wheel for Python 3.10 on Linux x86_64 is bundled at `pyds-1.1.10-py3-none-linux_x86_64.whl`.

### 3 — (Optional) Build TensorRT engine

If you want to pre-compile the ONNX model to a TensorRT engine for faster startup:

```bash
bash apps/vision_service/scripts/build_engine.sh
```

### 4 — Run locally (Docker)

```bash
make docker-build   # Build vision-service + backend-api images
make docker-up      # Start all services
make docker-down   # Stop all services
```

### 5 — Run locally (host Python, for development)

```bash
# Set DeepStream environment variables
export GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream/lib/gst-plugins
export LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream/lib:$LD_LIBRARY_PATH
export PYTHONPATH="${PWD}:${PYTHONPATH}"

python apps/vision_service/src/main.py
```

### 6 — Run tests

```bash
make test
# or directly:
pytest apps/vision_service/tests/ -v
```

---

## Project Structure

```
MLOps/
├── .env.example                     # Environment variable template
├── .gitignore                       # Git ignore rules
├── .vscode/                         # VS Code workspace config
│   ├── launch.json                  # (keep) Debug launcher
│   ├── tasks.json                   # (keep) Build tasks
│   └── setting.json                 # (ignore) Personal settings
│
├── apps/
│   └── vision_service/
│       ├── configs/                 # YAML config files
│       │   ├── app.yaml             # Root application config
│       │   ├── infer/               # DeepStream inference configs
│       │   │   ├── pgie_yolov8_helmet.txt
│       │   │   └── pgie_peoplenet.txt
│       │   └── camera/              # Per-camera config (RTSP URLs, etc.)
│       │       └── camera.yaml
│       ├── models/                  # Model artifacts
│       │   ├── yolov8/              # YOLOv8 helmet model
│       │   │   ├── yolov8_helmet.onnx
│       │   │   └── labels.txt
│       │   └── peoplenet/           # NVIDIA PeopleNet model
│       │       ├── resnet34_peoplenet.onnx
│       │       └── labels.txt
│       ├── scripts/
│       │   ├── build_engine.sh      # TensorRT engine builder
│       │   └── run_docker.sh        # Docker entrypoint
│       ├── src/
│       │   ├── main.py              # CLI entry point
│       │   ├── app.py               # VisionApp — pipeline lifecycle
│       │   ├── settings.py          # Pydantic settings from YAML
│       │   ├── logger.py            # Structured logging setup
│       │   ├── domain/              # Domain models & schemas
│       │   │   ├── event_schema.py  # ViolationEvent Pydantic model
│       │   │   ├── detection_schema.py
│       │   │   ├── camera_schema.py
│       │   │   └── rules.py         # Violation detection rules
│       │   ├── pipeline/            # GStreamer pipeline components
│       │   │   ├── builder.py       # PipelineBuilder — assembles pipeline
│       │   │   ├── source_file.py   # File source element
│       │   │   ├── source_hls.py    # HLS stream source
│       │   │   ├── infer.py         # Nvinfer (primary detector)
│       │   │   ├── tracker.py       # NvMultiObjectTracker
│       │   │   ├── osd.py           # On-screen display overlay
│       │   │   ├── sink.py          # Output sink (rtsp/display/fake)
│       │   │   ├── rtsp_output.py  # RTSP server output bin
│       │   │   ├── muxer.py        # StreamMux configuration
│       │   │   ├── bus_handler.py  # GstBus message handler
│       │   │   └── frame_monitor.py # Frame rate / buffer monitoring
│       │   ├── probes/              # DeepStream metadata probes
│       │   │   ├── infer_probe.py  # Post-inference probe
│       │   │   ├── tracker_probe.py
│       │   │   └── violation_probe.py  # Violation detection logic
│       │   └── services/           # Application services
│       │       ├── snapshot_service.py
│       │       ├── clip_service.py
│       │       ├── event_publisher.py   # Writes events.jsonl
│       │       └── health_service.py
│       └── tests/
│           └── unit/
│               └── test_settings.py
│
├── storage/                        # Runtime output (gitignored)
│   ├── logs/
│   │   └── events.jsonl            # Structured event log
│   ├── snapshots/                  # Per-violation JPEG snapshots
│   ├── clips/                      # Per-violation MP4 clips
│   ├── evidence/                   # Compiled violation evidence bundles
│   └── rolling/                    # Rolling window metrics
│
├── Dockerfile.ds64_glib            # DeepStream 6.4 + GLib 2.76 build image
├── Dockerfile.dev                  # Lightweight development image
├── docker-compose.yml              # Full stack (GPU + RTSP + API)
├── Makefile                        # Developer convenience targets
├── requirements.txt                # Python pip dependencies
├── pyproject.toml                   # Project metadata & tool config
└── README.md                       # This file
```

---

## Configuration Guide

### `configs/app.yaml` — Root config

| Section | Key | Default | Description |
|---|---|---|---|
| `app.name` | Service name | `helmet-violation-service` | Used in logs and health endpoint |
| `app.env` | Environment | `development` | |
| `app.log_level` | Logging verbosity | `INFO` | DEBUG / INFO / WARNING / ERROR |
| `app.gpu_id` | GPU device index | `0` | |
| `storage.*` | Output directories | `storage/{logs,snapshots,clips}` | Must be writable |
| `pipeline.sink` | Output sink | `rtsp` | `fake` / `display` / `rtsp` |
| `infer.config_file` | DeepStream inference config | `pgie_*.txt` | Path relative to project root |
| `rtsp.enabled` | Enable RTSP output | `true` | |
| `rtsp.rtsp_port` | RTSP server port | `8554` | Must match docker-compose port mapping |
| `rules.trigger_label` | Class to monitor | `person` | |
| `rules.cooldown_sec` | Suppress duplicate events | `10` | Seconds between events for same track_id |

### Per-camera config — `configs/camera/camera.yaml`

```yaml
cameras:
  - camera_id: cam_001
    uri: "file:///workspace/data/test.mp4"
    enabled: true
    # Or use RTSP:
    # uri: "rtsp://user:pass@192.168.1.100:554/stream"
```

### Environment variables (`.env`)

```bash
APP_ENV=development
APP_NAME=uit-medseg-vision
LOG_LEVEL=INFO
GPU_ID=0
CONFIG_DIR=apps/vision_service/configs
```

All settings in `app.yaml` can be overridden with uppercase env vars
(e.g. `LOG_LEVEL=DEBUG` overrides `app.log_level`).

---

## Output Format — `events.jsonl`

Each line is a valid JSON object (JSONL / newline-delimited JSON).

### Violation Event

```json
{
  "event_type": "helmet_violation",
  "event_id": "evt_cam001_00001",
  "camera_id": "cam_001",
  "timestamp": "2026-04-13T10:23:45.123456+07:00",
  "track_id": 42,
  "confidence": 0.873,
  "bbox": [120.5, 340.2, 210.8, 480.0],
  "snapshot_path": "storage/snapshots/cam_001_evt_00001.jpg",
  "clip_path": "storage/clips/cam_001_evt_00001.mp4",
  "model_version": "yolov8_helmet_v1"
}
```

### Detection Window Summary (periodic)

```json
{
  "event_type": "detection_window_summary",
  "timestamp": "2026-04-13T10:25:00.000000+07:00",
  "payload": {
    "window_sec": 5.0,
    "frames": 150,
    "objects": 12,
    "counts_by_label": {"person": 12, "helmet": 8},
    "buffer_rate": 30.0
  }
}
```

---

## RTSP Streaming

### Push stream to the relay server (from camera / NVR)

```bash
ffmpeg -re -stream_loop -1 -i input.mp4 \
  -c copy -f rtsp rtsp://localhost:8554/cam_001
```

### Pull stream from DeepStream output

```
rtsp://localhost:8554/vision
```

DeepStream encodes annotated frames (with bbox + labels + clock overlay) and
publishes them to the `/vision` mount point on the `rtsp-simple-server`.

### Stream via HLS (HTTP)

The `rtsp-simple-server` also exposes the stream over HTTP at:
`http://localhost:8000/cam_001/hls/live.m3u8`

---

## Docker Deployment

### Full stack (GPU + RTSP + API)

```bash
# Build all images
make docker-build

# Start
make docker-up

# Check logs
docker compose logs -f vision-service
docker compose logs -f backend-api

# Stop
make docker-down
```

### Environment variables for production

```bash
APP_ENV=production
LOG_LEVEL=WARNING
GPU_ID=0
RTSP_RTSPPORTS=8554
# Optionally enable RTSP authentication:
# RTSP_RTSPTOKENS=1
```

### Health check

```bash
# Vision service
curl http://localhost:8090/health

# Backend API
curl http://localhost:8080/health

# Check events
tail -f storage/logs/events.jsonl
```

---

## Development

### Code formatting

```bash
make format      # Apply black + isort
make lint        # Check without modifying
```

### Running tests

```bash
make test
pytest apps/vision_service/tests/ -v --cov=apps.vision_service.src
```

### Adding a new rule

1. Edit `apps/vision_service/src/domain/rules.py`.
2. The `ViolationProbe` calls `evaluate_rules(frame_meta, obj_meta)`.
3. Emit a `ViolationEvent` via `EventPublisher` if the rule fires.

### Adding a new camera

1. Create `apps/vision_service/configs/camera/camera_<id>.yaml`.
2. Restart the service — cameras are auto-loaded from `streams.scan_camera_dir`.

---

## Authors & License

| | |
|---|---|
| **Project** | UIT-MedSeg MLOps |
| **Organization** | [UIT MMLab](https://mmlab.uit.edu.vn) — University of Information Technology, VNU-HCM |
| **License** | MIT |
| **Contact** | mmlab@uit.edu.vn |

Contributions are welcome! Please open an issue or submit a pull request.

---

*Built with NVIDIA DeepStream 6.4 · GStreamer · Python 3 · Pydantic · FastAPI*
