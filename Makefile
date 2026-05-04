.PHONY: run build up down clean mediamtx-up mediamtx-down mediamtx-status publishers-up publishers-down publishers-status

IMAGE  ?= uit_medseg/mlops_thuc:dev
PYTHON := python3
COMPOSE := docker compose
MEDIAMTX_SCRIPT := bash scripts/rtsp_sim_mediamtx.sh

# ── Run ───────────────────────────────────────────────────────────────────────

## Run vision-service (no rebuild — uses existing image)
## Auto-removes any stale container with the same name before starting.
run:
	@HOST_GPU_ID=$${HOST_GPU_ID:-1}; \
	CONTAINER_GPU_ID=$${GPU_ID:-0}; \
	echo ">>> Starting mlops_thuc (host GPU $$HOST_GPU_ID -> container GPU $$CONTAINER_GPU_ID)..."; \
	docker rm -f uit_medseg_vision 2>/dev/null || true; \
	docker run -it --rm \
		--name uit_medseg_vision \
		--gpus device=$$HOST_GPU_ID \
		--net=host \
		-e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics \
		-e APP_ENV=$${APP_ENV:-development} \
		-e LOG_LEVEL=$${LOG_LEVEL:-INFO} \
		-e GPU_ID=$$CONTAINER_GPU_ID \
		-e CONFIG_DIR=/workspace/apps/vision_service/configs \
		-e PYTHONPATH=/workspace \
		-v $(PWD):/workspace \
		$(IMAGE) \
		python3 /workspace/apps/vision_service/src/main.py

# ── Build ─────────────────────────────────────────────────────────────────────

## Build Docker image (with --no-cache to pick up code changes)
build:
	@echo ">>> Building $(IMAGE)..."
	docker build -f Dockerfile.ds64_glib -t $(IMAGE) .
	@echo ">>> Done: $(IMAGE)"

# ── Compose ───────────────────────────────────────────────────────────────────

## Start all services (docker compose up -d)
up:
	$(COMPOSE) up -d

## Stop all services
down:
	$(COMPOSE) down

## Rebuild and restart (no cache)
rebuild: down build up

# ── Model export & engine build ───────────────────────────────────────────────

## Export YOLOv8 best.pt → ONNX on the host machine (no Docker needed)
export-onnx:
	python3 scripts/export_yolov8_to_onnx.py

## Build TensorRT engine from ONNX (inside container)
build-engine:
	@GPU_ID=$${GPU_ID:-0}; \
	docker run --rm --gpus '"device='"$$GPU_ID"'"' \
		--net=host \
		-e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics \
		-e PYTHONPATH=/workspace \
		-v $(PWD):/workspace \
		-v $(PWD)/apps/vision_service/models:/workspace/apps/vision_service/models \
		$(IMAGE) \
		/workspace/scripts/build_engine.sh

# ── Cleanup ───────────────────────────────────────────────────────────────────

## Remove __pycache__, .pyc, build artifacts
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/ 2>/dev/null || true
	$(MAKE) -C apps/vision_service/src/deepstream/custom_parser clean 2>/dev/null || true
	@echo ">>> Clean done."

# ── MediaMTX (RTSP/RTMP ingest server) ──────────────────────────────────────

## Start MediaMTX server (RTSP :8554, RTMP :1935, API :8888)
mediamtx-up:
	$(MEDIAMTX_SCRIPT) up

## Stop MediaMTX server
mediamtx-down:
	$(MEDIAMTX_SCRIPT) down

## Check MediaMTX server status
mediamtx-status:
	$(MEDIAMTX_SCRIPT) status

# ── HLS Publishers (ffmpeg → RTMP → MediaMTX → HLS) ─────────────────────────

## Start 4 HLS publishers (cam01..cam04 loop MP4 → RTMP → MediaMTX → HLS)
publishers-up:
	PROTOCOL=hls bash scripts/rtsp_sim_publishers.sh up

## Stop all HLS publishers
publishers-down:
	bash scripts/rtsp_sim_publishers.sh down

## Check HLS publishers status
publishers-status:
	PROTOCOL=hls bash scripts/rtsp_sim_publishers.sh status
