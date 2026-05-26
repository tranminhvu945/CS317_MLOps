.PHONY: run build up down stack-up stack-down sim-check scale-1 scale-2 scale-4 clean mediamtx-up mediamtx-down mediamtx-status publishers-up publishers-down publishers-status monitoring-up monitoring-down monitoring-restart monitoring-status monitoring-logs compile-parser

IMAGE  ?= uit_medseg/mlops_thuc:dev
PYTHON := python3
COMPOSE_ENV_FILE ?= apps/vision_service/.env
COMPOSE := docker compose $(if $(wildcard $(COMPOSE_ENV_FILE)),--env-file $(COMPOSE_ENV_FILE),)
MEDIAMTX_SCRIPT := bash scripts/rtsp_sim_mediamtx.sh

# ── Run ───────────────────────────────────────────────────────────────────────

## Run vision-service (no rebuild — uses existing image)
## Auto-removes any stale container with the same name before starting.
## Compile custom Yolo parser C++ code into .so library
compile-parser:
	@echo ">>> Compiling custom Yolo parser C++ code..."
	@mkdir -p apps/vision_service/libs/deepstream/lib
	@docker run --rm \
		-v $(PWD):/workspace \
		$(IMAGE) \
		make -C /workspace/apps/vision_service/src/deepstream/custom_parser -j$$(nproc) DS_LIB=/workspace/apps/vision_service/libs/deepstream/lib

run: compile-parser
	@echo ">>> Ensuring dependencies are running: redis + mediamtx..."
	@$(COMPOSE) up -d redis mediamtx
	@HOST_GPU_ID=$${HOST_GPU_ID:-7}; \
	CONTAINER_GPU_ID=$${GPU_ID:-0}; \
	echo ">>> Starting mlops_thuc (host GPU $$HOST_GPU_ID -> container GPU $$CONTAINER_GPU_ID)..."; \
	docker rm -f uit_medseg_vision 2>/dev/null || true; \
	docker run -it --rm \
		--name uit_medseg_vision \
		--gpus device=$$HOST_GPU_ID \
		--net=host \
		--add-host mediamtx:127.0.0.1 \
		--add-host redis:127.0.0.1 \
		-e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics \
		-e APP_ENV=$${APP_ENV:-development} \
		-e LOG_LEVEL=$${LOG_LEVEL:-INFO} \
		-e GPU_ID=$$CONTAINER_GPU_ID \
		-e CONFIG_DIR=/workspace/apps/vision_service/configs \
		-e RTMP_LOCATION="$${RTMP_LOCATION:-rtmp://127.0.0.1:1935/vision1 live=1}" \
		-e TELEGRAM_ENABLED=$${TELEGRAM_ENABLED:-true} \
		-e TELEGRAM_REDIS_HOST=$${TELEGRAM_REDIS_HOST:-127.0.0.1} \
		-e TELEGRAM_REDIS_PORT=$${TELEGRAM_REDIS_PORT:-6379} \
		-e TELEGRAM_REDIS_TOPIC=$${TELEGRAM_REDIS_TOPIC:-helmet_violations} \
		-e TELEGRAM_MIN_CONSEC_NO_HELMET_FRAMES=$${TELEGRAM_MIN_CONSEC_NO_HELMET_FRAMES:-3} \
		-e TELEGRAM_SNAPSHOT_SOURCE=$${TELEGRAM_SNAPSHOT_SOURCE:-probe} \
		-e TELEGRAM_SNAPSHOT_DIR=$${TELEGRAM_SNAPSHOT_DIR:-/workspace/storage/snapshots} \
		-e TELEGRAM_SNAPSHOT_RTMP_URL=$${TELEGRAM_SNAPSHOT_RTMP_URL:-rtmp://127.0.0.1:1935/vision1} \
		-e TELEGRAM_SNAPSHOT_HLS_URL=$${TELEGRAM_SNAPSHOT_HLS_URL:-http://127.0.0.1:8888/vision1/index.m3u8} \
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

## Start alert stack only (redis + mediamtx + vision-service + telegram-worker)
stack-up:
	@echo ">>> Cleaning potential name conflicts..."
	@docker rm -f uit_medseg_vision uit_medseg_telegram_worker 2>/dev/null || true
	@echo ">>> Stopping standalone MediaMTX simulator (if running)..."
	@$(MEDIAMTX_SCRIPT) down >/dev/null 2>&1 || true
	@if [ -f "$(COMPOSE_ENV_FILE)" ]; then set -a; . "$(COMPOSE_ENV_FILE)"; set +a; fi; \
	if [ -n "$${TELEGRAM_BOT_TOKEN:-}" ] && [ -n "$${TELEGRAM_CHAT_ID:-}" ]; then \
		echo ">>> Starting stack: redis + mediamtx + vision-service + telegram-worker"; \
		$(COMPOSE) up -d redis mediamtx vision-service telegram-worker; \
	else \
		echo ">>> TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set."; \
		echo ">>> Starting core stack only: redis + mediamtx + vision-service"; \
		$(COMPOSE) up -d redis mediamtx vision-service; \
	fi

## Stop all services
down:
	$(COMPOSE) down

## Stop alert stack only
stack-down:
	$(COMPOSE) stop vision-service telegram-worker redis mediamtx

## Verify 4 simulated HLS streams are reachable before scaling tests
sim-check:
	@echo "=== Simulated HLS endpoints ==="
	@for cam in cam01 cam02 cam03 cam04; do \
		url="http://127.0.0.1:8888/$$cam/index.m3u8"; \
		tmpfile=$$(mktemp); \
		code=$$(curl -sSL -o "$$tmpfile" -w "%{http_code}" "$$url" || true); \
		if [ "$$code" = "200" ] && grep -q "^#EXTM3U" "$$tmpfile"; then \
			echo "[OK] $$cam -> $$url (HLS playlist valid)"; \
		else \
			echo "[FAIL] $$cam -> $$url (http=$$code, invalid playlist)"; \
		fi; \
		rm -f "$$tmpfile"; \
	done

## Enable only first 1 camera config (cam_001)
scale-1:
	bash scripts/set_camera_count.sh 1

## Enable first 2 camera configs (cam_001..cam_002)
scale-2:
	bash scripts/set_camera_count.sh 2

## Enable first 4 camera configs (cam_001..cam_004)
scale-4:
	bash scripts/set_camera_count.sh 4

## Rebuild and restart (no cache)
rebuild: down build up

# ── Model export & engine build ───────────────────────────────────────────────

## Pack raw data into WebDataset shards (e.g. make pack-shards NEW_DATA=dataset/data_new)
pack-shards:
	@if [ -z "$(NEW_DATA)" ]; then echo "Lỗi: Vui lòng cung cấp biến NEW_DATA (ví dụ: make pack-shards NEW_DATA=dataset/data_new)"; exit 1; fi
	@echo ">>> Packing raw data from $(NEW_DATA) into shards..."
	python3 scripts/pack_shards.py --input-dir $(NEW_DATA)
	@echo ">>> Done! Now run 'dvc add dataset/shards' and 'make dvc-train'."

## Train model via DVC pipeline (extract → train → register to MLflow Registry)
dvc-train:
	@echo ">>> Running DVC pipeline (train stages)..."
	dvc repro
	@echo ">>> Done! Check MLflow UI at http://localhost:5001 for registered model."

## Export YOLOv8 best.pt → ONNX (tải tự động từ MLflow Model Registry)
## Dùng --alias để chỉ định alias khác (mặc định: Production từ params.yaml)
## Ví dụ: make export-onnx ALIAS=Staging
export-onnx:
	python3 scripts/export_onnx.py $(if $(ALIAS),--alias $(ALIAS),)

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
	@echo ">>> Ensuring mediamtx is running before starting publishers..."
	@$(COMPOSE) up -d mediamtx
	@sleep 2
	PROTOCOL=hls bash scripts/rtsp_sim_publishers.sh up

## Stop all HLS publishers
publishers-down:
	bash scripts/rtsp_sim_publishers.sh down

## Check HLS publishers status
publishers-status:
	PROTOCOL=hls bash scripts/rtsp_sim_publishers.sh status

# ── Monitoring (Prometheus + Grafana) ────────────────────────────────────────

## Start Prometheus + Grafana monitoring stack
monitoring-up:
	@echo ">>> Starting monitoring stack (Prometheus :9090, Grafana :3000)..."
	$(COMPOSE) up -d prometheus grafana
	@echo ">>> Grafana UI  : http://localhost:3000  (admin/admin)"
	@echo ">>> Prometheus  : http://localhost:9090"
	@echo ">>> Metrics src : http://localhost:9100/metrics"

## Stop Prometheus + Grafana monitoring stack
monitoring-down:
	@echo ">>> Stopping monitoring stack..."
	$(COMPOSE) stop prometheus grafana
	$(COMPOSE) rm -f prometheus grafana

## Restart monitoring stack (reload config changes)
monitoring-restart:
	$(COMPOSE) restart prometheus grafana

## Show monitoring stack status
monitoring-status:
	@echo "=== Monitoring Stack Status ==="
	@$(COMPOSE) ps prometheus grafana 2>/dev/null || echo "(not running)"
	@echo ""
	@echo "=== Metrics endpoint ==="
	@curl -s --max-time 2 http://localhost:9100/metrics | head -20 || echo "(vision-service not running or metrics disabled)"

## Tail logs from monitoring containers
monitoring-logs:
	$(COMPOSE) logs -f --tail=50 prometheus grafana

# ── MLOps Automation Pipeline ────────────────────────────────────────────────

## Run the end-to-end MLOps pipeline (data pull -> train -> evaluate -> compile)
mlops-pipeline:
	@echo ">>> Pulling latest data and pipeline state via DVC..."
	dvc pull
	@echo ">>> Executing automated MLOps pipeline stages..."
	dvc repro

## Deploy the active model by updating symlink and restarting the vision container
deploy-model:
	@echo ">>> Updating symlink for active engine..."
	cd apps/vision_service/models/yolov8 && ln -sf yolov8_helmet.onnx_b1_gpu0_fp16.engine yolov8_helmet_active.engine
	@echo ">>> Restarting vision-service container to load new engine..."
	docker restart uit_medseg_vision 2>/dev/null || $(COMPOSE) restart vision-service

