.PHONY: run build build-mlops-image up down stack-up stack-down sim-check scale-1 scale-2 scale-4 clean mediamtx-up mediamtx-down mediamtx-status publishers-up publishers-down publishers-status monitoring-up monitoring-down monitoring-restart monitoring-status monitoring-logs compile-parser sync-storage-data format-data-new split-data-new pack-yolo-shards prepare-data pack-shards retrain dvc-train export-onnx build-engine rollback mlops-pipeline deploy-model mlflow-up mlflow-down mlflow-status

IMAGE  ?= uit_medseg/mlops_thuc:dev
MLOPS_TRAIN_IMAGE ?= uit_medseg/mlops_train:dev
PYTHON := python3
COMPOSE_ENV_FILE ?= apps/vision_service/.env
COMPOSE := docker compose $(if $(wildcard $(COMPOSE_ENV_FILE)),--env-file $(COMPOSE_ENV_FILE),)
MEDIAMTX_SCRIPT := bash scripts/rtsp_sim_mediamtx.sh
PREP_RAW_DIR ?= dataset/data_new
PREP_YOLO_DIR ?= dataset/extracted/yolo_helmet_dataset_new
PREP_SPLIT_NAME ?= train
PREP_SEED ?= 42
PREP_SPLIT_RATIOS ?=
PREP_SYNC_FROM_STORAGE ?= 1
PREP_EVENTS_FILE ?= storage/logs/events.jsonl
PREP_SNAPSHOTS_DIR ?= storage/snapshots
PREP_SYNC_MODE ?= replace
PREP_MIN_CONF ?= 0.0

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

run: compile-parser monitoring-up publishers-up
	@echo ">>> Ensuring dependencies are running: redis + mediamtx + telegram-worker..."
	@$(COMPOSE) up -d redis mediamtx telegram-worker
	@HOST_GPU_ID=$${HOST_GPU_ID:-0}; \
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

## Build MLOps training Docker image (containing ultralytics and mlflow)
build-mlops-image:
	@echo ">>> Building $(MLOPS_TRAIN_IMAGE)..."
	docker build -f Dockerfile.mlops -t $(MLOPS_TRAIN_IMAGE) .
	@echo ">>> Done: $(MLOPS_TRAIN_IMAGE)"

## Build all Docker images (runtime + MLOps training)
build: build-mlops-image
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
	@echo ">>> Stopping standalone vision container (if running via make run)..."
	@docker stop uit_medseg_vision 2>/dev/null && docker rm -f uit_medseg_vision 2>/dev/null || true
	@echo ">>> Stopping compose stack..."
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

# ── Data preparation pipeline ─────────────────────────────────────────────────

## Đồng bộ detect data từ storage (events + snapshots) sang PREP_RAW_DIR
sync-storage-data:
	$(PYTHON) scripts/sync_storage_to_data_new.py \
		--events-file $(PREP_EVENTS_FILE) \
		--snapshots-dir $(PREP_SNAPSHOTS_DIR) \
		--output-dir $(PREP_RAW_DIR) \
		--mode $(PREP_SYNC_MODE) \
		--min-confidence $(PREP_MIN_CONF) \
		--shards-dir dataset/shards

## Step 1: Chuẩn hóa data mới về YOLO format tạm (images/train + labels/train)
## Ví dụ:
##   make format-data-new PREP_RAW_DIR=dataset/data_new PREP_YOLO_DIR=dataset/extracted/yolo_helmet_dataset_new
format-data-new:
	$(if $(filter 1 true yes TRUE,$(PREP_SYNC_FROM_STORAGE)),$(MAKE) sync-storage-data,)
	$(PYTHON) scripts/format_data_new_yolo.py \
		--raw-dir $(PREP_RAW_DIR) \
		--output-dir $(PREP_YOLO_DIR) \
		--split-name $(PREP_SPLIT_NAME)

## Step 2: Chia train/val/test từ dataset YOLO tạm
## Ví dụ:
##   make split-data-new PREP_SPLIT_RATIOS="0.8 0.1 0.1"
split-data-new:
	$(PYTHON) scripts/split_yolo_dataset.py --dataset-dir $(PREP_YOLO_DIR) --seed $(PREP_SEED) $(if $(strip $(PREP_SPLIT_RATIOS)),--split-ratios $(PREP_SPLIT_RATIOS),)

## Step 3: Đóng gói WebDataset shards tại dataset/shards/{train,val,test}/*.tar
## Ví dụ:
##   make pack-yolo-shards PREP_YOLO_DIR=dataset/extracted/yolo_helmet_dataset_new PREP_SPLIT_RATIOS="0.8 0.1 0.1"
pack-yolo-shards:
	$(PYTHON) scripts/pack_yolo_to_shards.py --input-dir $(PREP_YOLO_DIR) --seed $(PREP_SEED) $(if $(strip $(PREP_SPLIT_RATIOS)),--split-ratios $(PREP_SPLIT_RATIOS),)

## Full data preparation pipeline:
## format_data_new_yolo -> split_yolo_dataset -> pack_yolo_to_shards
prepare-data: format-data-new split-data-new pack-yolo-shards
	@echo ">>> Data preparation done."
	@echo ">>> Shards output: dataset/shards/{train,val,test}/*.tar"

## Backward-compatible alias
pack-shards: pack-yolo-shards

# ── Retrain pipeline ──────────────────────────────────────────────────────────

## Run retrain pipeline via DVC:
## extract -> train -> evaluate -> export -> compile
retrain:
	@echo ">>> [1/2] Kiểm tra và khởi động MLflow server..."
	@if ! docker ps --format '{{.Names}}' | grep -q '^uit_medseg_mlflow$$'; then \
		echo "    MLflow chưa chạy → khởi động container..."; \
		$(COMPOSE) up -d mlflow-server; \
		echo "    Chờ MLflow server sẵn sàng tại http://localhost:5001 ..."; \
		for i in $$(seq 1 30); do \
			if curl -sf http://localhost:5001/health > /dev/null 2>&1; then \
				echo "    ✅ MLflow server đã sẵn sàng (sau $$i giây)."; \
				break; \
			fi; \
			if [ $$i -eq 30 ]; then \
				echo "    ❌ MLflow server không phản hồi sau 30s. Dừng lại."; \
				exit 1; \
			fi; \
			sleep 1; \
		done; \
	else \
		echo "    ✅ MLflow đã đang chạy, dùng server hiện tại."; \
	fi
	@echo ">>> [2/2] Running DVC retrain pipeline (extract -> train -> evaluate -> export -> compile)..."
	dvc repro extract train evaluate export compile
	@echo ">>> Done! Xem kết quả tại MLflow UI: http://localhost:5001"
	@echo ">>>        Dừng MLflow thủ công nếu cần: make mlflow-down"

## Backward-compatible alias
dvc-train: retrain

# ── Model export & engine build ───────────────────────────────────────────────

## Export YOLOv8 best.pt → ONNX (tải tự động từ MLflow Model Registry)
## Dùng --alias để chỉ định alias khác (mặc định: Production từ params.yaml)
## Ví dụ: make export-onnx ALIAS=Staging
export-onnx:
	$(PYTHON) scripts/export_onnx.py $(if $(ALIAS),--alias $(ALIAS),)

## Rollback: tải model version N từ MLflow, export ONNX, build engine, và khởi động lại
## Bỏ qua Quality Gate check — dùng khi cần khôi phục model cũ khẩn cấp
## Ví dụ: make rollback VERSION=7
rollback:
	@if [ -z "$(VERSION)" ]; then \
		echo "[ERROR] Vui lòng chỉ định VERSION. Ví dụ: make rollback VERSION=7"; \
		echo ""; \
		echo "Danh sách các version có sẵn:"; \
		sqlite3 mlruns/mlflow.db "SELECT mv.version, rma.alias FROM model_versions mv LEFT JOIN registered_model_aliases rma ON mv.name=rma.name AND mv.version=rma.version WHERE mv.name='YOLOv8_Helmet_Model' ORDER BY CAST(mv.version AS INTEGER) DESC LIMIT 10;" 2>/dev/null || echo "  (Không truy vấn được DB)"; \
		exit 1; \
	fi
	@echo "============================================================"
	@echo "ROLLBACK → MLflow version $(VERSION)"
	@echo "============================================================"
	@echo ">>> [1/3] Export ONNX từ MLflow version $(VERSION) (chạy trong Docker)..."
	docker run --rm --net=host \
		--user "$$(id -u):$$(id -g)" \
		-e HOME=/tmp \
		-e YOLO_CONFIG_DIR=/tmp \
		-v $(PWD):/workspace \
		-w /workspace \
		-e PYTHONPATH=/workspace \
		$(MLOPS_TRAIN_IMAGE) \
		python3 scripts/export_onnx.py --version $(VERSION)
	@echo ">>> [2/3] Build TensorRT engine..."
	$(MAKE) build-engine
	@echo ">>> [3/3] Cập nhật symlink và khởi động lại vision-service..."
	$(PYTHON) scripts/deploy_model.py --rollback
	@echo ">>> [4/4] Cập nhật alias 'Production' trong MLflow Registry → version $(VERSION)..."
	@sqlite3 mlruns/mlflow.db \
		"INSERT OR REPLACE INTO registered_model_aliases(name, alias, version) VALUES('YOLOv8_Helmet_Model', 'production', '$(VERSION)');" \
		&& echo "[OK]   MLflow alias 'Production' → version $(VERSION)" \
		|| echo "[WARN] Không cập nhật được MLflow alias (sqlite3 không có hoặc DB lỗi)"
	@echo "============================================================"
	@echo "Rollback version $(VERSION) hoàn tất!"
	@echo "  Engine  : apps/vision_service/models/yolov8/yolov8_helmet_active.engine"
	@echo "  MLflow  : YOLOv8_Helmet_Model alias 'Production' → version $(VERSION)"
	@echo "============================================================"


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
	@echo ">>> Starting monitoring stack (Prometheus :9091, Grafana :3005)..."
	$(COMPOSE) up -d prometheus grafana
	@echo ">>> Grafana UI  : http://localhost:3005  (admin/admin)"
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

# ── MLflow server management ─────────────────────────────────────────────────

## Start MLflow server (UI at http://localhost:5001)
mlflow-up:
	@echo ">>> Starting MLflow server..."
	$(COMPOSE) up -d mlflow-server
	@echo ">>> MLflow UI: http://localhost:5001"

## Stop MLflow server
mlflow-down:
	@echo ">>> Stopping MLflow server..."
	$(COMPOSE) stop mlflow-server

## Show MLflow server status
mlflow-status:
	@docker ps --filter name=uit_medseg_mlflow --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" || echo "(not running)"

# ── MLOps Automation Pipeline ────────────────────────────────────────────────

## Run end-to-end MLOps pipeline (pull -> prepare-data -> retrain -> deploy)
mlops-pipeline:
	@echo ">>> Pulling latest data and pipeline state via DVC..."
	dvc pull -f
	@echo ">>> Preparing raw data, formatting, splitting, and packing shards..."
	$(MAKE) prepare-data
	@echo ">>> Executing retrain pipeline stages..."
	$(MAKE) retrain
	@echo ">>> Running smart deployment logic..."
	$(MAKE) deploy-model

## Deploy the active model conditionally (using evaluate_status.json Quality Gate)
deploy-model:
	$(PYTHON) scripts/deploy_model.py
