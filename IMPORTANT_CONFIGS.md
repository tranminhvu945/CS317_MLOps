# CS317_MLOps Important Configs

Tài liệu này chỉ giữ lại các config quan trọng nhất của project để tra cứu nhanh khi vận hành, debug hoặc tinh chỉnh pipeline.

## 1. Runtime chính

File: `apps/vision_service/configs/app.yaml`

| Key | Giá trị hiện tại | Ý nghĩa | Khi nào cần sửa |
|---|---|---|---|
| `app.env` | `development` | Môi trường chạy của app | Khi chuyển sang staging/production |
| `app.log_level` | `INFO` | Mức chi tiết log | Khi cần debug sâu hơn hoặc giảm log |
| `app.gpu_id` | `0` | GPU mà pipeline sử dụng | Khi đổi GPU chạy inference |
| `pipeline.sink` | `rtmp` | Kiểu output chính của pipeline | Khi muốn đổi sang `rtsp`, `fake`, `display` |
| `pipeline.max_sources` | `16` | Số camera tối đa hệ thống dự kiến hỗ trợ | Khi scale số lượng nguồn lớn hơn |
| `infer.config_file` | `/workspace/apps/vision_service/configs/infer/pgie_yolov8_helmet.txt` | File config model DeepStream | Khi đổi model hoặc profile infer |
| `rtmp.location` | `rtmp://mediamtx:1935/vision1` | Đích publish stream RTMP | Khi đổi server stream hoặc stream name |
| `metrics.port` | `9100` | Cổng exporter Prometheus | Khi bị trùng port hoặc đổi monitoring setup |
| `telegram.enabled` | `true` | Bật/tắt logic alert Telegram | Khi muốn chạy không gửi cảnh báo |
| `telegram.redis_host` | `redis` | Redis nhận alert events | Khi Redis không nằm trong compose mặc định |
| `telegram.redis_topic` | `helmet_violations` | Topic Redis cho alert | Khi đổi tên kênh giao tiếp giữa services |
| `telegram.min_consecutive_no_helmet_frames` | `3` | Số frame vi phạm liên tiếp để tạo alert | Khi muốn giảm/tăng độ nhạy cảnh báo |

## 2. Camera đầu vào

Files:

- `apps/vision_service/configs/camera/camera.yaml`
- `apps/vision_service/configs/camera/camera_002.yaml`
- `apps/vision_service/configs/camera/camera_003.yaml`
- `apps/vision_service/configs/camera/camera_004.yaml`

| Key | Ví dụ hiện tại | Ý nghĩa | Khi nào cần sửa |
|---|---|---|---|
| `enabled` | `true` | Camera có được nạp vào pipeline hay không | Khi bật/tắt một nguồn camera |
| `stream.type` | `hls` | Loại nguồn video | Khi nguồn là `rtsp` hoặc `file` |
| `stream.uri` | `http://mediamtx:8888/cam02/index.m3u8?cookieCheck=1` | URL/URI thực của luồng vào | Khi đổi camera thật hoặc stream giả lập |
| `stream.reconnect_interval_sec` | `10` | Chu kỳ reconnect khi mất nguồn | Khi stream không ổn định |
| `stream.timeout_sec` | `15` | Timeout khi đọc stream | Khi nguồn chậm hoặc hay timeout |
| `detection.min_confidence` | `0.5` | Ngưỡng confidence tối thiểu của detection | Khi muốn giảm false positive hoặc tăng độ nhạy |
| `detection.roi` | polygon hoặc rỗng | Vùng quan tâm áp dụng cho camera | Khi chỉ muốn detect trong một vùng cụ thể |

Ghi chú:

- `camera.yaml` hiện là camera thật trong LAN nội bộ.
- `camera_002` đến `camera_004` hiện là camera giả lập qua MediaMTX.

## 3. Config model infer

File: `apps/vision_service/configs/infer/pgie_yolov8_helmet.txt`

| Key | Giá trị hiện tại | Ý nghĩa | Khi nào cần sửa |
|---|---|---|---|
| `model-engine-file` | `/workspace/apps/vision_service/models/yolov8/yolov8_helmet_active.engine` | Engine TensorRT đang dùng để infer | Khi deploy model mới |
| `onnx-file` | `/workspace/apps/vision_service/models/yolov8/yolov8_helmet.onnx` | File ONNX nguồn | Khi export model mới |
| `labelfile-path` | `/workspace/apps/vision_service/models/yolov8/labels.txt` | File mapping tên class | Khi đổi class labels |
| `batch-size` | `1` | Số frame/nguồn infer trong một batch | Khi tối ưu throughput |
| `infer-dims` | `3;640;640` | Kích thước input model | Khi model mới có input size khác |
| `network-mode` | `2` | Precision mode, hiện là FP16 | Khi cần đổi precision |
| `num-detected-classes` | `2` | Số class detection | Khi đổi bài toán hoặc model |
| `pre-cluster-threshold` | `0.25` | Confidence threshold trước NMS | Khi muốn lọc detect yếu |
| `nms-iou-threshold` | `0.55` | Ngưỡng IoU của NMS | Khi cần tinh chỉnh overlap filtering |
| `custom-lib-path` | `/workspace/apps/vision_service/libs/deepstream/lib/libnvdsinfer_custom_impl_Yolo.so` | Parser custom cho output YOLO | Khi đổi parser hoặc rebuild DeepStream parser |

## 4. Hạ tầng container

File: `docker-compose.yml`

| Thành phần | Giá trị hiện tại | Ý nghĩa | Khi nào cần sửa |
|---|---|---|---|
| `mlflow-server` | `5001:5001` | UI và tracking backend cho MLflow | Khi đổi cổng MLflow |
| `redis` | `6379:6379` | Broker cho alert pipeline | Khi Redis nằm ngoài stack |
| `mediamtx` | `8554`, `1935`, `8888`, `9997` | Relay RTSP/RTMP/HLS/API | Khi đổi topology stream |
| `vision-service` | `9105:9100` | Expose metrics Prometheus ra host | Khi đổi cổng monitoring |
| `telegram-worker` | dùng `REDIS_*`, `TELEGRAM_*` | Worker gửi ảnh/text tới Telegram | Khi đổi kênh cảnh báo |
| `prometheus` | `9091:9090` | Metrics collector | Khi thay đổi hạ tầng giám sát |
| `grafana` | `3005:3000` | Dashboard monitoring | Khi đổi UI port |

## 5. Media relay

File: `configs/mediamtx.yml`

| Key | Giá trị hiện tại | Ý nghĩa | Khi nào cần sửa |
|---|---|---|---|
| `apiAddress` | `:9997` | Cổng API quản trị MediaMTX | Khi tích hợp automation khác |
| `rtspAddress` | `:8554` | Cổng RTSP output | Khi đổi endpoint player RTSP |
| `rtmpAddress` | `:1935` | Cổng RTMP publish | Khi `vision-service` publish sang server khác |
| `hlsAddress` | `:8888` | Cổng HLS playback | Khi đổi web playback endpoint |
| `paths."~^.*$".source` | `publisher` | Mọi stream path nhận nguồn do publisher push lên | Khi muốn cố định source theo từng path |

## 6. MLOps training/deploy

File: `params.yaml`

| Key | Giá trị hiện tại | Ý nghĩa | Khi nào cần sửa |
|---|---|---|---|
| `model.name` | `dataset/TrainYolo/runs/detect/helmet_training/run_12/weights/best.pt` | Checkpoint khởi tạo để train tiếp | Khi muốn đổi base model |
| `train.epochs` | `200` | Số epoch tối đa | Khi thay đổi chiến lược train |
| `train.imgsz` | `640` | Kích thước ảnh train | Khi model/input size thay đổi |
| `train.batch` | `64` | Batch size | Khi thiếu VRAM hoặc muốn tăng throughput |
| `train.workers` | `4` | Số dataloader workers | Khi I/O chậm hoặc lỗi multiprocess |
| `train.patience` | `10` | Early stopping patience | Khi muốn train dừng sớm hoặc lâu hơn |
| `train.device` | `[0]` | GPU dùng để train | Khi chuyển GPU train |
| `train.project` | `Helmet_Detection_Project` | Thư mục project output và experiment name logic | Khi muốn tách run train |
| `train.name` | `yolov8_binary_class` | Tên run hiện tại | Khi tạo run mới riêng biệt |
| `mlflow.tracking_uri` | `http://localhost:5001` | MLflow server URI | Khi MLflow nằm ở server khác |
| `mlflow.registry_name` | `YOLOv8_Helmet_Model` | Tên model trong registry | Khi quản lý nhiều model khác nhau |
| `mlflow.deploy_alias` | `Production` | Alias model đang active để deploy | Khi dùng `Staging`, `Candidate` hoặc alias khác |

## 7. Lệnh vận hành quan trọng

File: `Makefile`

| Target | Chức năng | Khi dùng |
|---|---|---|
| `run` | Chạy `vision-service` standalone với host network | Debug nhanh pipeline chính |
| `stack-up` | Dựng core stack bằng Docker Compose | Chạy hệ thống gần production |
| `prepare-data` | Chuẩn hoá dữ liệu và đóng gói shards | Chuẩn bị dữ liệu mới để retrain |
| `retrain` | Chạy DVC pipeline train/evaluate/export/compile | Huấn luyện lại model |
| `build-engine` | Biên dịch ONNX sang TensorRT engine | Sau khi có model ONNX mới |
| `deploy-model` | Deploy engine mới nếu quality gate đạt | Sau retrain thành công |
| `rollback` | Quay về model version cũ theo MLflow version | Khi deploy mới có vấn đề |
| `mlops-pipeline` | Chạy toàn bộ pipeline end-to-end | Khi muốn automation đầy đủ |

## 8. Monitoring

Files:

- `monitoring/prometheus/prometheus.yml`
- `monitoring/grafana/provisioning/datasources/prometheus.yml`

| Key | Giá trị hiện tại | Ý nghĩa | Khi nào cần sửa |
|---|---|---|---|
| `global.scrape_interval` | `5s` | Tần suất Prometheus scrape metrics | Khi cần metrics chi tiết hơn hoặc giảm tải |
| `job vision_service target` | `host.docker.internal:9100` | Endpoint metrics hiện tại của `vision-service` | Khi đổi cách chạy app hoặc đổi port |
| `datasource.url` | `http://prometheus:9090` | URL Grafana query Prometheus | Khi Prometheus nằm nơi khác |

## 9. Environment variables quan trọng

Files:

- `.env.example`
- `apps/vision_service/.env.example`

| Env | Giá trị mẫu | Ý nghĩa | Khi nào cần sửa |
|---|---|---|---|
| `GPU_ID` | `0` | GPU mặc định cho runtime/train tùy context | Khi đổi GPU |
| `CONFIG_DIR` | `apps/vision_service/configs` | Thư mục config cho app | Khi đổi cấu trúc thư mục config |
| `TELEGRAM_BOT_TOKEN` | `your_bot_token_here` | Token Telegram bot | Khi bật cảnh báo thật |
| `TELEGRAM_CHAT_ID` | `your_chat_id_here` | Chat ID nhận cảnh báo | Khi đổi người/group nhận alert |
| `REDIS_CHANNEL` | `helmet_violations` | Topic Redis cho worker | Khi đổi pipeline alert |
| `TELEGRAM_SNAPSHOT_SOURCE` | `probe` | Nguồn snapshot mặc định | Khi muốn fallback RTMP/HLS |

## 10. Nếu chỉ xem 6 file

Nếu bạn cần hiểu nhanh project, hãy đọc theo thứ tự:

1. `apps/vision_service/configs/app.yaml`
2. `apps/vision_service/configs/camera/*.yaml`
3. `apps/vision_service/configs/infer/pgie_yolov8_helmet.txt`
4. `docker-compose.yml`
5. `params.yaml`
6. `Makefile`

