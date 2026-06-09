# CS317_MLOps Configuration Summary

Tài liệu này tổng hợp toàn bộ bề mặt cấu hình chính của project `CS317_MLOps`, gồm runtime, MLOps pipeline, Docker/Compose, monitoring, dataset và tooling.

## 1. Thứ tự nạp cấu hình

`vision-service` nạp cấu hình tại `apps/vision_service/src/settings.py` theo thứ tự:

1. Đọc `CONFIG_DIR/app.yaml`
2. Ghi đè một phần bằng biến môi trường
3. Resolve các path tương đối theo root project
4. Quét `streams.scan_camera_dir` để nạp các file camera `*.yaml`
5. Chỉ giữ camera `enabled: true`
6. Mặc định lọc camera không reachable nếu `CHECK_CAMERA_REACHABILITY=true`

Các env override đang được code hỗ trợ trực tiếp:

- `APP_NAME`
- `APP_ENV`
- `LOG_LEVEL`
- `GPU_ID`
- `RTMP_LOCATION`
- `TELEGRAM_ENABLED`
- `TELEGRAM_SNAPSHOT_SOURCE`
- `TELEGRAM_REDIS_HOST`
- `TELEGRAM_REDIS_PORT`
- `TELEGRAM_REDIS_TOPIC`
- `TELEGRAM_COOLDOWN_SEC`
- `TELEGRAM_MIN_CONSEC_NO_HELMET_FRAMES`
- `TELEGRAM_SNAPSHOT_DIR`
- `TELEGRAM_SNAPSHOT_RTMP_URL`
- `TELEGRAM_SNAPSHOT_HLS_URL`
- `CONFIG_DIR`
- `CHECK_CAMERA_REACHABILITY`

## 2. Runtime config chính

### 2.1 Vision service

Nguồn chính: `apps/vision_service/configs/app.yaml`

#### `app`

| Key | Giá trị hiện tại | Ý nghĩa |
|---|---|---|
| `name` | `uit-medseg-vision` | Tên logical của service, thường xuất hiện trong log và metadata runtime. |
| `env` | `development` | Môi trường vận hành hiện tại của ứng dụng. |
| `log_level` | `INFO` | Mức chi tiết log mà app sẽ ghi ra stdout/stderr. |
| `gpu_id` | `0` | GPU mặc định mà pipeline DeepStream ưu tiên sử dụng. |

#### `storage`, `events`, `streams`

| Key | Giá trị hiện tại | Ý nghĩa |
|---|---|---|
| `storage.logs_dir` | `storage/logs` | Thư mục gốc chứa log cục bộ do app sinh ra. |
| `events.output_file` | `storage/logs/events.jsonl` | File JSONL ghi lại các sự kiện/vi phạm theo từng dòng. |
| `streams.scan_camera_dir` | `camera` | Thư mục con chứa các file YAML camera để app tự quét khi khởi động. |

#### `pipeline`

| Key | Giá trị hiện tại | Ý nghĩa |
|---|---|---|
| `streammux_width` | `960` | Chiều rộng frame sau khi được `nvstreammux` chuẩn hoá trước infer. |
| `streammux_height` | `544` | Chiều cao frame sau khi được `nvstreammux` chuẩn hoá trước infer. |
| `batched_push_timeout_usec` | `40000` | Thời gian chờ tối đa để gom batch trước khi đẩy sang bước kế tiếp. |
| `max_sources` | `16` | Số nguồn camera tối đa mà pipeline dự kiến hỗ trợ. |
| `sink` | `rtmp` | Loại đầu ra cuối của pipeline; hiện tại publish sang RTMP. |
| `frame_log_interval_sec` | `5.0` | Chu kỳ in log thống kê tốc độ xử lý frame/FPS. |

#### `tiler`

| Key | Giá trị hiện tại | Ý nghĩa |
|---|---|---|
| `enabled` | `true` | Bật ghép nhiều camera thành một khung tổng trước khi xuất hình. |
| `rows` | `0` | Số hàng tiler; `0` nghĩa là để app tự tính theo số camera. |
| `cols` | `0` | Số cột tiler; `0` nghĩa là để app tự tính theo số camera. |
| `width` | `1920` | Chiều rộng của khung mosaic sau khi ghép nhiều nguồn. |
| `height` | `1080` | Chiều cao của khung mosaic sau khi ghép nhiều nguồn. |

#### `infer`

| Key | Giá trị hiện tại | Ý nghĩa |
|---|---|---|
| `enabled` | `true` | Bật khối suy luận YOLO trong pipeline. |
| `config_file` | `/workspace/apps/vision_service/configs/infer/pgie_yolov8_helmet.txt` | File cấu hình `nvinfer` mà DeepStream sẽ dùng để load model. |
| `unique_id` | `1` | ID của GIE trong metadata DeepStream để các probe tham chiếu đúng model. |
| `summary_interval_sec` | `5.0` | Chu kỳ in log tổng hợp về infer. |
| `emit_frame_events` | `false` | Nếu bật, app có thể phát event theo từng frame thay vì chỉ theo logic vi phạm. |

#### `visualization`

| Key | Giá trị hiện tại | Ý nghĩa |
|---|---|---|
| `enabled` | `true` | Bật lớp hiển thị OSD lên video đầu ra. |
| `display_text` | `true` | Hiển thị text label/trạng thái lên khung hình. |
| `display_bbox` | `true` | Hiển thị bounding box quanh đối tượng phát hiện. |
| `display_clock` | `false` | Quyết định có chèn đồng hồ lên video hay không. |
| `osd_process_mode` | `0` | Chế độ xử lý của `nvdsosd`; giữ nguyên theo cấu hình gốc của project. |

#### `rtsp`

| Key | Giá trị hiện tại | Ý nghĩa |
|---|---|---|
| `enabled` | `false` | Bật/tắt nhánh xuất RTSP của pipeline. |
| `host` | `127.0.0.1` | Địa chỉ bind cho RTSP server nội bộ. |
| `udp_port` | `5400` | Cổng UDP nội bộ dùng để chuyển RTP sang RTSP server. |
| `rtsp_port` | `8554` | Cổng mà client RTSP sẽ truy cập. |
| `mount_point` | `/vision` | Đường dẫn mount của stream RTSP, ví dụ `rtsp://host:8554/vision`. |
| `codec` | `h264` | Codec video dùng cho nhánh RTSP. |
| `bitrate` | `2500000` | Bitrate mã hoá của stream RTSP. |
| `iframe_interval` | `15` | Khoảng cách giữa các keyframe/GOP. |
| `payload_type` | `96` | RTP payload type cho luồng video. |
| `rtp_mtu` | `1400` | Kích thước gói RTP tối đa trước khi phân mảnh. |
| `udp_buffer_size` | `2097152` | Kích thước buffer UDP cho nhánh streaming. |
| `sps_pps_interval` | `-1` | Tần suất chèn SPS/PPS vào H264; `-1` giữ theo mặc định encoder. |
| `rtsp_repay_enabled` | `true` | Bật nhánh tái đóng gói RTP/RTSP trước khi phục vụ client. |
| `rtsp_repay_jitter_latency_ms` | `0` | Độ trễ bộ đệm jitter cho nhánh RTSP repayload. |
| `rtsp_repay_jitter_drop_on_latency` | `true` | Cho phép bỏ frame/gói quá trễ để giữ độ trễ thấp. |
| `rtsp_repay_leaky_queue_enabled` | `true` | Cho phép queue bỏ bớt dữ liệu khi downstream RTSP chậm. |
| `rtsp_transport` | `tcp` | Giao thức RTSP transport ưu tiên khi client kết nối. |
| `udpsink_sync` | `true` | Buộc sink UDP đồng bộ theo clock pipeline. |
| `udpsink_async` | `false` | Không chuyển sink UDP sang async state change. |
| `udpsink_qos` | `false` | Tắt QoS feedback ở sink UDP để tránh làm pipeline tự điều chỉnh ngoài ý muốn. |
| `debug_h264_output_file` | `""` | File dump H264 để debug; để rỗng nghĩa là không ghi file debug. |

#### `rtmp`

| Key | Giá trị hiện tại | Ý nghĩa |
|---|---|---|
| `enabled` | `true` | Bật nhánh publish RTMP. |
| `location` | `rtmp://mediamtx:1935/vision1` | URL RTMP mà pipeline sẽ publish video tới. |
| `sink_sync` | `false` | Không ép sink RTMP chờ clock realtime tuyệt đối. |
| `sink_async` | `false` | Không dùng async state change cho sink RTMP. |
| `streamable_mux` | `true` | Tạo FLV/RTMP ở chế độ streamable để client nhận dữ liệu ngay. |

#### `tracker`

| Key | Giá trị hiện tại | Ý nghĩa |
|---|---|---|
| `enabled` | `true` | Bật bộ theo dõi đối tượng sau bước detection. |
| `gpu_id` | `0` | GPU dùng riêng cho tracker. |
| `tracker_width` | `640` | Độ rộng nội bộ mà tracker dùng để tính toán. |
| `tracker_height` | `640` | Độ cao nội bộ mà tracker dùng để tính toán. |
| `ll_lib_file` | `/opt/nvidia/deepstream/deepstream-6.4/lib/libnvds_nvmultiobjecttracker.so` | Thư viện tracker low-level của DeepStream. |
| `ll_config_file` | `/opt/nvidia/deepstream/deepstream-6.4/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml` | File cấu hình thuật toán NvDCF. |
| `display_tracking_id` | `false` | Quyết định có vẽ ID track lên màn hình hay không. |

#### `metrics`

| Key | Giá trị hiện tại | Ý nghĩa |
|---|---|---|
| `enabled` | `true` | Bật exporter Prometheus trong process `vision-service`. |
| `port` | `9100` | Cổng HTTP mà exporter metrics lắng nghe. |

#### `telegram`

| Key | Giá trị hiện tại | Ý nghĩa |
|---|---|---|
| `enabled` | `true` | Bật logic tạo alert, snapshot và publish sang Redis cho Telegram worker. |
| `snapshot_source` | `probe` | Nguồn lấy ảnh chụp vi phạm; hiện ưu tiên lấy ngay từ probe trong pipeline. |
| `redis_host` | `redis` | Host Redis nhận event alert. |
| `redis_port` | `6379` | Cổng Redis nhận event alert. |
| `redis_topic` | `helmet_violations` | Tên channel/topic Redis dùng để publish alert. |
| `cooldown_sec` | `5.0` | Thời gian chờ tối thiểu giữa hai alert liên tiếp cho cùng logic. |
| `min_consecutive_no_helmet_frames` | `3` | Số frame vi phạm liên tiếp cần đủ trước khi phát alert. |
| `snapshot_dir` | `/workspace/storage/snapshots` | Thư mục lưu file ảnh chụp vi phạm. |
| `snapshot_rtmp_url` | `rtmp://mediamtx:1935/vision1` | URL RTMP fallback để lấy snapshot nếu không dùng probe. |
| `snapshot_hls_url` | `http://mediamtx:8888/vision1/index.m3u8?cookieCheck=1` | URL HLS fallback để lấy snapshot nếu RTMP không sẵn sàng. |

### 2.2 DeepStream infer config

Nguồn:

- `apps/vision_service/configs/infer/pgie_yolov8_helmet.txt`
- `apps/vision_service/configs/infer/pgie_yolov8_helmet_b3.txt`

Giá trị chính:

| Key | Bản `batch-size=1` | Bản `batch-size=3` | Ý nghĩa |
|---|---|---|---|
| `gpu-id` | `0` | `0` | GPU mà `nvinfer` chạy trên đó. |
| `onnx-file` | `/workspace/apps/vision_service/models/yolov8/yolov8_helmet.onnx` | giống nhau | File model ONNX nguồn để rebuild engine khi cần. |
| `model-engine-file` | `/workspace/apps/vision_service/models/yolov8/yolov8_helmet_active.engine` | giống nhau | TensorRT engine thực tế được dùng khi infer. |
| `labelfile-path` | `/workspace/apps/vision_service/models/yolov8/labels.txt` | giống nhau | File ánh xạ class ID sang tên lớp. |
| `batch-size` | `1` | `3` | Số input được suy luận trong một batch. |
| `network-mode` | `2` | `2` | Chế độ precision của model; `2` tương ứng FP16 trong DeepStream. |
| `num-detected-classes` | `2` | `2` | Tổng số class mà model sinh ra. |
| `gie-unique-id` | `1` | `1` | ID metadata của khối infer trong pipeline. |
| `infer-dims` | `3;640;640` | `3;640;640` | Kích thước tensor đầu vào theo dạng `C;H;W`. |
| `maintain-aspect-ratio` | `1` | `1` | Giữ tỉ lệ gốc của ảnh khi resize vào input tensor. |
| `symmetric-padding` | `1` | `1` | Padding cân đối hai bên khi ảnh không cùng tỉ lệ với input model. |
| `parse-bbox-func-name` | `NvDsInferParseYolo` | `NvDsInferParseYolo` | Hàm C++ custom dùng để parse output YOLO thành bbox. |
| `custom-lib-path` | `/workspace/apps/vision_service/libs/deepstream/lib/libnvdsinfer_custom_impl_Yolo.so` | giống nhau | Thư viện `.so` chứa parser custom cho YOLO. |
| `topk` | `100` | `100` | Giới hạn số detection tối đa giữ lại sau parse ban đầu. |
| `nms-iou-threshold` | `0.55` | `0.55` | Ngưỡng IoU dùng cho Non-Maximum Suppression. |
| `pre-cluster-threshold` | `0.25` | `0.25` | Ngưỡng confidence tối thiểu trước khi vào bước NMS/clustering. |

### 2.3 Camera config hiện tại

Nguồn: `apps/vision_service/configs/camera/*.yaml`

| File | Camera ID | Enabled | Stream type | URI | Min conf | ROI | Ý nghĩa |
|---|---|---|---|---|---|---|---|
| `camera.yaml` | `cam_001` | `true` | `hls` | `http://192.168.28.78:8080/210743ddafb0305394d540bb2dd29c35/hls/gqbb9Lhhcu/xNrdIEOExv/s.m3u8` | `0.5` | bật cấu hình polygon nhưng `enabled=false` | Camera thật trong LAN, đóng vai trò nguồn live chính của hệ thống. |
| `camera_002.yaml` | `cam_002` | `true` | `hls` | `http://mediamtx:8888/cam02/index.m3u8?cookieCheck=1` | `0.5` | rỗng | Camera giả lập qua MediaMTX để test đa nguồn và scale. |
| `camera_003.yaml` | `cam_003` | `true` | `hls` | `http://mediamtx:8888/cam03/index.m3u8?cookieCheck=1` | `0.5` | rỗng | Camera giả lập thứ ba để kiểm thử layout và throughput. |
| `camera_004.yaml` | `cam_004` | `true` | `hls` | `http://mediamtx:8888/cam04/index.m3u8?cookieCheck=1` | `0.5` | rỗng | Camera giả lập thứ tư để scale test đủ bốn nguồn. |

Schema camera:

| Key | Mặc định / ràng buộc | Ý nghĩa |
|---|---|---|
| `stream.type` | `hls | file | rtsp` | Chỉ loại nguồn video mà camera được phép dùng. |
| `uri` | phải bắt đầu bằng `http://`, `https://`, `file://` hoặc `rtsp://` | Địa chỉ thực tế của luồng hoặc file video đầu vào. |
| `reconnect_interval_sec` | `10` | Chu kỳ chờ trước khi thử kết nối lại nếu nguồn bị mất. |
| `timeout_sec` | `15` | Thời gian timeout khi đọc nguồn camera. |
| `decoder_drop_frame_interval` | `0` | Khoảng bỏ frame ở decoder để giảm tải; `0` nghĩa là không bỏ. |
| `loop` | `true`, chỉ áp dụng cho `file` | Cho phép video file chạy lặp lại khi phát xong. |

## 3. Environment variables

### 3.1 Root env template

Nguồn: `.env.example`

| Key | Giá trị mẫu | Ý nghĩa |
|---|---|---|
| `APP_ENV` | `development` | Môi trường chạy mặc định khi khởi động project. |
| `APP_NAME` | `MLOps` | Tên app hiển thị/logical ở mức env. |
| `LOG_LEVEL` | `INFO` | Mức log mặc định nếu không override nơi khác. |
| `GPU_ID` | `0` | GPU mặc định dùng khi chạy container/script có hỗ trợ GPU. |
| `CONFIG_DIR` | `apps/vision_service/configs` | Thư mục config mà app sẽ đọc nếu dùng đường dẫn tương đối trên host. |
| `TELEGRAM_BOT_TOKEN` | `your_bot_token_here` | Token của bot Telegram để gửi thông báo. |
| `TELEGRAM_CHAT_ID` | `your_chat_id_here` | Chat hoặc group đích nhận alert Telegram. |
| `REDIS_CHANNEL` | `helmet_violations` | Channel Redis mà worker sẽ subscribe hoặc publish alert. |
| `TELEGRAM_MIN_CONSEC_NO_HELMET_FRAMES` | `3` | Số frame vi phạm liên tiếp cần đủ trước khi bắn cảnh báo. |
| `TELEGRAM_SNAPSHOT_SOURCE` | `probe` | Nguồn lấy snapshot mặc định cho alert. |

### 3.2 Vision service env template

Nguồn: `apps/vision_service/.env.example`

| Key | Giá trị mẫu | Ý nghĩa |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | `your_bot_token_here` | Thông tin bí mật để Telegram worker có thể gọi Bot API. |
| `TELEGRAM_CHAT_ID` | `your_chat_id_here` | Đích nhận cảnh báo từ worker. |
| `APP_NAME` | comment only | Cho phép override tên app nếu bỏ comment và set giá trị thật. |
| `APP_ENV` | comment only | Cho phép override môi trường chạy so với `app.yaml`. |
| `LOG_LEVEL` | comment only | Cho phép đổi mức log mà không sửa YAML. |
| `GPU_ID` | comment only | Cho phép ép service dùng GPU khác mà không sửa YAML. |

### 3.3 Env được truyền khi chạy `make run`

Nguồn: `Makefile`

| Key | Giá trị mặc định khi không set | Ý nghĩa |
|---|---|---|
| `APP_ENV` | `development` | Gắn môi trường chạy cho container standalone. |
| `LOG_LEVEL` | `INFO` | Điều chỉnh độ chi tiết log khi chạy `make run`. |
| `GPU_ID` | `0` | GPU ID bên trong container DeepStream. |
| `CONFIG_DIR` | `/workspace/apps/vision_service/configs` | Nơi app tìm `app.yaml` và camera configs trong container. |
| `RTMP_LOCATION` | `rtmp://127.0.0.1:1935/vision1 live=1` | Đích publish RTMP khi chạy standalone bằng host networking. |
| `TELEGRAM_ENABLED` | `true` | Bật logic alert/snapshot trong `vision-service`. |
| `TELEGRAM_REDIS_HOST` | `127.0.0.1` | Redis host mà container standalone sẽ publish alert tới. |
| `TELEGRAM_REDIS_PORT` | `6379` | Redis port tương ứng với host bên trên. |
| `TELEGRAM_REDIS_TOPIC` | `helmet_violations` | Tên topic Redis khi chạy standalone. |
| `TELEGRAM_MIN_CONSEC_NO_HELMET_FRAMES` | `3` | Ngưỡng frame vi phạm liên tiếp cho alert. |
| `TELEGRAM_SNAPSHOT_SOURCE` | `probe` | Chọn cách lấy ảnh vi phạm trong luồng standalone. |
| `TELEGRAM_SNAPSHOT_DIR` | `/workspace/storage/snapshots` | Nơi lưu snapshot trên volume repo mount vào container. |
| `TELEGRAM_SNAPSHOT_RTMP_URL` | `rtmp://127.0.0.1:1935/vision1` | Nguồn RTMP fallback để chụp snapshot khi cần. |
| `TELEGRAM_SNAPSHOT_HLS_URL` | `http://127.0.0.1:8888/vision1/index.m3u8` | Nguồn HLS fallback để chụp snapshot khi cần. |
| `PYTHONPATH` | `/workspace` | Đảm bảo module `apps.*` import được trong container. |

### 3.4 Env của Docker Compose

Nguồn: `docker-compose.yml`

#### `vision-service`

| Key | Giá trị | Ý nghĩa |
|---|---|---|
| `APP_ENV` | `${APP_ENV:-development}` | Môi trường chạy được inject vào service compose. |
| `APP_NAME` | `uit-medseg-vision` | Tên app trong container compose. |
| `LOG_LEVEL` | `${LOG_LEVEL:-INFO}` | Mức log cho service compose. |
| `GPU_ID` | `${GPU_ID:-0}` | GPU ID dùng cho service compose. |
| `CONFIG_DIR` | `/workspace/apps/vision_service/configs` | Đường dẫn config bên trong container. |
| `NVIDIA_DRIVER_CAPABILITIES` | `compute,utility,video,graphics` | Khai báo capability GPU cần cấp cho container. |
| `PYTHONPATH` | `/workspace` | Đảm bảo Python import đúng package trong repo mount. |
| `TELEGRAM_MIN_CONSEC_NO_HELMET_FRAMES` | `${TELEGRAM_MIN_CONSEC_NO_HELMET_FRAMES:-3}` | Ngưỡng frame vi phạm cho logic alert. |
| `MIN_CONSECUTIVE_NO_HELMET_FRAMES` | `${MIN_CONSECUTIVE_NO_HELMET_FRAMES:-3}` | Biến backward-compatible cho cùng một logic threshold. |
| `TELEGRAM_SNAPSHOT_SOURCE` | `${TELEGRAM_SNAPSHOT_SOURCE:-probe}` | Chọn nguồn snapshot mặc định trong mode compose. |

#### `telegram-worker`

| Key | Giá trị | Ý nghĩa |
|---|---|---|
| `REDIS_HOST` | `redis` | Host Redis mà worker kết nối để nhận alert. |
| `REDIS_PORT` | `6379` | Cổng Redis tương ứng. |
| `REDIS_CHANNEL` | `${REDIS_CHANNEL:-helmet_violations}` | Channel Pub/Sub mà worker subscribe. |
| `TELEGRAM_BOT_TOKEN` | `${TELEGRAM_BOT_TOKEN:-}` | Token bot dùng để gọi Telegram API. |
| `TELEGRAM_CHAT_ID` | `${TELEGRAM_CHAT_ID:-}` | ID chat/group nhận tin nhắn cảnh báo. |
| `DEBUG_ALERT_TEXT_FIRST` | `${DEBUG_ALERT_TEXT_FIRST:-false}` | Cờ debug để ưu tiên gửi text alert trước trong một số luồng kiểm thử. |

#### `grafana`

| Key | Giá trị | Ý nghĩa |
|---|---|---|
| `GF_SECURITY_ADMIN_USER` | `${GF_ADMIN_USER:-admin}` | Username admin mặc định của Grafana. |
| `GF_SECURITY_ADMIN_PASSWORD` | `${GF_ADMIN_PASSWORD:-admin}` | Password admin mặc định của Grafana. |
| `GF_USERS_ALLOW_SIGN_UP` | `false` | Không cho phép người dùng tự đăng ký tài khoản mới. |
| `GF_SERVER_ROOT_URL` | `http://localhost:3005` | Base URL mà Grafana dùng để render link nội bộ. |
| `GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH` | `/var/lib/grafana/dashboards/helmet_detection.json` | Dashboard mặc định khi mở Grafana lần đầu. |

## 4. Docker, services và cổng mạng

Nguồn: `docker-compose.yml`

| Service | Image | Cổng | Ý nghĩa |
|---|---|---|---|
| `mlflow-server` | `uit_medseg/mlflow-server:dev` | `5001:5001` | UI và backend tracking cho thí nghiệm/train/model registry. |
| `redis` | `redis:7-alpine` | `6379:6379` | Message broker cho luồng alert thời gian thực. |
| `mediamtx` | `bluenviron/mediamtx:latest` | `8554:8554`, `1935:1935`, `8888:8888`, `9997:9997` | Relay media nhận RTMP và phục vụ RTSP/HLS/API. |
| `vision-service` | `uit_medseg/vision-service:dev` | `9105:9100` | Service DeepStream chính chạy infer, tracking và xuất metrics. |
| `telegram-worker` | `uit_medseg/telegram-worker:dev` | không publish | Worker nền đọc alert từ Redis rồi gửi Telegram. |
| `prometheus` | `prom/prometheus:v2.51.0` | `9091:9090` | Hệ thống scrape và lưu trữ metrics monitoring. |
| `grafana` | `grafana/grafana:10.4.2` | `3005:3000` | Dashboard trực quan hoá cho Prometheus metrics. |

Ghi chú:

- `vision-service` dùng `shm_size: 2g`
- `vision-service` bind mount toàn bộ repo vào `/workspace`
- `vision-service` reserve GPU theo `device_ids: ['${GPU_ID:-0}']`
- `telegram-worker` mount `storage` vào `/workspace/storage`

## 5. MediaMTX và simulator

### 5.1 MediaMTX

Nguồn: `configs/mediamtx.yml`

| Key | Giá trị | Ý nghĩa |
|---|---|---|
| `logLevel` | `info` | Mức chi tiết log của MediaMTX. |
| `logDestinations` | `[stdout]` | Nơi MediaMTX ghi log; hiện chỉ in ra stdout container. |
| `api` | `yes` | Bật HTTP API nội bộ của MediaMTX. |
| `apiAddress` | `:9997` | Cổng API quản trị của MediaMTX. |
| `rtspAddress` | `:8554` | Cổng phục vụ RTSP client. |
| `rtmpAddress` | `:1935` | Cổng nhận/publish RTMP. |
| `hlsAddress` | `:8888` | Cổng phục vụ HLS và một số endpoint HTTP media. |
| `paths."~^.*$".source` | `publisher` | Mọi path đều chấp nhận nguồn do publisher đẩy lên thay vì pull từ upstream cố định. |

### 5.2 RTSP/HLS simulator env

Nguồn: `scripts/rtsp_sim_publishers.sh`

| Key | Default | Ý nghĩa |
|---|---|---|
| `RTSP_SIM_STATE_DIR` | `storage/rtsp_sim` | Thư mục lưu PID/state file của publisher giả lập. |
| `RTSP_SIM_LOG_DIR` | `storage/logs/rtsp_sim` | Thư mục log cho từng publisher camera giả lập. |
| `FFMPEG_BIN` | `ffmpeg` | Binary ffmpeg dùng khi chạy trực tiếp trên host. |
| `RTSP_SIM_HOST` | `127.0.0.1` | Host đích cho RTSP simulator mode. |
| `RTSP_SIM_PORT` | `8554` | Cổng RTSP publish/serve trong mode RTSP. |
| `RTSP_SIM_TRANSPORT` | `tcp` | Transport RTSP mà ffmpeg sử dụng để publish. |
| `RTSP_SIM_RTMP_HOST` | `127.0.0.1` | Host đích cho mode publish RTMP. |
| `RTSP_SIM_RTMP_PORT` | `1935` | Cổng RTMP publish cho simulator. |
| `RTSP_SIM_HLS_HOST` | `127.0.0.1` | Host phục vụ HLS được publish qua MediaMTX. |
| `RTSP_SIM_HLS_PORT` | `8888` | Cổng HLS tương ứng của MediaMTX. |
| `RTSP_SIM_FFMPEG_IMAGE` | `jrottenberg/ffmpeg:8-scratch` | Docker image ffmpeg fallback nếu không chạy ffmpeg host. |
| `RTSP_SIM_FORCE_DOCKER_FFMPEG` | `0` | Buộc dùng ffmpeg trong Docker thay vì binary host. |
| `RTSP_SIM_PUBLISHER_PREFIX` | `uit_medseg_rtsp_pub` | Prefix tên container publisher khi chạy bằng Docker. |
| `RTSP_SIM_CAM01_FILE` | `data/test.mp4` | Video nguồn mặc định cho camera giả lập `cam01`. |
| `RTSP_SIM_CAM02_FILE` | `data/cam2_fake.mp4` | Video nguồn mặc định cho camera giả lập `cam02`. |
| `RTSP_SIM_CAM03_FILE` | `data/cam3_fake.mp4` | Video nguồn mặc định cho camera giả lập `cam03`. |
| `RTSP_SIM_CAM04_FILE` | `data/cam4_fake.mp4` | Video nguồn mặc định cho camera giả lập `cam04`. |
| `PROTOCOL` | `rtsp` | Chọn simulator chạy theo RTSP thuần hay push RTMP để MediaMTX phát HLS. |

### 5.3 Standalone MediaMTX simulator env

Nguồn: `scripts/rtsp_sim_mediamtx.sh`

| Key | Default | Ý nghĩa |
|---|---|---|
| `RTSP_SIM_CONTAINER_NAME` | `uit_medseg_rtsp_sim` | Tên container MediaMTX khi chạy simulator độc lập. |
| `RTSP_SIM_MEDIAMTX_IMAGE` | `bluenviron/mediamtx:latest` | Image MediaMTX dùng cho simulator độc lập. |
| `RTSP_SIM_RTSP_PORT` | `8554` | Cổng RTSP publish ra host cho simulator độc lập. |
| `RTSP_SIM_RTMP_PORT` | `1935` | Cổng RTMP publish ra host cho simulator độc lập. |
| `RTSP_SIM_API_PORT` | `8888` | Cổng HTTP/HLS/API publish ra host cho simulator độc lập. |

## 6. MLOps pipeline config

### 6.1 Makefile variables

Nguồn: `Makefile`

| Key | Default | Ý nghĩa |
|---|---|---|
| `IMAGE` | `uit_medseg/mlops_thuc:dev` | Image runtime chứa DeepStream và vision pipeline. |
| `MLOPS_TRAIN_IMAGE` | `uit_medseg/mlops_train:dev` | Image huấn luyện/evaluate/export cho YOLO và MLflow. |
| `PYTHON` | `python3` | Python executable dùng cho các script host-side trong Makefile. |
| `COMPOSE_ENV_FILE` | `apps/vision_service/.env` | File env mà `docker compose` sẽ nạp mặc định. |
| `MEDIAMTX_SCRIPT` | `bash scripts/rtsp_sim_mediamtx.sh` | Lệnh wrapper để quản lý MediaMTX simulator. |
| `PREP_RAW_DIR` | `dataset/data_new` | Thư mục raw data đầu vào của pipeline chuẩn hoá. |
| `PREP_YOLO_DIR` | `dataset/extracted/yolo_helmet_dataset_new` | Thư mục YOLO dataset tạm sau bước format/split. |
| `PREP_SPLIT_NAME` | `train` | Tên split mặc định khi format raw data mới. |
| `PREP_SEED` | `42` | Seed ngẫu nhiên để split/shuffle tái lập được. |
| `PREP_SPLIT_RATIOS` | rỗng | Cho phép override tỉ lệ train/val/test từ CLI. |
| `PREP_SYNC_FROM_STORAGE` | `1` | Quyết định có đồng bộ dữ liệu từ `storage` trước khi format hay không. |
| `PREP_EVENTS_FILE` | `storage/logs/events.jsonl` | File nguồn chứa log sự kiện để dựng data mới. |
| `PREP_SNAPSHOTS_DIR` | `storage/snapshots` | Thư mục ảnh snapshot dùng để tạo dataset mới. |
| `PREP_SYNC_MODE` | `replace` | Cách đồng bộ data mới vào thư mục đích; hiện là thay thế hoàn toàn. |
| `PREP_MIN_CONF` | `0.0` | Ngưỡng confidence tối thiểu khi trích dữ liệu từ storage sang dataset. |

### 6.2 Mục tiêu pipeline chính

| Target | Chuỗi gọi chính | Ý nghĩa |
|---|---|---|
| `prepare-data` | `format-data-new -> split-data-new -> pack-yolo-shards` | Chuẩn hoá dữ liệu mới và đóng gói lại thành shards phục vụ train. |
| `retrain` | start MLflow nếu cần, rồi `dvc repro extract train evaluate export compile` | Chạy trọn chuỗi retrain có quality gate và export model. |
| `build-engine` | compile ONNX sang TensorRT engine | Biên dịch model ONNX thành engine tối ưu cho DeepStream inference. |
| `rollback` | export theo `VERSION`, build engine, deploy rollback | Khôi phục nhanh về model version cũ đã có trong MLflow. |
| `mlops-pipeline` | `dvc pull -f -> prepare-data -> retrain -> deploy-model` | Pipeline end-to-end từ dữ liệu tới deploy model mới. |
| `deploy-model` | chạy `scripts/deploy_model.py` | Thực hiện bước CD thông minh dựa trên kết quả quality gate. |

### 6.3 DVC stages

Nguồn: `dvc.yaml`

| Stage | Command | Ý nghĩa |
|---|---|---|
| `extract` | `python scripts/extract_shards.py --shards-dir dataset/shards --output-dir dataset/extracted/yolo_helmet_dataset --limit ${sharding.extract_limit}` | Giải nén WebDataset shards về lại cấu trúc YOLO để train. |
| `train` | `docker run ... uit_medseg/mlops_train:dev python3 dataset/TrainYolo/train_yolo.py` | Huấn luyện YOLO trong container training chuẩn hoá. |
| `evaluate` | `docker run ... uit_medseg/mlops_train:dev python3 scripts/evaluate_model.py` | So sánh candidate với production trên Gold Standard dataset. |
| `export` | `docker run ... uit_medseg/mlops_train:dev python3 scripts/export_onnx.py` | Tải/promote model hợp lệ rồi export sang ONNX. |
| `compile` | `make build-engine` | Biên dịch ONNX vừa export thành TensorRT engine cho deploy. |

### 6.4 Params trung tâm

Nguồn: `params.yaml`

#### Dataset

| Key | Giá trị | Ý nghĩa |
|---|---|---|
| `dataset.root` | `dataset/raw/yolo_helmet_dataset` | Thư mục gốc của dataset raw/legacy được tham chiếu khi train. |
| `dataset.yaml` | `dataset/dataset.yaml` | File schema YOLO mô tả train/val/test và class names. |

#### Model

| Key | Giá trị | Ý nghĩa |
|---|---|---|
| `model.name` | `dataset/TrainYolo/runs/detect/helmet_training/run_12/weights/best.pt` | Checkpoint khởi tạo được dùng làm base weight cho lần train hiện tại. |

#### Sharding

| Key | Giá trị | Ý nghĩa |
|---|---|---|
| `sharding.output_dir` | `dataset/shards` | Nơi lưu các file `.tar` shards sau khi đóng gói. |
| `sharding.shard_size` | `1000` | Số mẫu tối đa trong mỗi shard. |
| `sharding.extract_limit` | `0` | Giới hạn số shard giải nén mỗi split; `0` nghĩa là giải nén tất cả. |
| `sharding.split_ratios` | `[0.8, 0.1, 0.1]` | Tỉ lệ chia train/val/test khi đóng gói dataset mới. |
| `sharding.selected_shards.train` | `["shard-*"]` | Pattern shard được phép dùng cho split train khi extract. |
| `sharding.selected_shards.val` | `["shard-*"]` | Pattern shard được phép dùng cho split val khi extract. |
| `sharding.selected_shards.test` | `["shard-*"]` | Pattern shard được phép dùng cho split test khi extract. |

#### Training

| Key | Giá trị | Ý nghĩa |
|---|---|---|
| `train.epochs` | `200` | Số epoch train tối đa. |
| `train.imgsz` | `640` | Kích thước ảnh đầu vào cho YOLO khi train/val. |
| `train.batch` | `64` | Batch size huấn luyện. |
| `train.workers` | `4` | Số worker DataLoader để đọc dữ liệu. |
| `train.patience` | `10` | Số epoch chờ improvement trước khi early stop. |
| `train.device` | `[0]` | Danh sách GPU device ID dùng cho train. |
| `train.project` | `Helmet_Detection_Project` | Tên thư mục project/result và cũng là experiment name logic. |
| `train.name` | `yolov8_binary_class` | Tên run cụ thể bên trong project train. |

#### MLflow

| Key | Giá trị | Ý nghĩa |
|---|---|---|
| `mlflow.tracking_uri` | `http://localhost:5001` | Địa chỉ MLflow server để log experiment và model. |
| `mlflow.registry_name` | `YOLOv8_Helmet_Model` | Tên model trong MLflow Model Registry. |
| `mlflow.deploy_alias` | `Production` | Alias version được xem là model đang sẵn sàng deploy. |

## 7. Dataset schema

### 7.1 Training dataset

Nguồn: `dataset/dataset.yaml`

| Key | Giá trị | Ý nghĩa |
|---|---|---|
| `path` | `dataset/extracted/yolo_helmet_dataset` | Thư mục root mà YOLO dùng để resolve các split bên dưới. |
| `train` | `images/train` | Đường dẫn tương đối tới ảnh train. |
| `val` | `images/val` | Đường dẫn tương đối tới ảnh validation. |
| `test` | `images/test` | Đường dẫn tương đối tới ảnh test. |
| `nc` | `2` | Số class trong bài toán detection. |
| `names.0` | `helmet` | Nhãn class ID `0`. |
| `names.1` | `no_helmet` | Nhãn class ID `1`. |

### 7.2 Gold standard dataset

Nguồn: `dataset/gold_standard.yaml`

| Key | Giá trị | Ý nghĩa |
|---|---|---|
| `path` | `dataset/gold_standard` | Root của tập chuẩn dùng cho quality gate. |
| `train` | `images` | Trỏ cùng thư mục ảnh để thoả schema YOLO dù không train trên tập này. |
| `val` | `images` | Trỏ cùng thư mục ảnh để hỗ trợ lệnh `val` của YOLO. |
| `test` | `images` | Tập ảnh thực tế được dùng khi evaluate quality gate. |
| `nc` | `2` | Số class của tập chuẩn. |
| `names.0` | `helmet` | Tên class an toàn. |
| `names.1` | `no_helmet` | Tên class vi phạm. |

## 8. Monitoring config

### 8.1 Prometheus

Nguồn: `monitoring/prometheus/prometheus.yml`

| Key | Giá trị | Ý nghĩa |
|---|---|---|
| `global.scrape_interval` | `5s` | Chu kỳ mặc định Prometheus đi lấy metrics. |
| `global.evaluation_interval` | `5s` | Chu kỳ đánh giá rule/recording rule. |
| `external_labels.project` | `uit-medseg-deepstream` | Nhãn toàn cục gắn vào mọi time series của project này. |
| `external_labels.env` | `development` | Nhãn môi trường gắn vào mọi time series. |
| `job vision_service target` | `host.docker.internal:9100` | Endpoint metrics của `vision-service` khi app chạy bằng host network. |
| `job vision_service metrics_path` | `/metrics` | Đường dẫn HTTP mà Prometheus sẽ scrape. |
| `job vision_service scrape_timeout` | `4s` | Timeout tối đa cho mỗi lần scrape `vision-service`. |
| `job prometheus target` | `localhost:9090` | Endpoint tự giám sát của chính Prometheus. |

### 8.2 Grafana

Nguồn:

- `monitoring/grafana/provisioning/datasources/prometheus.yml`
- `monitoring/grafana/provisioning/dashboards/dashboard.yml`

| Nhóm | Key | Giá trị | Ý nghĩa |
|---|---|---|---|
| datasource | `name` | `Prometheus` | Tên datasource hiển thị trong Grafana UI. |
| datasource | `uid` | `prometheus` | ID ổn định để dashboard tham chiếu datasource này. |
| datasource | `url` | `http://prometheus:9090` | URL nội bộ Grafana dùng để query Prometheus. |
| datasource | `isDefault` | `true` | Đặt Prometheus làm datasource mặc định cho dashboard mới. |
| datasource | `timeInterval` | `5s` | Bước thời gian tối thiểu Grafana gợi ý khi query. |
| dashboard provider | `name` | `helmet-detection` | Tên provider tự động nạp dashboard từ file. |
| dashboard provider | `updateIntervalSeconds` | `10` | Chu kỳ Grafana quét lại thư mục dashboard để reload. |
| dashboard provider | `allowUiUpdates` | `false` | Không cho sửa dashboard trực tiếp từ UI rồi ghi ngược ra file. |
| dashboard provider | `path` | `/var/lib/grafana/dashboards` | Thư mục trong container chứa dashboard JSON được mount vào. |

## 9. Web UI config

Nguồn:

- `apps/web_ui/app.py`
- `apps/web_ui/camera_config.py`

| Key | Default | Ý nghĩa |
|---|---|---|
| `DEEPSTREAM_API_HOST` | `127.0.0.1` | Host mà Web UI gọi để thêm/xoá stream động trong DeepStream. |
| `DEEPSTREAM_API_PORT` | `9091` | Cổng REST API của DeepStream mà dashboard sẽ sử dụng. |
| `MEDIAMTX_HOST` | `127.0.0.1` | Host MediaMTX để Web UI dựng URL phát lại stream. |
| `MEDIAMTX_PORT` | `8888` | Cổng HTTP/HLS của MediaMTX cho player/proxy trong Web UI. |
| `CAMERA_CONFIG_DIR` | `/workspace/apps/vision_service/configs/camera` | Thư mục YAML camera mà Web UI CRUD trực tiếp. |
| `snapshots mount` | `storage/snapshots` | Thư mục ảnh vi phạm được mount để dashboard có thể serve trực tiếp. |
| `violations DB path` | `storage/violations.db` | SQLite database mà dashboard đọc lịch sử vi phạm. |

Web UI cung cấp:

- CRUD camera config qua YAML
- Gọi DeepStream REST API `/stream/add` và `/stream/remove`
- Serve dashboard tĩnh
- Serve snapshot ảnh
- Đọc SQLite `violations.db`

## 10. Telegram worker config

Nguồn:

- `apps/telegram_worker/main.py`
- `apps/telegram_worker/requirements.txt`

| Key | Default | Ý nghĩa |
|---|---|---|
| `REDIS_HOST` | `redis` | Host Redis mà worker kết nối để nhận alert. |
| `REDIS_PORT` | `6379` | Cổng Redis của worker. |
| `REDIS_CHANNEL` | `helmet_violations` | Channel Pub/Sub mà worker đang lắng nghe. |
| `TELEGRAM_BOT_TOKEN` | không có mặc định | Thông tin bắt buộc để gọi Telegram Bot API. |
| `TELEGRAM_CHAT_ID` | không có mặc định | ID đích nhận tin nhắn/ảnh cảnh báo. |
| `DB_PATH` | `/workspace/storage/violations.db` | SQLite DB lưu lịch sử vi phạm mà worker ghi xuống. |

Dependencies:

- `redis>=5.0.0`
- `httpx>=0.25.0`

## 11. Packaging, dependency và tooling

### 11.1 Python package

Nguồn: `pyproject.toml`

| Key | Giá trị | Ý nghĩa |
|---|---|---|
| `project.name` | `uit-medseg` | Tên package Python của project. |
| `version` | `0.1.0` | Version package hiện tại. |
| `requires-python` | `>=3.8` | Phiên bản Python tối thiểu được hỗ trợ. |
| `script entrypoint` | `uit-medseg = apps.vision_service.src.main:main` | Lệnh CLI được expose khi cài package. |

### 11.2 Test và lint

| Tool | Config | Ý nghĩa |
|---|---|---|
| `pytest` | `testpaths=apps/vision_service/tests`, `--strict-markers`, `--tb=short` | Chuẩn hoá nơi đặt test và cách hiển thị lỗi khi chạy test. |
| `coverage` | source `apps/vision_service/src` | Chỉ tính coverage cho mã nguồn chính của vision service. |
| `black` | line length `100` | Quy tắc format code mặc định. |
| `isort` | profile `black`, line length `100` | Quy tắc sắp xếp import tương thích với `black`. |
| `ruff` | line length `100`, target `py38`, `fix=true` | Quy tắc lint/fix nhanh cho codebase Python. |

### 11.3 Host requirements

Nguồn: `requirements.txt`

Nhóm package chính:

- Core: `pydantic`, `pydantic-settings`, `PyYAML`
- Testing: `pytest`, `pytest-cov`
- HTTP/UI: `fastapi`, `uvicorn`
- Tooling: `python-dotenv`, `black`, `isort`, `flake8`, `mypy`
- ML: `onnx`, `onnxruntime`, `numpy`
- MLOps: `dvc`, `mlflow`, `webdataset`
- Monitoring/alerting: `prometheus-client`, `redis`, `opencv-python-headless`

### 11.4 VS Code workspace

Nguồn:

- `.vscode/launch.json`
- `.vscode/tasks.json`
- `.vscode/setting.json`

| File | Nội dung chính | Ý nghĩa |
|---|---|---|
| `launch.json` | cấu hình `Run Vision Service` | Cho phép chạy/debug service trực tiếp từ VS Code. |
| `tasks.json` | task `Run service`, `Run tests` | Tạo shortcut chạy app và test trong workspace. |
| `setting.json` | interpreter `.venv/bin/python`, `formatOnSave=true`, exclude cache files | Thiết lập môi trường Python và hành vi editor chung cho nhóm. |

## 12. Build images

### 12.1 Runtime image

Nguồn: `Dockerfile.ds64_glib`

| Thành phần | Giá trị | Ý nghĩa |
|---|---|---|
| Base image | `nvcr.io/nvidia/deepstream:6.4-samples-multiarch` | Nền tảng GPU/DeepStream chính để chạy vision pipeline. |
| Python deps | `pydantic`, `PyYAML`, `prometheus-client`, `redis`, `opencv-python-headless` | Các thư viện Python tối thiểu cần cho config, metrics, Redis và xử lý ảnh. |
| DeepStream Python binding | `pyds-1.1.10-py3-none-linux_x86_64.whl` | Binding Python để truy cập metadata DeepStream từ code Python. |
| CMD | `python3 /workspace/apps/vision_service/src/main.py` | Lệnh khởi động mặc định của image runtime. |

### 12.2 Training image

Nguồn: `Dockerfile.mlops`

| Thành phần | Giá trị | Ý nghĩa |
|---|---|---|
| Base image | `nvcr.io/nvidia/pytorch:22.10-py3` | Nền tảng train GPU với CUDA/PyTorch được NVIDIA đóng gói sẵn. |
| Torch | `2.1.0` | Phiên bản PyTorch được pin để tương thích với pipeline train/export. |
| Torchvision | `0.16.0` | Phiên bản torchvision đồng bộ với PyTorch bên trên. |
| Python deps | `ultralytics`, `mlflow`, `opencv-python-headless` | Bộ thư viện chính để train YOLO, log MLflow và xử lý ảnh. |
| Protobuf | `3.20.3` | Phiên bản protobuf pin để tránh lỗi tương thích với container base. |

### 12.3 MLflow image

Nguồn: `Dockerfile.mlflow`

| Thành phần | Giá trị | Ý nghĩa |
|---|---|---|
| Base image | `python:3.9-slim` | Image nhẹ để chạy riêng MLflow server. |
| Package | `mlflow` | Gói duy nhất cần để khởi động tracking/model registry server. |
| CMD | `mlflow server --host 0.0.0.0 --port 5001 --backend-store-uri sqlite:///mlruns/mlflow.db --default-artifact-root ./mlartifacts` | Lệnh khởi động MLflow với SQLite backend và artifact local. |

### 12.4 Telegram worker image

Nguồn: `apps/telegram_worker/Dockerfile`

| Thành phần | Giá trị | Ý nghĩa |
|---|---|---|
| Base image | `python:3.10-slim` | Image nhẹ dành riêng cho worker gửi Telegram. |
| Workdir | `/app` | Thư mục làm việc mặc định bên trong container worker. |
| CMD | `python main.py` | Lệnh khởi động worker subscriber. |

## 13. DVC remote

Nguồn: `.dvc/config`

| Key | Giá trị | Ý nghĩa |
|---|---|---|
| `cache.type` | `hardlink,symlink` | Cách DVC tối ưu lưu cache trên local filesystem để tiết kiệm dung lượng. |
| `cache.protected` | `true` | Bảo vệ cache khỏi bị chỉnh sửa trực tiếp từ working tree. |
| `core.remote` | `local_remote` | Remote mặc định mà DVC sẽ dùng khi `pull/push`. |
| `remote.local_remote.url` | `/mmlab_students/storageStudents/nguyenvd/uit_medseg/datasetMLOps` | Đường dẫn storage local đang giữ dữ liệu versioned của DVC. |

## 14. Các điểm cần chú ý

- `web_ui` mặc định gọi DeepStream REST API ở `127.0.0.1:9091`, nhưng `docker-compose.yml` hiện publish `9091` cho Prometheus container (`9091:9090`), không phải `vision-service`.
- `make run` dùng `RTMP_LOCATION=rtmp://127.0.0.1:1935/vision1 live=1`, trong khi `app.yaml` mặc định dùng `rtmp://mediamtx:1935/vision1`.
- Có hai file template env: `.env.example` và `apps/vision_service/.env.example`; file env compose mặc định mà `Makefile` trỏ tới là `apps/vision_service/.env`.
- `camera.yaml` đang chứa một URL HLS thật trong mạng LAN nội bộ, khác với `cam_002` đến `cam_004` đang dùng luồng giả lập qua MediaMTX.

## 15. Danh sách file config đã thống kê

- `Makefile`
- `docker-compose.yml`
- `.env.example`
- `apps/vision_service/.env.example`
- `apps/vision_service/configs/app.yaml`
- `apps/vision_service/configs/infer/pgie_yolov8_helmet.txt`
- `apps/vision_service/configs/infer/pgie_yolov8_helmet_b3.txt`
- `apps/vision_service/configs/camera/camera.yaml`
- `apps/vision_service/configs/camera/camera_002.yaml`
- `apps/vision_service/configs/camera/camera_003.yaml`
- `apps/vision_service/configs/camera/camera_004.yaml`
- `params.yaml`
- `dvc.yaml`
- `.dvc/config`
- `dataset/dataset.yaml`
- `dataset/gold_standard.yaml`
- `configs/mediamtx.yml`
- `monitoring/prometheus/prometheus.yml`
- `monitoring/grafana/provisioning/datasources/prometheus.yml`
- `monitoring/grafana/provisioning/dashboards/dashboard.yml`
- `pyproject.toml`
- `requirements.txt`
- `apps/telegram_worker/requirements.txt`
- `.vscode/launch.json`
- `.vscode/tasks.json`
- `.vscode/setting.json`
