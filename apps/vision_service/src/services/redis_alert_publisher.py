from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from apps.vision_service.src.logger import get_logger
from apps.vision_service.src.settings import RootSettings
from apps.vision_service.src.utils.file_utils import ensure_dir

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Persistent Frame Cache
# ---------------------------------------------------------------------------

class PersistentFrameCache:
    """
    Background thread giữ kết nối RTMP/HLS mở liên tục và lưu latest frame.

    Thay thế việc mở cv2.VideoCapture mới cho mỗi event (~4.5s) bằng cách
    lưu frame mới nhất trong memory và trả về ngay khi có yêu cầu (<1ms).
    """

    _RECONNECT_DELAY_SEC = 2.0
    _READ_TIMEOUT_MSEC = 5000

    def __init__(self, url: str, *, name: str = "PersistentFrameCache") -> None:
        self.url = url
        self._lock = threading.Lock()
        self._latest_frame: Any | None = None
        self._frame_ts: float = 0.0
        self._stop_event = threading.Event()
        self._connected = False

        self._thread = threading.Thread(
            target=self._reader_loop,
            name=name,
            daemon=True,
        )
        self._thread.start()
        logger.info("PersistentFrameCache started | url=%s", url)

    def get_frame(self) -> Tuple[Any | None, float]:
        """
        Trả về (frame_copy, age_ms).
        frame_copy là None nếu chưa có frame.
        """
        with self._lock:
            if self._latest_frame is None:
                return None, 0.0
            age_ms = (time.time() - self._frame_ts) * 1000
            return self._latest_frame.copy(), age_ms

    def stop(self) -> None:
        self._stop_event.set()

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _reader_loop(self) -> None:
        """Vòng lặp đọc frame liên tục, tự động reconnect khi mất kết nối."""
        try:
            import cv2  # noqa: PLC0415
        except ImportError:
            logger.error("cv2 not installed — PersistentFrameCache disabled")
            return

        while not self._stop_event.is_set():
            cap = None
            try:
                logger.info("[LATENCY][PERSISTENT_STREAM_CONNECT] url=%s", self.url)
                cap = cv2.VideoCapture(self.url)

                if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self._READ_TIMEOUT_MSEC)

                if not cap.isOpened():
                    logger.warning("[LATENCY][PERSISTENT_STREAM_CONNECT_FAILED] url=%s", self.url)
                    self._connected = False
                    self._stop_event.wait(self._RECONNECT_DELAY_SEC)
                    continue

                self._connected = True
                logger.info("[LATENCY][PERSISTENT_STREAM_CONNECTED] url=%s", self.url)

                consecutive_failures = 0
                while not self._stop_event.is_set():
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        with self._lock:
                            self._latest_frame = frame
                            self._frame_ts = time.time()
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
                        if consecutive_failures >= 10:
                            logger.warning(
                                "[LATENCY][PERSISTENT_STREAM_RECONNECT] "
                                "url=%s failures=%d — reconnecting",
                                self.url,
                                consecutive_failures,
                            )
                            self._connected = False
                            break

            except Exception as exc:  # noqa: BLE001
                logger.warning("[LATENCY][PERSISTENT_STREAM_ERROR] url=%s error=%s", self.url, exc)
                self._connected = False
            finally:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                self._connected = False
                if not self._stop_event.is_set():
                    self._stop_event.wait(self._RECONNECT_DELAY_SEC)

        logger.info("PersistentFrameCache stopped | url=%s", self.url)


# ---------------------------------------------------------------------------
# RedisAlertPublisher
# ---------------------------------------------------------------------------

class RedisAlertPublisher:
    """
    Async publisher for Telegram alert events.

    Pipeline thread only enqueues violation metadata. Snapshot capture + Redis
    publish run in a daemon thread to avoid blocking GStreamer callbacks.

    Mitigation 2: sử dụng PersistentFrameCache thay vì mở cv2.VideoCapture
    mới cho mỗi event (loại bỏ ~4.5s bottleneck).
    """

    def __init__(
        self,
        settings: RootSettings,
        *,
        queue_size: int = 200,
        start_thread: bool = True,
    ) -> None:
        self.settings = settings
        self.snapshot_dir = ensure_dir(Path(settings.telegram.snapshot_dir))

        self._event_queue: queue.Queue[Optional[Dict[str, Any]]] = queue.Queue(maxsize=queue_size)
        self._cooldown_lock = threading.Lock()
        self._last_event_at: dict[str, float] = {}

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # ── Persistent frame cache (Fix 2) ──────────────────────────────────
        self._frame_cache: PersistentFrameCache | None = None
        if (
            settings.telegram.enabled
            and settings.telegram.snapshot_source == "rtmp"
        ):
            rtmp_url = settings.telegram.snapshot_rtmp_url.strip()
            if rtmp_url:
                self._frame_cache = PersistentFrameCache(
                    rtmp_url,
                    name="RTMPFrameCache",
                )
            else:
                logger.warning(
                    "snapshot_source=rtmp but snapshot_rtmp_url is empty — frame cache disabled"
                )

        if start_thread:
            self._thread = threading.Thread(
                target=self._publisher_loop,
                name="RedisAlertPublisher",
                daemon=True,
            )
            self._thread.start()

        logger.info(
            "RedisAlertPublisher initialized | topic=%s | snapshot_dir=%s | frame_cache=%s",
            self.settings.telegram.redis_topic,
            str(self.snapshot_dir),
            "enabled" if self._frame_cache is not None else "disabled",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enqueue_violation(
        self,
        event: Dict[str, Any],
        *,
        snapshot_frame: Any | None = None,
    ) -> None:
        """Enqueue one violation event with per-camera cooldown."""
        camera_id = str(event.get("camera_id") or "unknown")

        if self.settings.telegram.cooldown_sec > 0:
            now = time.monotonic()
            with self._cooldown_lock:
                last = self._last_event_at.get(camera_id, 0.0)
                if (now - last) < self.settings.telegram.cooldown_sec:
                    logger.debug(
                        "Skip alert due to cooldown | camera=%s | cooldown=%.2fs",
                        camera_id,
                        self.settings.telegram.cooldown_sec,
                    )
                    return
                self._last_event_at[camera_id] = now

        normalized = self._normalize_event(event)
        normalized["ts_enqueue"] = time.time()

        queue_item: Dict[str, Any] = {
            "event": normalized,
            "snapshot_frame": snapshot_frame,
        }
        try:
            self._event_queue.put_nowait(queue_item)
            logger.info(
                f"[LATENCY][QUEUE_PUT] "
                f"event_id={normalized.get('event_id')} "
                f"queue_size={self._event_queue.qsize()} "
                f"detect_to_enqueue_ms={(normalized['ts_enqueue'] - normalized.get('ts_detect', normalized['ts_enqueue'])) * 1000:.2f}"
            )
        except queue.Full:
            logger.warning(
                f"[LATENCY][QUEUE_FULL] "
                f"event_id={normalized.get('event_id')} "
                f"queue_size={self._event_queue.qsize()}"
            )

    def stop(self) -> None:
        """Stop background publisher thread and frame cache."""
        self._stop_event.set()
        if self._frame_cache is not None:
            self._frame_cache.stop()
        try:
            self._event_queue.put_nowait(None)
        except queue.Full:
            pass

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        bbox = event.get("bbox") or []
        if not isinstance(bbox, list):
            bbox = []

        res = {
            "event_type": str(event.get("event_type") or "helmet_violation"),
            "event_id": str(event.get("event_id") or uuid.uuid4()),
            "camera_id": str(event.get("camera_id") or "unknown"),
            "timestamp": float(event.get("timestamp") or time.time()),
            "confidence": float(event.get("confidence") or 0.0),
            "frame_num": int(event.get("frame_num") or 0),
            "bbox": [float(x) for x in bbox[:4]],
            "snapshot_path": "",
        }
        for k, v in event.items():
            if k not in res:
                res[k] = v
        return res

    def _build_snapshot_path(self, camera_id: str, event_id: str) -> Path:
        return self.snapshot_dir / f"violation_{camera_id}_{event_id}.jpg"

    def _crop_tiled_frame(
        self,
        image: Any,
        source_id: int,
    ) -> tuple[Any, tuple[int, int]]:
        """
        Crop tiled frame (NxM grid) ra quadrant của camera source_id.

        Layout tiler: source 0 → (row=0,col=0), source 1 → (row=0,col=1), ...
        Giống cách nvmultistreamtiler sắp xếp: source_id = row * cols + col.

        Returns:
            (cropped_image, (tile_x0, tile_y0)) — offset của quadrant trong tiled frame.
        """
        import math  # noqa: PLC0415

        n_cameras = len(self.settings.cameras)
        tiler_cfg = self.settings.tiler

        # Tính grid đúng theo cùng logic với tiler.py
        rows_cfg = tiler_cfg.rows
        cols_cfg = tiler_cfg.cols
        if rows_cfg > 0 and cols_cfg > 0:
            rows, cols = rows_cfg, cols_cfg
        elif rows_cfg > 0:
            cols = math.ceil(n_cameras / rows_cfg)
            rows = rows_cfg
        elif cols_cfg > 0:
            rows = math.ceil(n_cameras / cols_cfg)
            cols = cols_cfg
        else:
            cols = math.ceil(math.sqrt(n_cameras))
            rows = math.ceil(n_cameras / cols)

        tiled_h, tiled_w = image.shape[:2]
        tile_w = tiled_w // cols
        tile_h = tiled_h // rows

        # source_id → (row_idx, col_idx)
        row_idx = source_id // cols
        col_idx = source_id % cols

        x0 = col_idx * tile_w
        y0 = row_idx * tile_h
        x1 = min(x0 + tile_w, tiled_w)
        y1 = min(y0 + tile_h, tiled_h)

        cropped = image[y0:y1, x0:x1]
        logger.info(
            "[TILER_CROP] source_id=%d grid=%dx%d tile=(%d,%d,%d,%d) "
            "tiled_frame=%dx%d crop=%dx%d",
            source_id, rows, cols, x0, y0, x1, y1,
            tiled_w, tiled_h, cropped.shape[1], cropped.shape[0],
        )
        return cropped, (x0, y0)

    def _save_snapshot_from_probe_frame(
        self,
        *,
        camera_id: str,
        event_id: str,
        frame: Any,
        bbox: Any | None,
        event: Dict[str, Any] | None = None,
    ) -> str | None:
        """Save snapshot directly from probe frame (preferred source — Fix 3)."""
        try:
            import cv2  # noqa: PLC0415
            import numpy as np  # noqa: PLC0415
        except ImportError:
            logger.warning("cv2/numpy not installed — skip probe snapshot")
            return None

        if frame is None:
            return None

        try:
            image = np.array(frame, copy=True)
            if image.ndim != 3:
                return None

            # pyds surface is RGBA after Fix 3.
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                return None

            # ── Crop per-camera quadrant khi tiler enabled ────────────────
            if (
                event is not None
                and self.settings.tiler is not None
                and self.settings.tiler.enabled
                and len(self.settings.cameras) > 1
            ):
                source_id = int(event.get("source_id", -1))
                if source_id >= 0:
                    image, _ = self._crop_tiled_frame(
                        image, source_id
                    )

            # ── Draw bbox ─────────────────────────────────────────────────
            # bbox trong event là tọa độ streammux space (streammux_width × streammux_height).
            # Sau khi tiler scale, mỗi tile có kích thước tile_w × tile_h.
            # → Cần scale bbox: bbox_tile = bbox_streammux * (tile_w/smux_w, tile_h/smux_h)
            # → Toạ độ trong ảnh đã crop = bbox sau scale (tile_offset đã bị loại bỏ khi crop)
            t_before_bbox = time.time()
            has_bbox = isinstance(bbox, list) and len(bbox) == 4
            if has_bbox:
                left_raw, top_raw, width_raw, height_raw = [float(v) for v in bbox]

                # Scale factors: streammux → tile trong tiled frame
                smux_w = float(self.settings.pipeline.streammux_width)
                smux_h = float(self.settings.pipeline.streammux_height)
                crop_h, crop_w = image.shape[:2]
                # crop_w/crop_h là kích thước tile (sau khi _crop_tiled_frame đã cắt)
                scale_x = crop_w / smux_w
                scale_y = crop_h / smux_h

                x1 = max(0, int(left_raw * scale_x))
                y1 = max(0, int(top_raw * scale_y))
                x2 = min(crop_w - 1, int((left_raw + width_raw) * scale_x))
                y2 = min(crop_h - 1, int((top_raw + height_raw) * scale_y))

                logger.info(
                    "[LATENCY][BBOX_META] event_id=%s "
                    "bbox_raw=[%.1f,%.1f,%.1f,%.1f] "
                    "smux=%dx%d tile=%dx%d scale=(%.3f,%.3f) "
                    "bbox_scaled=[%d,%d,%d,%d]",
                    event_id,
                    left_raw, top_raw, width_raw, height_raw,
                    int(smux_w), int(smux_h), crop_w, crop_h,
                    scale_x, scale_y,
                    x1, y1, x2, y2,
                )

                if x2 > x1 and y2 > y1:
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(
                        image,
                        "no_helmet",
                        (x1, max(12, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

            t_after_bbox = time.time()
            logger.info(
                "[LATENCY][DRAW_BBOX] event_id=%s has_bbox=%s draw_ms=%.2f%s",
                event_id,
                has_bbox,
                (t_after_bbox - t_before_bbox) * 1000,
                "" if has_bbox else " reason=missing_bbox",
            )

            snapshot_path = self._build_snapshot_path(camera_id, event_id)
            t_before_imwrite = time.time()
            if event is not None:
                event["ts_before_imwrite"] = t_before_imwrite
            write_ok = cv2.imwrite(
                str(snapshot_path),
                image,
                [cv2.IMWRITE_JPEG_QUALITY, 85],
            )
            t_after_imwrite = time.time()
            if event is not None:
                event["ts_after_imwrite"] = t_after_imwrite

            logger.info(
                f"[LATENCY][IMWRITE] "
                f"event_id={event_id} "
                f"ok={write_ok} "
                f"imwrite_ms={(t_after_imwrite - t_before_imwrite) * 1000:.2f} "
                f"snapshot_path={snapshot_path} "
                f"image_shape={image.shape} "
                f"jpeg_quality=85"
            )

            if not write_ok:
                return None

            logger.info(
                "Snapshot captured from probe | camera=%s | event_id=%s | path=%s",
                camera_id,
                event_id,
                str(snapshot_path),
            )
            return str(snapshot_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Probe snapshot save failed: %s", exc)
            return None

    def _grab_snapshot_from_cache(self, camera_id: str, event_id: str) -> str | None:
        """
        Lấy frame từ PersistentFrameCache (Fix 2).
        Thay thế _grab_snapshot_from_stream — không mở VideoCapture mới.
        """
        try:
            import cv2  # noqa: PLC0415
        except ImportError:
            logger.warning("cv2 not installed — skip cache snapshot")
            return None

        if self._frame_cache is None:
            logger.warning(
                "[LATENCY][FRAME_CACHE_MISS] "
                "event_id=%s reason=cache_not_initialized",
                event_id,
            )
            return None

        t_before = time.time()
        frame, age_ms = self._frame_cache.get_frame()
        t_after = time.time()

        if frame is None:
            logger.warning(
                f"[LATENCY][FRAME_CACHE_MISS] "
                f"event_id={event_id} "
                f"reason=no_frame_yet "
                f"cache_connected={self._frame_cache.is_connected}"
            )
            return None

        logger.info(
            f"[LATENCY][FRAME_CACHE_HIT] "
            f"event_id={event_id} "
            f"get_frame_ms={(t_after - t_before) * 1000:.2f}"
        )
        logger.info(
            f"[LATENCY][FRAME_CACHE_AGE_MS] "
            f"event_id={event_id} "
            f"age_ms={age_ms:.2f}"
        )

        snapshot_path = self._build_snapshot_path(camera_id, event_id)
        t_before_imwrite = time.time()
        write_ok = cv2.imwrite(
            str(snapshot_path),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 85],
        )
        t_after_imwrite = time.time()

        logger.info(
            f"[LATENCY][IMWRITE] "
            f"event_id={event_id} "
            f"ok={write_ok} "
            f"imwrite_ms={(t_after_imwrite - t_before_imwrite) * 1000:.2f} "
            f"snapshot_path={snapshot_path} "
            f"image_shape={frame.shape} "
            f"jpeg_quality=85"
        )

        if not write_ok:
            logger.warning(
                "[LATENCY][FRAME_CACHE_IMWRITE_FAIL] event_id=%s", event_id
            )
            return None

        logger.info(
            "Snapshot captured from cache | camera=%s | event_id=%s | path=%s",
            camera_id,
            event_id,
            str(snapshot_path),
        )
        return str(snapshot_path)

    def _resolve_snapshot(
        self,
        *,
        event: Dict[str, Any],
        snapshot_frame: Any | None,
    ) -> str | None:
        event_id = str(event.get("event_id") or uuid.uuid4())
        camera_id = str(event.get("camera_id") or "unknown")
        snapshot_source = self.settings.telegram.snapshot_source

        if snapshot_source == "probe":
            snapshot_path = self._save_snapshot_from_probe_frame(
                camera_id=camera_id,
                event_id=event_id,
                frame=snapshot_frame,
                bbox=event.get("bbox"),
                event=event,
            )
            if snapshot_path:
                return snapshot_path

            logger.warning(
                "Probe snapshot unavailable; falling back to cache | camera=%s | event_id=%s",
                camera_id,
                event_id,
            )
            # Fallback to cache when probe fails
            return self._grab_snapshot_from_cache(camera_id, event_id)

        # snapshot_source == "rtmp" → use persistent cache
        return self._grab_snapshot_from_cache(camera_id, event_id)

    def _connect_redis(self):
        """Connect to Redis with retry until stop signal."""
        try:
            import redis  # noqa: PLC0415
        except ImportError:
            logger.error("Redis python client not installed — alert publish disabled")
            return None

        host = self.settings.telegram.redis_host
        port = self.settings.telegram.redis_port

        while not self._stop_event.is_set():
            try:
                client = redis.Redis(
                    host=host,
                    port=port,
                    socket_timeout=3,
                    socket_connect_timeout=3,
                    decode_responses=True,
                )
                client.ping()
                logger.info("Connected to Redis | host=%s | port=%d", host, port)
                return client
            except Exception as exc:  # noqa: BLE001
                logger.warning("Redis connect failed (%s) — retry in 5s", exc)
                self._stop_event.wait(5.0)

        return None

    def _publish_payload(self, client: Any, event: Dict[str, Any]) -> Any | None:
        """Publish payload and reconnect on failure. Returns active client."""
        topic = self.settings.telegram.redis_topic
        event["ts_after_redis_publish"] = time.time()
        payload = json.dumps(event, ensure_ascii=False)

        t_before_redis = time.time()
        try:
            client.publish(topic, payload)
            t_after_redis = time.time()
            logger.info(
                f"[LATENCY][REDIS_PUBLISH] "
                f"event_id={event.get('event_id')} "
                f"redis_publish_ms={(t_after_redis - t_before_redis) * 1000:.2f} "
                f"detect_to_redis_ms={(t_after_redis - event.get('ts_detect', t_after_redis)) * 1000:.2f}"
            )
            return client
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis publish failed (%s) — reconnecting", exc)

        new_client = self._connect_redis()
        if new_client is None:
            return None

        event["ts_after_redis_publish"] = time.time()
        payload = json.dumps(event, ensure_ascii=False)
        t_before_redis = time.time()
        try:
            new_client.publish(topic, payload)
            t_after_redis = time.time()
            logger.info(
                f"[LATENCY][REDIS_PUBLISH] (retry) "
                f"event_id={event.get('event_id')} "
                f"redis_publish_ms={(t_after_redis - t_before_redis) * 1000:.2f} "
                f"detect_to_redis_ms={(t_after_redis - event.get('ts_detect', t_after_redis)) * 1000:.2f}"
            )
            return new_client
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis publish retry failed: %s", exc)
            return None

    def _process_event(
        self,
        client: Any,
        event: Dict[str, Any],
        *,
        snapshot_frame: Any | None = None,
    ) -> Any | None:
        event_id = str(event.get("event_id") or uuid.uuid4())
        camera_id = str(event.get("camera_id") or "unknown")
        snapshot_path = self._resolve_snapshot(
            event=event,
            snapshot_frame=snapshot_frame,
        )
        event["snapshot_path"] = snapshot_path or ""

        next_client = self._publish_payload(client, event)
        if next_client is not None:
            logger.info(
                "Alert published | camera=%s | event_id=%s | snapshot=%s",
                camera_id,
                event_id,
                "YES" if snapshot_path else "NO",
            )
        return next_client

    def _publisher_loop(self) -> None:
        client = self._connect_redis()
        if client is None:
            logger.info("RedisAlertPublisher thread exited (no redis connection)")
            return

        while not self._stop_event.is_set():
            try:
                event = self._event_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if event is None:
                break

            if client is None:
                client = self._connect_redis()
                if client is None:
                    if self._stop_event.is_set():
                        break
                    continue

            if isinstance(event, dict) and "event" in event:
                payload = event.get("event")
                snapshot_frame = event.get("snapshot_frame")
            else:
                payload = event
                snapshot_frame = None

            if not isinstance(payload, dict):
                continue

            payload["ts_publisher_start"] = time.time()
            logger.info(
                f"[LATENCY][PUBLISHER_START] "
                f"event_id={payload.get('event_id')} "
                f"queue_wait_ms={(payload['ts_publisher_start'] - payload.get('ts_enqueue', payload['ts_publisher_start'])) * 1000:.2f} "
                f"queue_size={self._event_queue.qsize()}"
            )

            client = self._process_event(
                client,
                payload,
                snapshot_frame=snapshot_frame,
            )

        logger.info("RedisAlertPublisher thread stopped")
