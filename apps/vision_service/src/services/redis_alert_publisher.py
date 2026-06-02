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
            "class_id": int(event.get("class_id") if event.get("class_id") is not None else 1),
            "class_name": str(event.get("class_name") or "no_helmet"),
            "bbox": [float(x) for x in bbox[:4]],
            "snapshot_path": "",
        }
        for k, v in event.items():
            if k not in res:
                res[k] = v
        return res

    def _build_snapshot_path(self, camera_id: str, event_id: str) -> Path:
        return self.snapshot_dir / f"violation_{camera_id}_{event_id}.jpg"

    def _save_snapshot_from_probe_frame(
        self,
        *,
        camera_id: str,
        event_id: str,
        frame: Any,
        bbox: Any | None,
        event: Dict[str, Any] | None = None,
    ) -> str | None:
        """Save snapshot from pre-tiler per-source probe frame.

        Snapshot branch (snapshot-capsfilter) provides RGBA frame in streammux space
        (960×544 per-source), before the tiler. No tiled-frame cropping needed.
        Bbox coordinates are in the same streammux space → scale by (snapshot/source) ratio.
        """
        try:
            import cv2  # noqa: PLC0415
            import numpy as np  # noqa: PLC0415
        except ImportError:
            logger.warning("cv2/numpy not installed — skip probe snapshot")
            return None

        if frame is None:
            logger.warning("[SNAPSHOT_WARNING] snapshot_frame is None")
            return None

        try:
            image = np.array(frame, copy=True)
            if image.ndim != 3:
                return None

            # RGBA → BGR for cv2.imwrite
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                return None

            snap_h, snap_w = image.shape[:2]

            # ── Sanity check: cảnh báo nếu snapshot trông giống tiled frame ──────
            # Tiled frame thường có width >> streammux_width (vd: 1920 vs 960)
            smux_w = float(self.settings.pipeline.streammux_width)
            smux_h = float(self.settings.pipeline.streammux_height)
            if snap_w > smux_w * 1.3:
                logger.warning(
                    "[SNAPSHOT_WARNING] event_id=%s "
                    "reason='snapshot appears to be tiled output; alert snapshot should be before tiler' "
                    "snapshot_w=%d smux_w=%.0f",
                    event_id, snap_w, smux_w,
                )

            # ── Draw all bounding boxes if present ──────────────────────────────────
            t_before_bbox = time.time()
            all_objects = (event or {}).get("all_objects")
            has_bbox = False
            
            # Scale factor based on streammux resolution (coordinates are in streammux space)
            scale_x = snap_w / smux_w if smux_w > 0 else 1.0
            scale_y = snap_h / smux_h if smux_h > 0 else 1.0

            if all_objects is not None and isinstance(all_objects, list):
                logger.debug(
                    "[DRAW_BBOX_DEBUG] event_id=%s drawing all_objects count=%d | scaling=(%.3f, %.3f)",
                    event_id, len(all_objects), scale_x, scale_y
                )
                for obj in all_objects:
                    obj_bbox = obj.get("bbox")
                    if not (isinstance(obj_bbox, list) and len(obj_bbox) == 4):
                        continue

                    left_raw, top_raw, width_raw, height_raw = [float(v) for v in obj_bbox]
                    draw_x = int(left_raw * scale_x)
                    draw_y = int(top_raw * scale_y)
                    draw_w = int(width_raw * scale_x)
                    draw_h = int(height_raw * scale_y)

                    x1 = max(0, draw_x)
                    y1 = max(0, draw_y)
                    x2 = min(snap_w - 1, draw_x + draw_w)
                    y2 = min(snap_h - 1, draw_y + draw_h)

                    obj_class_id = obj.get("class_id")
                    obj_class_name = obj.get("class_name")
                    obj_conf = float(obj.get("confidence") or 0.0)

                    # Colors styling: helmet = green, no_helmet = red (BGR)
                    green = (0, 255, 0)
                    red = (0, 0, 255)
                    
                    if obj_class_name == "helmet" or obj_class_id == 0:
                        obj_color = green
                        obj_color_name = "green"
                    else:
                        obj_color = red
                        obj_color_name = "red"

                    logger.debug(
                        "[DRAW_BBOX_DEBUG] event_id=%s object: class_id=%s name=%s color=%s raw_bbox=%s",
                        event_id, obj_class_id, obj_class_name, obj_color_name, obj_bbox
                    )

                    if x2 > x1 and y2 > y1:
                        cv2.rectangle(image, (x1, y1), (x2, y2), obj_color, 3)
                        has_bbox = True
                        
                        label_text = obj_class_name or ("helmet" if obj_class_id == 0 else "no_helmet")
                        label_str = f"{label_text} {obj_conf:.1%}" if obj_conf > 0 else label_text
                        cv2.putText(
                            image,
                            label_str,
                            (x1, max(12, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            obj_color,
                            2,
                            cv2.LINE_AA,
                        )
            elif isinstance(bbox, list) and len(bbox) == 4:
                # Fallback to single triggering bbox (compatibility mode)
                left_raw, top_raw, width_raw, height_raw = [float(v) for v in bbox]
                draw_x = int(left_raw * scale_x)
                draw_y = int(top_raw * scale_y)
                draw_w = int(width_raw * scale_x)
                draw_h = int(height_raw * scale_y)

                x1 = max(0, draw_x)
                y1 = max(0, draw_y)
                x2 = min(snap_w - 1, draw_x + draw_w)
                y2 = min(snap_h - 1, draw_y + draw_h)

                green = (0, 255, 0)
                red = (0, 0, 255)
                color = red
                color_name = "red"

                class_name = (event or {}).get("class_name")
                class_id = (event or {}).get("class_id")

                if class_name == "helmet" or class_id == 0:
                    color = green
                    color_name = "green"
                else:
                    color = red
                    color_name = "red"

                logger.info(
                    f"[DRAW_BBOX_DEBUG] event_id={event_id} (fallback) "
                    f"class_id={class_id} class_name={class_name} "
                    f"color={color_name} bbox={bbox}"
                )

                if x2 > x1 and y2 > y1:
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
                    has_bbox = True
                    confidence = float((event or {}).get("confidence", 0))
                    label_text = class_name or ("helmet" if class_id == 0 else "no_helmet")
                    label = f"{label_text} {confidence:.1%}" if confidence > 0 else label_text
                    cv2.putText(
                        image,
                        label,
                        (x1, max(12, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                        cv2.LINE_AA,
                    )
            else:
                logger.info(
                    "[DRAW_BBOX_DEBUG] event_id=%s has_bbox=False reason=missing_bbox_and_all_objects "
                    "snapshot_width=%d snapshot_height=%d",
                    event_id, snap_w, snap_h,
                )

            t_after_bbox = time.time()
            logger.info(
                "[LATENCY][DRAW_BBOX] event_id=%s has_bbox=%s draw_ms=%.2f",
                event_id, has_bbox, (t_after_bbox - t_before_bbox) * 1000,
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
                "[LATENCY][IMWRITE] event_id=%s ok=%s imwrite_ms=%.2f "
                "snapshot_path=%s image_shape=%s jpeg_quality=85",
                event_id, write_ok,
                (t_after_imwrite - t_before_imwrite) * 1000,
                snapshot_path, image.shape,
            )

            if not write_ok:
                logger.warning("[SNAPSHOT_WARNING] imwrite failed")
                return None

            logger.info(
                "Snapshot captured from probe | camera=%s | event_id=%s | path=%s",
                camera_id, event_id, str(snapshot_path),
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
