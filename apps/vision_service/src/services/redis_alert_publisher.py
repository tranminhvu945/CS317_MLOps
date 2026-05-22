from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from apps.vision_service.src.logger import get_logger
from apps.vision_service.src.settings import RootSettings
from apps.vision_service.src.utils.file_utils import ensure_dir

logger = get_logger(__name__)


class RedisAlertPublisher:
    """
    Async publisher for Telegram alert events.

    Pipeline thread only enqueues violation metadata. Snapshot capture + Redis
    publish run in a daemon thread to avoid blocking GStreamer callbacks.
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

        if start_thread:
            self._thread = threading.Thread(
                target=self._publisher_loop,
                name="RedisAlertPublisher",
                daemon=True,
            )
            self._thread.start()

        logger.info(
            "RedisAlertPublisher initialized | topic=%s | snapshot_dir=%s",
            self.settings.telegram.redis_topic,
            str(self.snapshot_dir),
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

        import time as time_lib
        normalized = self._normalize_event(event)
        normalized["ts_enqueue"] = time_lib.time()

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
        """Stop background publisher thread."""
        self._stop_event.set()
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

    def _save_snapshot_from_probe_frame(
        self,
        *,
        camera_id: str,
        event_id: str,
        frame: Any,
        bbox: Any | None,
        event: Dict[str, Any] | None = None,
    ) -> str | None:
        """Save snapshot directly from probe frame (preferred source)."""
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

            # pyds surface is typically RGBA.
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                return None

            # Draw bbox for visual context if available.
            t_before_bbox = time_lib.time()
            if isinstance(bbox, list) and len(bbox) == 4:
                left, top, width, height = [int(float(v)) for v in bbox]
                x1 = max(0, left)
                y1 = max(0, top)
                x2 = min(image.shape[1] - 1, left + width)
                y2 = min(image.shape[0] - 1, top + height)
                if x2 > x1 and y2 > y1:
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(
                        image,
                        "no_helmet",
                        (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
            t_after_bbox = time_lib.time()
            logger.info(
                f"[LATENCY][DRAW_BBOX] "
                f"event_id={event_id} "
                f"draw_ms={(t_after_bbox - t_before_bbox) * 1000:.2f}"
            )

            snapshot_path = self._build_snapshot_path(camera_id, event_id)
            import time as time_lib
            t_before_imwrite = time_lib.time()
            if event is not None:
                event["ts_before_imwrite"] = t_before_imwrite
            write_ok = cv2.imwrite(
                str(snapshot_path),
                image,
                [cv2.IMWRITE_JPEG_QUALITY, 75],
            )
            t_after_imwrite = time_lib.time()
            if event is not None:
                event["ts_after_imwrite"] = t_after_imwrite

            logger.info(
                f"[LATENCY][IMWRITE] "
                f"event_id={event_id} "
                f"ok={write_ok} "
                f"imwrite_ms={(t_after_imwrite - t_before_imwrite) * 1000:.2f} "
                f"snapshot_path={snapshot_path} "
                f"image_shape={image.shape} "
                f"jpeg_quality=75"
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
            logger.debug("Probe snapshot save failed: %s", exc)
            return None

    def _grab_snapshot_from_stream(self, camera_id: str, event_id: str) -> str | None:
        """Capture snapshot from RTMP stream, fallback to HLS stream."""
        try:
            import cv2  # noqa: PLC0415
        except ImportError:
            logger.warning("cv2 not installed — skip snapshot capture")
            return None

        urls = [
            self.settings.telegram.snapshot_rtmp_url.strip(),
            self.settings.telegram.snapshot_hls_url.strip(),
        ]
        snapshot_path = self._build_snapshot_path(camera_id, event_id)

        for url in urls:
            if not url:
                continue

            cap = None
            t_start = time.time()
            try:
                logger.info(f"[LATENCY][STREAM_OPEN_START] event_id={event_id} url={url}")
                cap = cv2.VideoCapture(url)
                t_after_open = time.time()
                is_opened = hasattr(cap, "isOpened") and cap.isOpened()
                logger.info(
                    f"[LATENCY][STREAM_OPEN_DONE] "
                    f"event_id={event_id} "
                    f"url={url} "
                    f"is_opened={is_opened} "
                    f"open_ms={(t_after_open - t_start) * 1000:.2f}"
                )
                if not is_opened:
                    continue

                # Keep newest frame only to reduce stale snapshots.
                if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)

                t_before_grab = time.time()
                for _ in range(3):
                    cap.grab()
                
                ok, frame = cap.read()
                t_after_read = time.time()
                logger.info(
                    f"[LATENCY][STREAM_READ_DONE] "
                    f"event_id={event_id} "
                    f"ok={ok} "
                    f"read_ms={(t_after_read - t_before_grab) * 1000:.2f}"
                )
                if not ok or frame is None:
                    continue

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
                    continue

                logger.info(
                    "Snapshot captured | camera=%s | event_id=%s | path=%s",
                    camera_id,
                    event_id,
                    str(snapshot_path),
                )
                return str(snapshot_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"[LATENCY][STREAM_CAPTURE_FAILED] "
                    f"event_id={event_id} "
                    f"url={url} "
                    f"error={exc}"
                )
            finally:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass

        logger.warning(
            "Snapshot capture failed for event | camera=%s | event_id=%s",
            camera_id,
            event_id,
        )
        return None

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
        import time as time_lib
        event["ts_after_redis_publish"] = time_lib.time()
        payload = json.dumps(event, ensure_ascii=False)

        t_before_redis = time_lib.time()
        try:
            client.publish(topic, payload)
            t_after_redis = time_lib.time()
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

        event["ts_after_redis_publish"] = time_lib.time()
        payload = json.dumps(event, ensure_ascii=False)
        t_before_redis = time_lib.time()
        try:
            new_client.publish(topic, payload)
            t_after_redis = time_lib.time()
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
                "Probe snapshot unavailable; sending text-only alert | camera=%s | event_id=%s",
                camera_id,
                event_id,
            )
            return None

        return self._grab_snapshot_from_stream(camera_id, event_id)

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

            import time as time_lib
            payload["ts_publisher_start"] = time_lib.time()
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
