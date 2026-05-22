from __future__ import annotations

import json
import os
import queue
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

import pyds

from apps.vision_service.src.logger import get_logger
from apps.vision_service.src.settings import RootSettings
from apps.vision_service.src.utils.file_utils import ensure_dir

logger = get_logger(__name__)

# Minimum seconds between events per camera (avoid event flood)
_EVENT_COOLDOWN_SEC = 5.0

# RTMP output stream URL — AI-processed video WITH bounding boxes
# MediaMTX receives this and re-serves it as HLS/RTMP for playback
_RTMP_OUTPUT_URL = "rtmp://127.0.0.1:1935/vision1"
# Fallback: HLS output từ MediaMTX
_HLS_OUTPUT_URL = "http://127.0.0.1:8888/vision1/index.m3u8?cookieCheck=1"


class MsgBrokerProbe:
    """
    Probe đặt SAU phần tử nvdsosd.

    Nhiệm vụ:
      - Quét obj_meta_list tìm vi phạm (class_id == 1 = no_helmet).
      - Đẩy event metadata vào một internal queue (non-blocking).
      - Một daemon thread riêng biệt:
          1. Grab snapshot từ RTMP output stream (có bounding box).
          2. Lưu snapshot vào storage/snapshots/.
          3. Publish JSON (kèm snapshot_path) lên Redis Pub/Sub.

    Thiết kế non-blocking:
      Probe callback KHÔNG bao giờ gọi Redis hoặc I/O trực tiếp — tránh làm
      chậm GStreamer pipeline thread. Tất cả I/O thực hiện trong daemon thread.
    """

    def __init__(self, settings: RootSettings) -> None:
        self.settings = settings
        self.snapshot_dir = Path("/workspace/storage/snapshots")
        ensure_dir(self.snapshot_dir)

        # Internal event queue (probe → redis publisher thread)
        self._event_queue: queue.Queue = queue.Queue(maxsize=100)

        # Cooldown tracker: camera_id -> last_event_time
        self._last_event: dict[str, float] = {}

        # Redis publisher daemon thread
        self._stop_event = threading.Event()
        self._publisher_thread = threading.Thread(
            target=self._redis_publisher_loop,
            name="MsgBrokerProbe-redis",
            daemon=True,
        )
        self._publisher_thread.start()

        logger.info("MsgBrokerProbe initialized.")

    def attach_to(self, pad: Gst.Pad) -> None:
        """Gắn callback vào pad."""
        pad.add_probe(Gst.PadProbeType.BUFFER, self._on_buffer_probe, None)
        logger.info(
            "MsgBrokerProbe attached to %s:%s",
            pad.get_parent_element().get_name(),
            pad.get_name(),
        )

    def stop(self) -> None:
        """Dừng publisher thread (gọi khi pipeline stop)."""
        self._stop_event.set()
        try:
            self._event_queue.put_nowait(None)  # unblock queue.get()
        except queue.Full:
            pass

    # ------------------------------------------------------------------
    # GStreamer probe callback (runs in GStreamer pipeline thread)
    # ------------------------------------------------------------------

    def _resolve_camera_id(self, frame_meta: Any) -> str:
        source_id = int(frame_meta.source_id)
        if 0 <= source_id < len(self.settings.cameras):
            return self.settings.cameras[source_id].camera_id
        return f"source_{source_id}"

    def _on_buffer_probe(
        self,
        _pad: Gst.Pad,
        info: Gst.PadProbeInfo,
        _user_data: object,
    ) -> Gst.PadProbeReturn:
        gst_buffer = info.get_buffer()
        if gst_buffer is None:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if batch_meta is None:
            return Gst.PadProbeReturn.OK

        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            camera_id = self._resolve_camera_id(frame_meta)

            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                # class_id == 1 → "no_helmet"
                if int(obj_meta.class_id) == 1:
                    now = time.monotonic()
                    last = self._last_event.get(camera_id, 0.0)

                    if (now - last) >= _EVENT_COOLDOWN_SEC:
                        self._last_event[camera_id] = now
                        event_id = str(uuid.uuid4())[:8]
                        confidence = float(obj_meta.confidence) if obj_meta.confidence else 0.0

                        # Tạo event payload
                        event = {
                            "event_type": "helmet_violation",
                            "camera_id": camera_id,
                            "event_id": event_id,
                            "timestamp": time.time(),
                            "confidence": round(confidence, 3),
                            "frame_num": int(frame_meta.frame_num),
                            "snapshot_path": "",  # Sẽ được điền bởi daemon thread
                        }

                        # Push vào queue (non-blocking — drop nếu queue đầy)
                        try:
                            self._event_queue.put_nowait(event)
                        except queue.Full:
                            logger.warning("[MsgBrokerProbe] Event queue full — dropping event")

                    break  # Chỉ cần 1 vi phạm/frame

                l_obj = l_obj.next

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    # ------------------------------------------------------------------
    # Snapshot capture (runs in daemon thread — safe to do I/O here)
    # ------------------------------------------------------------------

    def _grab_snapshot(self, camera_id: str, event_id: str) -> Optional[str]:
        """
        Grab một frame từ RTMP output stream (AI-processed, có bounding box).
        Lưu vào /workspace/storage/snapshots/violation_{camera_id}_{event_id}.jpg.
        Trả về path nếu thành công, None nếu thất bại.
        """
        try:
            import cv2  # noqa: PLC0415
        except ImportError:
            logger.warning("[MsgBrokerProbe] cv2 not available — skipping snapshot")
            return None

        snapshot_path = str(
            self.snapshot_dir / f"violation_{camera_id}_{event_id}.jpg"
        )

        # Thử từ RTMP output (AI-processed với bounding boxes)
        for attempt, url in enumerate([_RTMP_OUTPUT_URL, _HLS_OUTPUT_URL], start=1):
            cap = None
            try:
                cap = cv2.VideoCapture(url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)

                # Đọc vài frame để bỏ qua buffered frames cũ
                for _ in range(3):
                    cap.grab()

                ret, frame = cap.read()
                if ret and frame is not None:
                    cv2.imwrite(
                        snapshot_path,
                        frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 85],
                    )
                    logger.info(
                        "[MsgBrokerProbe] Snapshot saved | path=%s | url=%s",
                        snapshot_path, url,
                    )
                    return snapshot_path
                else:
                    logger.debug("[MsgBrokerProbe] No frame from %s (attempt %d)", url, attempt)
            except Exception as exc:
                logger.debug("[MsgBrokerProbe] Snapshot attempt %d failed: %s", attempt, exc)
            finally:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass

        logger.warning(
            "[MsgBrokerProbe] All snapshot attempts failed for %s/%s — will send text-only alert",
            camera_id, event_id,
        )
        return None

    # ------------------------------------------------------------------
    # Redis publisher daemon thread
    # ------------------------------------------------------------------

    def _connect_redis(self):
        """Tạo Redis connection với retry."""
        try:
            import redis  # noqa: PLC0415
        except ImportError:
            logger.error(
                "[MsgBrokerProbe] 'redis' package not installed. "
                "Run `make build` to rebuild the image. Redis publish DISABLED."
            )
            return None, self.settings.telegram.redis_topic

        host = self.settings.telegram.redis_host
        port = self.settings.telegram.redis_port
        topic = self.settings.telegram.redis_topic

        while not self._stop_event.is_set():
            try:
                r = redis.Redis(host=host, port=port, socket_timeout=3)
                r.ping()
                logger.info(
                    "[MsgBrokerProbe] Connected to Redis at %s:%d | topic=%s",
                    host, port, topic,
                )
                return r, topic
            except Exception as exc:
                logger.warning(
                    "[MsgBrokerProbe] Redis connection failed (%s) — retrying in 5s...", exc
                )
                self._stop_event.wait(5)

        return None, self.settings.telegram.redis_topic

    def _redis_publisher_loop(self) -> None:
        """Daemon thread: grab snapshot → publish to Redis."""
        r, topic = self._connect_redis()
        if r is None:
            logger.info("[MsgBrokerProbe] Publisher thread exiting (no Redis)")
            return

        while not self._stop_event.is_set():
            try:
                event = self._event_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if event is None:  # poison pill
                break

            # 1. Grab snapshot (in daemon thread — safe)
            snapshot_path = self._grab_snapshot(
                event["camera_id"], event["event_id"]
            )
            event["snapshot_path"] = snapshot_path or ""

            # 2. Publish to Redis
            try:
                payload = json.dumps(event, ensure_ascii=False)
                r.publish(topic, payload)
                logger.info(
                    "[MsgBrokerProbe] Published violation | camera=%s | event_id=%s | "
                    "confidence=%.2f | snapshot=%s",
                    event.get("camera_id"),
                    event.get("event_id"),
                    event.get("confidence", 0),
                    "YES" if snapshot_path else "NO (text-only)",
                )
            except Exception as exc:
                logger.warning("[MsgBrokerProbe] Redis publish failed: %s", exc)
                # Thử kết nối lại
                r, topic = self._connect_redis()
                if r is None:
                    break

        logger.info("[MsgBrokerProbe] Publisher thread stopped.")
