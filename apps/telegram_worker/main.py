import datetime
import json
import logging
import os
import re
import time

import httpx
import redis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("telegram_worker")

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_CHANNEL = os.getenv("REDIS_CHANNEL", "helmet_violations")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def send_text_to_telegram(text: str, bot_token: str, chat_id: str, event: dict | None = None) -> None:
    """Gửi tin nhắn text khi không có ảnh."""
    api_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    if event is not None:
        logger.info(f"[LATENCY][SENDING_TYPE] event_id={event.get('event_id')} type=sendMessage")
    t_before_tg = time.time()
    if event is not None:
        event["ts_before_telegram_send"] = t_before_tg
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                api_url,
                data={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            )
            t_after_tg = time.time()
            if event is not None:
                event["ts_after_telegram_send"] = t_after_tg
            response.raise_for_status()
            logger.info("Successfully sent text alert to Telegram")
            if event is not None:
                logger.info(
                    f"[LATENCY][TELEGRAM_SEND_MESSAGE_DONE] "
                    f"event_id={event.get('event_id')} "
                    f"telegram_ms={(t_after_tg - t_before_tg) * 1000:.2f} "
                    f"status_code={response.status_code} "
                    f"detect_to_telegram_ms={(t_after_tg - event.get('ts_detect', t_after_tg)) * 1000:.2f}"
                )
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to send text alert to Telegram: %s", exc)


def send_photo_to_telegram(snapshot_path: str, caption: str, bot_token: str, chat_id: str, event: dict | None = None) -> None:
    """Gửi ảnh nếu file tồn tại, fallback về text nếu không có."""
    if not os.path.exists(snapshot_path):
        if event is not None:
            logger.warning(
                f"[LATENCY][SNAPSHOT_MISSING] "
                f"event_id={event.get('event_id')} "
                f"snapshot_path={snapshot_path} "
                f"fallback=sendMessage"
            )
        else:
            logger.warning("Snapshot not found: %s — sending text alert instead", snapshot_path)
        send_text_to_telegram(caption, bot_token, chat_id, event=event)
        return

    api_url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    if event is not None:
        file_size_kb = os.path.getsize(snapshot_path) / 1024
        logger.info(
            f"[LATENCY][TELEGRAM_SEND_PHOTO_START] "
            f"event_id={event.get('event_id')} "
            f"snapshot_path={snapshot_path} "
            f"file_size_kb={file_size_kb:.2f}"
        )
        logger.info(f"[LATENCY][SENDING_TYPE] event_id={event.get('event_id')} type=sendPhoto")

    t_before_tg = time.time()
    if event is not None:
        event["ts_before_telegram_send"] = t_before_tg
    try:
        with open(snapshot_path, "rb") as photo_file:
            files = {"photo": photo_file}
            data = {"chat_id": chat_id, "caption": caption, "parse_mode": "HTML"}

            with httpx.Client(timeout=10.0) as client:
                response = client.post(api_url, data=data, files=files)
                t_after_tg = time.time()
                if event is not None:
                    event["ts_after_telegram_send"] = t_after_tg
                response.raise_for_status()
                logger.info("Successfully sent snapshot to Telegram: %s", snapshot_path)
                if event is not None:
                    logger.info(
                        f"[LATENCY][TELEGRAM_SEND_PHOTO_DONE] "
                        f"event_id={event.get('event_id')} "
                        f"telegram_ms={(t_after_tg - t_before_tg) * 1000:.2f} "
                        f"status_code={response.status_code} "
                        f"detect_to_telegram_ms={(t_after_tg - event.get('ts_detect', t_after_tg)) * 1000:.2f}"
                    )
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to send photo to Telegram: %s", exc)
        send_text_to_telegram(caption, bot_token, chat_id, event=event)


def _format_timestamp(raw_ts: object) -> str:
    if isinstance(raw_ts, (int, float)):
        return datetime.datetime.fromtimestamp(float(raw_ts)).strftime("%H:%M:%S %d/%m/%Y")

    if isinstance(raw_ts, str) and raw_ts.strip():
        value = raw_ts.strip()
        try:
            if value.endswith("Z"):
                value = value[:-1] + "+00:00"
            dt = datetime.datetime.fromisoformat(value)
            return dt.strftime("%H:%M:%S %d/%m/%Y")
        except ValueError:
            pass

    return datetime.datetime.now().strftime("%H:%M:%S %d/%m/%Y")


def _build_caption(payload: dict) -> str:
    """Tạo caption cho thông báo Telegram từ payload Redis."""
    caption = "🚨 <b>CẢNH BÁO VI PHẠM KHÔNG ĐỘI MŨ BẢO HIỂM</b> 🚨"

    camera_id = payload.get("camera_id")
    event_id = payload.get("event_id")

    # Backward compatibility với payload kiểu nvmsgconv cũ
    if not camera_id:
        sensor_str = None
        if "sensor" in payload and "id" in payload.get("sensor", {}):
            sensor_str = payload["sensor"]["id"]
        elif "sensorStr" in payload:
            sensor_str = payload["sensorStr"]

        if sensor_str and "camera=" in sensor_str:
            parts = dict(p.split("=") for p in sensor_str.split("|") if "=" in p)
            camera_id = parts.get("camera")
            event_id = event_id or parts.get("event_id")
        elif sensor_str:
            camera_id = sensor_str

    if camera_id:
        caption += f"\n📷 Camera: <b>{camera_id}</b>"
    if event_id:
        caption += f"\n🆔 Event ID: <code>{event_id}</code>"

    ts = _format_timestamp(payload.get("timestamp"))
    caption += f"\n⏰ Thời gian: {ts}"

    confidence = payload.get("confidence")
    if confidence is None and isinstance(payload.get("object"), dict):
        confidence = payload["object"].get("confidence")

    if confidence is not None:
        try:
            conf = float(confidence)
            if conf > 1.0:
                conf = conf / 100.0
            caption += f"\n🎯 Độ tin cậy: {conf:.1%}"
        except Exception:  # noqa: BLE001
            pass

    return caption


def _extract_snapshot_path(payload: dict) -> str | None:
    """Tìm snapshot_path trong payload JSON, ưu tiên schema Redis v1."""
    direct = payload.get("snapshot_path")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    checks = [
        payload.get("sensorStr") if "/workspace/storage/snapshots/" in str(payload.get("sensorStr", "")) else None,
        payload.get("sensor", {}).get("id") if "/workspace/storage/snapshots/" in str(payload.get("sensor", {}).get("id", "")) else None,
    ]
    for item in checks:
        if item:
            return item

    payload_dump = json.dumps(payload, ensure_ascii=False)
    match = re.search(r'(/workspace/storage/snapshots/[^"\'\\]+)', payload_dump)
    if match:
        return match.group(1)

    return None


def _connect_redis() -> redis.Redis:
    while True:
        try:
            client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
            client.ping()
            return client
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis connection failed (%s) — retry in 5s", exc)
            time.sleep(5)


def main() -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing. Worker will not start properly.")
        raise SystemExit(1)

    logger.info("Starting Telegram Worker, connecting to Redis at %s:%s", REDIS_HOST, REDIS_PORT)
    client = _connect_redis()

    pubsub = client.pubsub()
    pubsub.subscribe(REDIS_CHANNEL)
    logger.info("Subscribed to Redis channel: %s", REDIS_CHANNEL)

    for message in pubsub.listen():
        if message["type"] != "message":
            continue

        try:
            payload_str = message["data"]
            payload = json.loads(payload_str)
            
            payload["ts_worker_received"] = time.time()
            logger.info(
                f"[LATENCY][WORKER_RECEIVED] "
                f"event_id={payload.get('event_id')} "
                f"redis_to_worker_ms={(payload['ts_worker_received'] - payload.get('ts_after_redis_publish', payload['ts_worker_received'])) * 1000:.2f} "
                f"detect_to_worker_ms={(payload['ts_worker_received'] - payload.get('ts_detect', payload['ts_worker_received'])) * 1000:.2f}"
            )

            caption = _build_caption(payload)
            snapshot_path = _extract_snapshot_path(payload)

            DEBUG_ALERT_TEXT_FIRST = os.getenv("DEBUG_ALERT_TEXT_FIRST", "false").lower() == "true"
            if DEBUG_ALERT_TEXT_FIRST:
                logger.info("DEBUG_ALERT_TEXT_FIRST is enabled; sending text alert first.")
                send_text_to_telegram(caption, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, event=payload)
                if snapshot_path:
                    send_photo_to_telegram(snapshot_path, caption, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, event=payload)
            else:
                if snapshot_path:
                    send_photo_to_telegram(snapshot_path, caption, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, event=payload)
                else:
                    send_text_to_telegram(caption, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, event=payload)

        except json.JSONDecodeError:
            logger.error("Failed to decode JSON payload: %s", str(message.get("data"))[:100])
        except Exception as exc:  # noqa: BLE001
            logger.error("Error processing message: %s", exc, exc_info=True)


if __name__ == "__main__":
    main()
