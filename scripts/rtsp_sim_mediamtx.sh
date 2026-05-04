#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-status}"
CONTAINER_NAME="${RTSP_SIM_CONTAINER_NAME:-uit_medseg_rtsp_sim}"
IMAGE="${RTSP_SIM_MEDIAMTX_IMAGE:-bluenviron/mediamtx:latest}"
RTSP_PORT="${RTSP_SIM_RTSP_PORT:-8554}"
RTMP_PORT="${RTSP_SIM_RTMP_PORT:-1935}"
API_PORT="${RTSP_SIM_API_PORT:-8888}"

is_running() {
  docker ps --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"
}

case "${ACTION}" in
  up)
    if is_running; then
      echo "[rtsp-sim] MediaMTX already running: ${CONTAINER_NAME}"
      exit 0
    fi

    if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
      docker rm -f "${CONTAINER_NAME}" >/dev/null
    fi

    docker run -d \
      --name "${CONTAINER_NAME}" \
      -p "${RTSP_PORT}:8554" \
      -p "${RTMP_PORT}:1935" \
      -p "${API_PORT}:8888" \
      "${IMAGE}" >/dev/null

    echo "[rtsp-sim] MediaMTX started: ${CONTAINER_NAME}"
    echo "  RTSP : rtsp://127.0.0.1:${RTSP_PORT}/<path>"
    echo "  RTMP : rtmp://127.0.0.1:${RTMP_PORT}/<path>"
    echo "  API  : http://127.0.0.1:${API_PORT}"
    ;;

  down)
    if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
      docker rm -f "${CONTAINER_NAME}" >/dev/null
      echo "[rtsp-sim] MediaMTX stopped: ${CONTAINER_NAME}"
    else
      echo "[rtsp-sim] MediaMTX not found: ${CONTAINER_NAME}"
    fi
    ;;

  status)
    if is_running; then
      echo "[rtsp-sim] MediaMTX is running: ${CONTAINER_NAME}"
      docker ps --filter "name=${CONTAINER_NAME}" --format '  {{.Names}} | {{.Status}} | {{.Ports}}'
    else
      echo "[rtsp-sim] MediaMTX is not running"
    fi
    ;;

  logs)
    docker logs -f "${CONTAINER_NAME}"
    ;;

  *)
    echo "Usage: $0 {up|down|status|logs}" >&2
    exit 1
    ;;
esac
