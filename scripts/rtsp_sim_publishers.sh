#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-status}"
CAMERA="${2:-}"
MODE="${3:-realtime}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="${RTSP_SIM_STATE_DIR:-${ROOT_DIR}/storage/rtsp_sim}"
LOG_DIR="${RTSP_SIM_LOG_DIR:-${ROOT_DIR}/storage/logs/rtsp_sim}"
mkdir -p "${STATE_DIR}" "${LOG_DIR}"

FFMPEG_BIN="${FFMPEG_BIN:-ffmpeg}"
RTSP_HOST="${RTSP_SIM_HOST:-127.0.0.1}"
RTSP_PORT="${RTSP_SIM_PORT:-8554}"
RTSP_TRANSPORT="${RTSP_SIM_TRANSPORT:-tcp}"
RTMP_HOST="${RTSP_SIM_RTMP_HOST:-127.0.0.1}"
RTMP_PORT="${RTSP_SIM_RTMP_PORT:-1935}"
HLS_HOST="${RTSP_SIM_HLS_HOST:-127.0.0.1}"
HLS_PORT="${RTSP_SIM_HLS_PORT:-8888}"
FFMPEG_DOCKER_IMAGE="${RTSP_SIM_FFMPEG_IMAGE:-jrottenberg/ffmpeg:8-scratch}"
FORCE_DOCKER_FFMPEG="${RTSP_SIM_FORCE_DOCKER_FFMPEG:-0}"
PUBLISHER_CONTAINER_PREFIX="${RTSP_SIM_PUBLISHER_PREFIX:-uit_medseg_rtsp_pub}"

declare -A FILES
FILES[cam01]="${ROOT_DIR}/data/test.mp4"
FILES[cam02]="${ROOT_DIR}/data/cam2_fake.mp4"
FILES[cam03]="${ROOT_DIR}/data/cam3_fake.mp4"
FILES[cam04]="${ROOT_DIR}/data/cam4_fake.mp4"
FILES[cam01]="${RTSP_SIM_CAM01_FILE:-${FILES[cam01]}}"
FILES[cam02]="${RTSP_SIM_CAM02_FILE:-${FILES[cam02]}}"
FILES[cam03]="${RTSP_SIM_CAM03_FILE:-${FILES[cam03]}}"
FILES[cam04]="${RTSP_SIM_CAM04_FILE:-${FILES[cam04]}}"

# Protocol mode: 'rtsp' (default) hoặc 'hls' (push RTMP → MediaMTX → HLS)
PROTOCOL="${PROTOCOL:-rtsp}"

declare -a CAMS=(cam01 cam02 cam03 cam04)

pid_file() { echo "${STATE_DIR}/$1.pid"; }
backend_file() { echo "${STATE_DIR}/$1.backend"; }
log_file() { echo "${LOG_DIR}/$1.log"; }
stream_url() { echo "rtsp://${RTSP_HOST}:${RTSP_PORT}/$1"; }
rtmp_publish_url() { echo "rtmp://${RTMP_HOST}:${RTMP_PORT}/$1"; }
# HLS URL do MediaMTX serve (API port :8888)
hls_url() { echo "http://${HLS_HOST}:${HLS_PORT}/$1/index.m3u8"; }
container_name() { echo "${PUBLISHER_CONTAINER_PREFIX}_$1"; }

docker_container_exists() {
  local cname="$1"
  docker ps -a --format '{{.Names}}' | grep -Fxq "${cname}"
}

is_running_host() {
  local cam="$1"
  local pf
  pf="$(pid_file "${cam}")"
  [[ -f "${pf}" ]] || return 1
  local pid
  pid="$(cat "${pf}")"
  [[ -n "${pid}" ]] || return 1
  kill -0 "${pid}" 2>/dev/null
}

is_running_docker() {
  local cam="$1"
  local cname
  cname="$(container_name "${cam}")"
  docker ps --format '{{.Names}}' | grep -Fxq "${cname}"
}

is_running() {
  local cam="$1"
  is_running_host "${cam}" || is_running_docker "${cam}"
}

start_host_cam() {
  local cam="$1"
  local mode="$2"
  local input="$3"
  local url="$4"
  local lf
  lf="$(log_file "${cam}")"

  local -a cmd
  cmd=("${FFMPEG_BIN}" -hide_banner -loglevel warning)
  if [[ "${mode}" != "burst" ]]; then
    cmd+=(-re)
  fi
  cmd+=(-stream_loop -1 -i "${input}" -an -c:v copy -f rtsp -rtsp_transport "${RTSP_TRANSPORT}" "${url}")

  nohup "${cmd[@]}" >"${lf}" 2>&1 &
  local pid=$!
  echo "${pid}" >"$(pid_file "${cam}")"
  echo "host" >"$(backend_file "${cam}")"

  echo "[rtsp-sim] Started ${cam} | mode=${mode} | backend=host | pid=${pid} | url=${url}"
}

start_host_cam_rtmp() {
  local cam="$1"
  local mode="$2"
  local input="$3"
  local url="$4"   # RTMP publish URL
  local lf
  lf="$(log_file "${cam}")"

  local -a cmd
  cmd=("${FFMPEG_BIN}" -hide_banner -loglevel warning)
  if [[ "${mode}" != "burst" ]]; then
    cmd+=(-re)
  fi
  # Push RTMP → MediaMTX sẽ tự serve HLS
  cmd+=(-stream_loop -1 -i "${input}" -an -c:v copy -f flv "${url}")

  nohup "${cmd[@]}" >"${lf}" 2>&1 &
  local pid=$!
  echo "${pid}" >"$(pid_file "${cam}")"
  echo "host" >"$(backend_file "${cam}")"

  echo "[rtsp-sim] Started ${cam} | mode=${mode} | backend=host(rtmp) | pid=${pid} | publish=${url}"
}

start_docker_cam() {
  local cam="$1"
  local mode="$2"
  local input="$3"
  local url="$4"
  local cname
  cname="$(container_name "${cam}")"

  if docker_container_exists "${cname}"; then
    docker rm -f "${cname}" >/dev/null 2>&1 || true
  fi

  local -a cmd
  cmd=(-hide_banner -loglevel warning)
  if [[ "${mode}" != "burst" ]]; then
    cmd+=(-re)
  fi
  cmd+=(-stream_loop -1 -i /input.mp4 -an -c:v copy -f rtsp -rtsp_transport "${RTSP_TRANSPORT}" "${url}")

  docker run -d \
    --name "${cname}" \
    --network host \
    --entrypoint ffmpeg \
    -v "${input}:/input.mp4:ro" \
    "${FFMPEG_DOCKER_IMAGE}" \
    "${cmd[@]}" >/dev/null

  rm -f "$(pid_file "${cam}")"
  echo "docker" >"$(backend_file "${cam}")"

  echo "[rtsp-sim] Started ${cam} | mode=${mode} | backend=docker | container=${cname} | url=${url}"
}

start_docker_cam_rtmp() {
  local cam="$1"
  local mode="$2"
  local input="$3"
  local url="$4"   # RTMP publish URL
  local cname
  cname="$(container_name "${cam}")"

  if docker_container_exists "${cname}"; then
    docker rm -f "${cname}" >/dev/null 2>&1 || true
  fi

  local -a cmd
  cmd=(-hide_banner -loglevel warning)
  if [[ "${mode}" != "burst" ]]; then
    cmd+=(-re)
  fi
  # Push RTMP → MediaMTX sẽ tự serve HLS
  cmd+=(-stream_loop -1 -i /input.mp4 -an -c:v copy -f flv "${url}")

  docker run -d \
    --name "${cname}" \
    --network host \
    --entrypoint ffmpeg \
    -v "${input}:/input.mp4:ro" \
    "${FFMPEG_DOCKER_IMAGE}" \
    "${cmd[@]}" >/dev/null

  rm -f "$(pid_file "${cam}")"
  echo "docker" >"$(backend_file "${cam}")"

  echo "[rtsp-sim] Started ${cam} | mode=${mode} | backend=docker(rtmp) | container=${cname} | publish=${url}"
}

start_cam() {
  local cam="$1"
  local mode="$2"

  if [[ -z "${FILES[${cam}]:-}" ]]; then
    echo "[rtsp-sim] Unknown camera: ${cam}" >&2
    return 1
  fi

  local input
  input="${FILES[${cam}]}"
  if [[ ! -f "${input}" ]]; then
    echo "[rtsp-sim] Missing input file for ${cam}: ${input}" >&2
    return 1
  fi

  if is_running "${cam}"; then
    if is_running_host "${cam}"; then
      echo "[rtsp-sim] ${cam} already running (backend=host, pid=$(cat "$(pid_file "${cam}")"))"
    else
      echo "[rtsp-sim] ${cam} already running (backend=docker, container=$(container_name "${cam}"))"
    fi
    return 0
  fi

  local url
  if [[ "${PROTOCOL}" == "hls" ]]; then
    url="$(rtmp_publish_url "${cam}")"
    if [[ "${FORCE_DOCKER_FFMPEG}" == "1" ]]; then
      start_docker_cam_rtmp "${cam}" "${mode}" "${input}" "${url}"
      return 0
    fi
    if command -v "${FFMPEG_BIN}" >/dev/null 2>&1; then
      start_host_cam_rtmp "${cam}" "${mode}" "${input}" "${url}"
      return 0
    fi
    echo "[rtsp-sim] ffmpeg not found on host PATH, fallback to Docker image ${FFMPEG_DOCKER_IMAGE}"
    start_docker_cam_rtmp "${cam}" "${mode}" "${input}" "${url}"
  else
    url="$(stream_url "${cam}")"
    if [[ "${FORCE_DOCKER_FFMPEG}" == "1" ]]; then
      start_docker_cam "${cam}" "${mode}" "${input}" "${url}"
      return 0
    fi
    if command -v "${FFMPEG_BIN}" >/dev/null 2>&1; then
      start_host_cam "${cam}" "${mode}" "${input}" "${url}"
      return 0
    fi
    echo "[rtsp-sim] ffmpeg not found on host PATH, fallback to Docker image ${FFMPEG_DOCKER_IMAGE}"
    start_docker_cam "${cam}" "${mode}" "${input}" "${url}"
  fi
}

stop_cam() {
  local cam="$1"
  local pf
  pf="$(pid_file "${cam}")"
  local bf
  bf="$(backend_file "${cam}")"
  local stopped=0

  if is_running_host "${cam}"; then
    local pid
    pid="$(cat "${pf}")"
    kill "${pid}" 2>/dev/null || true
    sleep 0.2
    kill -9 "${pid}" 2>/dev/null || true
    stopped=1
    echo "[rtsp-sim] Stopped ${cam} (backend=host, pid=${pid})"
  fi

  local cname
  cname="$(container_name "${cam}")"
  if docker_container_exists "${cname}"; then
    docker rm -f "${cname}" >/dev/null 2>&1 || true
    stopped=1
    echo "[rtsp-sim] Stopped ${cam} (backend=docker, container=${cname})"
  fi

  rm -f "${pf}" "${bf}"
  if [[ "${stopped}" -eq 0 ]]; then
    echo "[rtsp-sim] ${cam} not running"
  fi
}

status_cam() {
  local cam="$1"
  local url
  if [[ "${PROTOCOL}" == "hls" ]]; then
    url="HLS=$(hls_url "${cam}") [publish=$(rtmp_publish_url "${cam}")]"
  else
    url="$(stream_url "${cam}")"
  fi
  if is_running_host "${cam}"; then
    echo "[rtsp-sim] ${cam}: RUNNING backend=host pid=$(cat "$(pid_file "${cam}")") | ${url}"
    return 0
  fi

  local cname
  cname="$(container_name "${cam}")"
  if is_running_docker "${cam}"; then
    echo "[rtsp-sim] ${cam}: RUNNING backend=docker container=${cname} | ${url}"
    return 0
  fi

  echo "[rtsp-sim] ${cam}: STOPPED | ${url}"
}

logs_cam() {
  local cam="$1"
  local cname
  cname="$(container_name "${cam}")"

  if docker_container_exists "${cname}"; then
    docker logs -f "${cname}"
    return 0
  fi

  local lf
  lf="$(log_file "${cam}")"
  if [[ -f "${lf}" ]]; then
    tail -f "${lf}"
    return 0
  fi

  echo "[rtsp-sim] No logs found for ${cam}" >&2
  return 1
}

case "${ACTION}" in
  up)
    for cam in "${CAMS[@]}"; do
      start_cam "${cam}" "${MODE}"
    done
    ;;

  down)
    for cam in "${CAMS[@]}"; do
      stop_cam "${cam}"
    done
    ;;

  restart)
    if [[ -z "${CAMERA}" ]]; then
      echo "Usage: $0 restart <cam01|cam02|cam03> [realtime|burst]" >&2
      exit 1
    fi
    stop_cam "${CAMERA}"
    start_cam "${CAMERA}" "${MODE}"
    ;;

  burst)
    if [[ -z "${CAMERA}" ]]; then
      echo "Usage: $0 burst <cam01|cam02|cam03>" >&2
      exit 1
    fi
    stop_cam "${CAMERA}"
    start_cam "${CAMERA}" "burst"
    ;;

  status)
    for cam in "${CAMS[@]}"; do
      status_cam "${cam}"
    done
    ;;

  logs)
    if [[ -z "${CAMERA}" ]]; then
      echo "Usage: $0 logs <cam01|cam02|cam03>" >&2
      exit 1
    fi
    logs_cam "${CAMERA}"
    ;;

  *)
    echo "Usage: $0 {up|down|status|restart <cam> [mode]|burst <cam>|logs <cam>}" >&2
    echo "  PROTOCOL env: rtsp (default) | hls (push RTMP → MediaMTX → HLS)" >&2
    echo "  Cameras: cam01 cam02 cam03 cam04" >&2
    exit 1
    ;;
esac
