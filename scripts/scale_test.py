#!/usr/bin/env python3
"""
scale_test.py — Script for load testing and scaling camera inputs.
Allows spinning up N simulated camera streams and generating configurations dynamically.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.resolve()
CONFIG_DIR = ROOT_DIR / "apps/vision_service/configs/camera"
VIDEO_PATH = ROOT_DIR / "data/cam2_fake.mp4"
CONTAINER_PREFIX = "uit_medseg_rtsp_pub_cam_scale"


def run_cmd(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, check=check, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command {' '.join(cmd)}: {e.stderr}", file=sys.stderr)
        if check:
            sys.exit(1)
        return e


def get_active_containers() -> list[str]:
    cmd = ["docker", "ps", "-a", "--filter", f"name={CONTAINER_PREFIX}", "--format", "{{.Names}}"]
    res = run_cmd(cmd, check=False)
    if res.returncode != 0:
        return []
    return [line.strip() for line in res.stdout.splitlines() if line.strip()]


def clean_up_scaling() -> None:
    print(">>> Cleaning up scaling configs and containers...")
    
    # 1. Remove all scale containers
    containers = get_active_containers()
    if containers:
        print(f"Stopping and removing {len(containers)} scaling containers...")
        for container in containers:
            run_cmd(["docker", "rm", "-f", container], check=False)
    else:
        print("No active scaling containers found.")

    # 2. Remove all scale config yaml files
    if CONFIG_DIR.exists():
        configs = list(CONFIG_DIR.glob("camera_scale_*.yaml"))
        if configs:
            print(f"Removing {len(configs)} scaling config files...")
            for config in configs:
                config.unlink(missing_ok=True)
        else:
            print("No scaling configs found.")
            
    print(">>> Cleanup completed successfully!")


def start_scaling(num_cams: int) -> None:
    if num_cams < 1:
        print("Number of cameras must be greater than or equal to 1.")
        sys.exit(1)
        
    print(f">>> Initializing scaling test for {num_cams} cameras...")
    
    # Ensure clean slate first
    clean_up_scaling()
    
    if not VIDEO_PATH.exists():
        print(f"Error: Base video file not found at {VIDEO_PATH}", file=sys.stderr)
        sys.exit(1)
        
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating configs and launching {num_cams} stream publishers...")
    
    for i in range(1, num_cams + 1):
        cam_id = f"cam_scale_{i:03d}"
        gate_name = f"scale_gate_{i:03d}"
        cname = f"{CONTAINER_PREFIX}_{i:03d}"
        stream_path = f"cam_scale_{i:03d}"
        
        # 1. Write vision-service camera config
        yaml_content = f"""camera_id: {cam_id}
name: {gate_name}
enabled: true

stream:
  type: hls
  uri: "http://mediamtx:8888/{stream_path}/index.m3u8?cookieCheck=1"
  reconnect_interval_sec: 10
  timeout_sec: 15
  decoder_drop_frame_interval: 0

detection:
  min_confidence: 0.5
  roi:
    enabled: false
    polygon: []
"""
        yaml_file = CONFIG_DIR / f"camera_scale_{i:03d}.yaml"
        yaml_file.write_text(yaml_content, encoding="utf-8")
        
        # 2. Launch ffmpeg publisher container
        cmd = [
            "docker", "run", "-d",
            "--name", cname,
            "--network", "cs317_mlops_default",
            "--entrypoint", "ffmpeg",
            "-v", f"{VIDEO_PATH}:/input.mp4:ro",
            "jrottenberg/ffmpeg:8-scratch",
            "-hide_banner", "-loglevel", "warning",
            "-re", "-stream_loop", "-1",
            "-i", "/input.mp4",
            "-an", "-c:v", "copy",
            "-f", "flv",
            f"rtmp://mediamtx:1935/{stream_path}"
        ]
        
        run_cmd(cmd)
        print(f" [+] Started {cam_id} (container: {cname})")
        
    print(f"\n>>> Success! Launched {num_cams} mock cameras successfully.")
    print(">>> To run inference on them, restart your vision service:")
    print("    TELEGRAM_ENABLED=false make run")
    print(">>> To shut down and restore original state, run:")
    print("    python3 scripts/scale_test.py down")


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepStream Camera Scale and Load Testing Script")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Subcommand: up
    parser_up = subparsers.add_parser("up", help="Scale up N cameras")
    parser_up.add_argument("-n", "--num-cams", type=int, required=True, help="Number of simulated cameras to scale")
    
    # Subcommand: down
    subparsers.add_parser("down", help="Tear down all scaling cameras and restore configs")
    
    args = parser.parse_args()
    
    if args.command == "up":
        start_scaling(args.num_cams)
    elif args.command == "down":
        clean_up_scaling()


if __name__ == "__main__":
    main()
