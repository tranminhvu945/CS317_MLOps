"""
app.py — Web UI Control Plane (FastAPI)
───────────────────────────────────────
Giao diện điều khiển trung tâm cho hệ thống giám sát đa camera.

Chạy độc lập trên cổng 8500. Khi người dùng thao tác trên UI, service này sẽ:
  1. Gọi REST API nội tại của nvmultiurisrcbin (cổng 9091) để áp dụng thay đổi
     NGAY LẬP TỨC mà không cần khởi động lại DeepStream pipeline (Zero Downtime).
  2. Ghi / xóa thông tin vào file YAML trong `configs/camera/` để đảm bảo
     tính bền vững (Persistence) khi khởi động lại server sau này.

Khởi chạy:
  uvicorn apps.web_ui.app:app --host 0.0.0.0 --port 8500 --reload
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from apps.web_ui import camera_config as cam_cfg

# ─── Config ────────────────────────────────────────────────────────────────────
DEEPSTREAM_API_HOST = os.getenv("DEEPSTREAM_API_HOST", "127.0.0.1")
DEEPSTREAM_API_PORT = int(os.getenv("DEEPSTREAM_API_PORT", "9091"))
DEEPSTREAM_API_BASE = f"http://{DEEPSTREAM_API_HOST}:{DEEPSTREAM_API_PORT}"

# MediaMTX — server nhận RTMP từ DeepStream và phục vụ WebRTC/WHEP
MEDIAMTX_HOST = os.getenv("MEDIAMTX_HOST", "127.0.0.1")
MEDIAMTX_PORT = int(os.getenv("MEDIAMTX_PORT", "8888"))

STATIC_DIR = Path(__file__).parent / "static"

# ─── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MLOps Camera Control Plane",
    description="Enterprise Zero-Downtime Multi-Camera Management Dashboard",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sử dụng client duy nhất để duy trì cookie/session xác thực với MediaMTX
http_client = httpx.AsyncClient(timeout=10.0)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOTS_DIR = PROJECT_ROOT / "storage" / "snapshots"
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/snapshots", StaticFiles(directory=str(SNAPSHOTS_DIR)), name="snapshots")

DB_PATH = PROJECT_ROOT / "storage" / "violations.db"

def get_violations(camera_id: Optional[str] = None, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    """Lấy danh sách vi phạm từ SQLite database."""
    if not DB_PATH.exists():
        return []
    
    violations = []
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM violations"
        params = []
        
        if camera_id:
            query += " WHERE camera_id = ?"
            params.append(camera_id)
            
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        for row in rows:
            violations.append(dict(row))
        conn.close()
    except Exception as exc:
        print(f"Error fetching violations from DB: {exc}")
    return violations



# ─── Schemas ───────────────────────────────────────────────────────────────────
class AddCameraRequest(BaseModel):
    camera_id: str = Field(..., description="ID duy nhất của camera, ví dụ: cam_005")
    name: str = Field(..., description="Tên hiển thị của camera")
    stream_type: Literal["rtsp", "hls", "file"] = Field(default="hls")
    uri: str = Field(..., description="URI của luồng video")
    enabled: bool = Field(default=True)
    min_confidence: float = Field(default=0.15, ge=0.0, le=1.0)


class ToggleCameraRequest(BaseModel):
    enabled: bool


# ─── Internal Helper ───────────────────────────────────────────────────────────
async def _call_deepstream_api(method: str, path: str, payload: Optional[Dict] = None) -> Dict[str, Any]:
    """Gọi REST API nội tại của nvmultiurisrcbin để thêm/xóa luồng động."""
    url = f"{DEEPSTREAM_API_BASE}{path}"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            if method == "POST":
                resp = await client.post(url, json=payload or {})
            else:
                resp = await client.request(method, url, json=payload or {})
        return {"status_code": resp.status_code, "body": resp.text[:500]}
    except httpx.ConnectError:
        return {"status_code": -1, "body": "DeepStream REST API không khả dụng (pipeline chưa chạy hoặc cổng 9091 chưa mở). Cấu hình YAML đã được lưu."}
    except Exception as exc:
        return {"status_code": -1, "body": str(exc)}


# ─── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_dashboard():
    """Phục vụ file HTML giao diện Dashboard chính."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>UI not found. Place index.html in apps/web_ui/static/</h1>", status_code=404)
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/api/cameras", response_model=List[Dict])
async def list_cameras():
    """Lấy danh sách toàn bộ camera đã cấu hình."""
    return cam_cfg.list_cameras()


@app.get("/api/cameras/{camera_id}")
async def get_camera(camera_id: str):
    """Lấy chi tiết cấu hình của một camera theo ID."""
    cam = cam_cfg.get_camera(camera_id)
    if cam is None:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")
    return cam


@app.post("/api/cameras", status_code=201)
async def add_camera(req: AddCameraRequest):
    """
    Thêm camera mới — Zero Downtime:
    1. Ghi file YAML cấu hình mới vào disk.
    2. Gọi REST API của DeepStream để nạp luồng vào GPU ngay lập tức.
    """
    try:
        config = cam_cfg.create_camera(
            camera_id=req.camera_id,
            name=req.name,
            stream_type=req.stream_type,
            uri=req.uri,
            enabled=req.enabled,
            min_confidence=req.min_confidence,
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    # Gọi DeepStream REST API để thêm luồng ngay lập tức (Zero Downtime)
    ds_result = {}
    if req.enabled:
        ds_result = await _call_deepstream_api("POST", "/stream/add", {"uri": req.uri})

    return {
        "message": "Camera added successfully.",
        "config": config,
        "deepstream_api": ds_result,
    }


@app.patch("/api/cameras/{camera_id}/toggle")
async def toggle_camera(camera_id: str, req: ToggleCameraRequest):
    """
    Bật hoặc tắt camera — Zero Downtime:
    1. Cập nhật `enabled` trong file YAML tương ứng.
    2. Nếu bật: gọi DeepStream API /stream/add để nạp luồng vào GPU.
       Nếu tắt: gọi DeepStream API /stream/remove để giải phóng GPU.
    """
    # Xác định source_id trong GPU (chỉ đếm các camera đang bật) trước khi cập nhật YAML
    active_cameras = [c for c in cam_cfg.list_cameras() if c.get("enabled") is not False]
    source_id = next(
        (i for i, c in enumerate(active_cameras) if c.get("camera_id") == camera_id),
        None,
    )

    try:
        config = cam_cfg.set_camera_enabled(camera_id, req.enabled)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    uri = config.get("stream", {}).get("uri", "")
    ds_result = {}
    if req.enabled and uri:
        ds_payload = {"uri": uri}
        if source_id is not None:
            ds_payload["source-id"] = source_id
        ds_result = await _call_deepstream_api("POST", "/stream/add", ds_payload)
    elif not req.enabled and source_id is not None:
        ds_result = await _call_deepstream_api("POST", "/stream/remove", {"source-id": source_id})

    return {
        "message": f"Camera '{camera_id}' {'enabled' if req.enabled else 'disabled'} successfully.",
        "config": config,
        "deepstream_api": ds_result,
    }


@app.delete("/api/cameras/{camera_id}")
async def delete_camera(camera_id: str):
    """
    Xóa camera vĩnh viễn — Zero Downtime:
    1. Gọi DeepStream API để gỡ luồng khỏi GPU ngay lập tức.
    2. Xóa file YAML cấu hình khỏi disk.
    """
    active_cameras = [c for c in cam_cfg.list_cameras() if c.get("enabled") is not False]
    source_id = next(
        (i for i, c in enumerate(active_cameras) if c.get("camera_id") == camera_id),
        None,
    )

    ds_result = {}
    if source_id is not None:
        ds_result = await _call_deepstream_api("POST", "/stream/remove", {"source-id": source_id})

    deleted = cam_cfg.delete_camera(camera_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")

    return {
        "message": f"Camera '{camera_id}' deleted successfully.",
        "deepstream_api": ds_result,
    }


@app.get("/api/violations", response_model=List[Dict])
async def list_violations(camera_id: Optional[str] = None, limit: int = 100, offset: int = 0):
    """Lấy danh sách toàn bộ vi phạm đã lưu."""
    return get_violations(camera_id, limit, offset)


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "web-ui-control-plane", "deepstream_api_port": DEEPSTREAM_API_PORT}


@app.get("/api/hls/{stream}/{filename:path}")
async def hls_proxy(request: Request, stream: str, filename: str):
    """Proxy luồng HLS từ MediaMTX.

    Giải quyết triệt để rào cản CORS và xác thực Cookie (cookieCheck) trên trình duyệt:
    Trình duyệt gọi endpoint này trên cổng 8500 (cùng origin với trang Web UI),
    FastAPI đóng vai trò client tự động xử lý redirect và duy trì session cookie với MediaMTX.
    """
    target_url = f"http://{MEDIAMTX_HOST}:{MEDIAMTX_PORT}/{stream}/{filename}"
    
    query_params = request.url.query
    if query_params:
        target_url = f"{target_url}?{query_params}"
        
    try:
        resp = await http_client.get(target_url, follow_redirects=True)
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers={
                "Content-Type": resp.headers.get("Content-Type", "application/octet-stream"),
                "Cache-Control": resp.headers.get("Cache-Control", "no-cache"),
                "Access-Control-Allow-Origin": "*",
            },
        )
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"Không thể kết nối MediaMTX tại {target_url}. Kiểm tra 'make mediamtx-up' và luồng '{stream}'.",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
