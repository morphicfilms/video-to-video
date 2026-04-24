# Copyright 2025-2026 Morphic Inc. Licensed under Apache 2.0.
"""
app.py — Main entry point for the V2V 3D Camera Path Visualizer.

Usage (local):
    python -m visualizer.app --video input.mp4 --depth depths.npz

Usage (remote server — open browser at http://localhost:8080 after SSH tunnel):
    ssh -L 8080:localhost:8080 user@server
    python -m visualizer.app --video input.mp4 --depth depths.npz --port 8080

Controls:
  - Navigate the 3D scene with mouse (orbit / pan / zoom)
  - Scrub through video frames using the Frame slider
  - Click "Add Camera at Current View" to place a keyframe at wherever you are looking
  - Drag the yellow gizmos to reposition/reorient keyframes
  - Click "Preview Path" to visualise the interpolated path
  - Click "Export cam_info.json" when satisfied
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
from pipeline_spec import (
    is_valid_wan_frame_count,
    max_wan_frames_for_source,
    render_frames_for_wan_output,
    snap_to_valid_wan_output,
    wan_consumed_frames,
)
import viser

from .camera_editor import CameraEditor, PRESET_NAMES, generate_preset, _wxyz_to_rotation
from .export import export_cam_info
from .interpolation import _OPENGL_TO_OPENCV, c2w_to_w2c, EASING_MODES
from .scene import (
    compute_intrinsics,
    get_source_camera,
    load_depth_maps,
    load_video_frames,
    transform_points_to_world,
    unproject_frame,
)


def _fov_from_K(K: np.ndarray, H: int) -> float:
    """Compute vertical field-of-view in degrees from intrinsic matrix."""
    fy = float(K[1, 1])
    return float(np.degrees(2 * np.arctan(H / (2 * fy))))


def _opencv_pts_to_viser(pts: np.ndarray) -> np.ndarray:
    """
    Transform 3-D points from OpenCV world space to Viser's Y-up world space.

    Applies: y_viser = -y_opencv, z_viser = -z_opencv (flip Y and Z).
    """
    out = pts.copy()
    out[:, 1] = -pts[:, 1]
    out[:, 2] = -pts[:, 2]
    return out


def _keyframe_c2w_to_export_opencv(c2w_internal: np.ndarray) -> np.ndarray:
    """
    Convert internally stored keyframe poses to export-time OpenCV c2w.

    Internal keyframes are already stored in OpenCV convention (produced by
    `_display_camera_pose_to_opencv`), so this is an identity conversion.
    """
    return c2w_internal.astype(np.float32)


def _estimate_scene_center(points_per_frame: list[np.ndarray]) -> np.ndarray:
    """
    Estimate a robust display-space scene center from sampled point clouds.

    Uses a weighted mean over per-frame subsamples to keep memory bounded.
    """
    acc = np.zeros(3, dtype=np.float64)
    n_total = 0
    for pts in points_per_frame:
        if pts.size == 0:
            continue
        step = max(1, len(pts) // 5000)
        sample = pts[::step]
        if sample.size == 0:
            continue
        acc += sample.sum(axis=0, dtype=np.float64)
        n_total += int(sample.shape[0])
    if n_total == 0:
        return np.zeros(3, dtype=np.float32)
    return (acc / n_total).astype(np.float32)


VIEWFINDER_MODES = ("Border only", "Rule of thirds", "Off")


def _make_viewfinder_overlay(aspect: float, mode: str = "Border only") -> np.ndarray | None:
    """
    Build a framing guide RGBA image (fully transparent interior).

    Shown as a camera-attached plane in 3D to indicate the captured frame bounds.
    Only displayed when the user is not actively navigating the scene.
    """
    if mode == "Off":
        return None

    h = 720
    w = max(320, min(1920, int(round(h * max(float(aspect), 1e-3)))))
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    color      = np.array([80, 160, 255], dtype=np.uint8)  # blue
    alpha_edge = 220
    alpha_grid = 120
    t = max(2, h // 240)   # border thickness (~3 px at 720p)

    # Full-frame border
    rgba[:t,  :,  :3] = color;  rgba[:t,  :,  3] = alpha_edge
    rgba[-t:, :,  :3] = color;  rgba[-t:, :,  3] = alpha_edge
    rgba[:,  :t,  :3] = color;  rgba[:,  :t,  3] = alpha_edge
    rgba[:, -t:, :3]  = color;  rgba[:, -t:, 3]  = alpha_edge

    if mode == "Rule of thirds":
        gt = max(1, t)
        for frac in (1, 2):
            y = h * frac // 3
            x = w * frac // 3
            rgba[max(0, y - gt):min(h, y + gt), :, :3] = color
            rgba[max(0, y - gt):min(h, y + gt), :, 3]  = alpha_grid
            rgba[:, max(0, x - gt):min(w, x + gt), :3]  = color
            rgba[:, max(0, x - gt):min(w, x + gt), 3]   = alpha_grid
    else:
        # Center crosshair for "Border only"
        cx, cy = w // 2, h // 2
        c_len = max(12, h // 30)
        ct    = max(1, t // 2)
        alpha_tick = 170
        rgba[max(0, cy - ct):min(h, cy + ct),
             max(0, cx - c_len):min(w, cx + c_len), :3] = color
        rgba[max(0, cy - ct):min(h, cy + ct),
             max(0, cx - c_len):min(w, cx + c_len), 3]  = alpha_tick
        rgba[max(0, cy - c_len):min(h, cy + c_len),
             max(0, cx - ct):min(w, cx + ct), :3] = color
        rgba[max(0, cy - c_len):min(h, cy + c_len),
             max(0, cx - ct):min(w, cx + ct), 3]  = alpha_tick
    return rgba


def _serve_redirect(port: int, target_url: str, timeout: float = 15.0) -> None:
    """Serve a temporary redirect page on *port* that sends browsers to *target_url*.

    The server shuts itself down after *timeout* seconds (enough for the
    browser to follow the redirect, then app_autodepth restarts the upload server).
    """
    from http.server import BaseHTTPRequestHandler, HTTPServer

    _HTML = (
        "<!DOCTYPE html><html><head>"
        f'<meta http-equiv="refresh" content="2;url={target_url}">'
        "</head><body style='background:#1a1a2e;color:#eee;font-family:system-ui,sans-serif;"
        "display:flex;align-items:center;justify-content:center;height:100vh;margin:0'>"
        f"<p>Redirecting to <a href='{target_url}' style='color:#7c8cf8'>upload page</a>...</p>"
        "</body></html>"
    ).encode()

    class _H(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(_HTML)))
            self.end_headers()
            self.wfile.write(_HTML)

        def log_message(self, *a: object) -> None:
            pass

    httpd = HTTPServer(("", port), _H)
    print(f"[app] Redirect server on :{port} → {target_url}")
    # Run in a daemon thread so it doesn't block the caller; it'll be cleaned
    # up when the process exits or the thread is GC'd after timeout.
    def _run_then_stop() -> None:
        time.sleep(timeout)
        httpd.shutdown()
        print("[app] Redirect server stopped.")

    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    threading.Thread(target=_run_then_stop, daemon=True).start()


def run(args: argparse.Namespace) -> None:
    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"[app] Loading video: {args.video}")
    frames, video_fps = load_video_frames(args.video, max_frames=args.max_frames, target_fps=args.target_fps)
    T, H, W, _ = frames.shape
    print(f"[app] Loaded {T} frames at {H}x{W}, effective fps={video_fps:.1f}")

    print(f"[app] Loading depth: {args.depth}")
    depths = load_depth_maps(args.depth)

    if depths.shape[0] != T:
        n = min(T, depths.shape[0])
        frames = frames[:n]
        depths = depths[:n]
        T = n
        print(f"[app] Trimmed to {T} frames (video/depth mismatch)")

    if depths.shape[1:] != (H, W):
        import cv2
        dH, dW = depths.shape[1], depths.shape[2]
        # Detect transposed dimensions (depth H,W swapped relative to video)
        if (dH, dW) == (W, H):
            print(f"[app] Transposing depth {dH}×{dW} → {H}×{W} (dimensions were swapped)")
            depths = np.transpose(depths, (0, 2, 1))
        else:
            print(f"[app] Resizing depth {dH}×{dW} → {H}×{W} to match video")
            depths = np.stack([
                cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR) for d in depths
            ])

    # ── Intrinsics & source camera ────────────────────────────────────────────
    K = compute_intrinsics(H, W, focal_multiplier=args.focal)
    valid_depths = depths[np.isfinite(depths) & (depths > 0)]
    if valid_depths.size == 0:
        raise ValueError("Depth file contains no finite positive values.")
    valid_depths_0 = depths[0][np.isfinite(depths[0]) & (depths[0] > 0)]
    depth_avg = float(np.median(valid_depths_0 if valid_depths_0.size else valid_depths))
    w2c_0, c2w_0 = get_source_camera(depth_avg, start_elevation=args.start_elevation)

    fov_deg = _fov_from_K(K, H)
    aspect  = W / H

    print(f"[app] depth_avg={depth_avg:.3f}  fov={fov_deg:.1f}  aspect={aspect:.3f}")

    # ── Pre-compute world-space point clouds for each frame ───────────────────
    print(f"[app] Unprojecting {T} frames (subsample={args.subsample}) ...")
    K_inv = np.linalg.inv(K)
    all_points: list[np.ndarray] = []   # Viser Y-up world space
    all_colors: list[np.ndarray] = []
    for i in range(T):
        pts, cols = unproject_frame(frames[i], depths[i], K, subsample=args.subsample, K_inv=K_inv)
        # Transform to OpenCV world space then flip to Viser Y-up space
        pts_world = transform_points_to_world(pts, c2w_0)
        pts_viser = _opencv_pts_to_viser(pts_world)
        all_points.append(pts_viser)
        all_colors.append(cols.astype(np.float32) / 255.0)
        if (i + 1) % 10 == 0 or i == T - 1:
            print(f"  {i+1}/{T}", end="\r", flush=True)
    print()

    # Recenter around frame 0 (source frame) so the source camera/frustum aligns
    # more intuitively with the first point cloud. Fallback to all frames if empty.
    scene_center_viser = _estimate_scene_center([all_points[0]])
    if all_points[0].size == 0 or not np.all(np.isfinite(scene_center_viser)):
        scene_center_viser = _estimate_scene_center(all_points)
    if np.linalg.norm(scene_center_viser) > 0:
        for i in range(T):
            if all_points[i].size:
                all_points[i] = all_points[i] - scene_center_viser[None, :]
    print(
        "[app] Display recenter offset (viser xyz): "
        f"{scene_center_viser[0]:.3f}, {scene_center_viser[1]:.3f}, {scene_center_viser[2]:.3f}"
    )

    # Estimate scene radius: 75th-percentile point distance from the recentered
    # centroid, computed on frame 0 (which is already shifted to origin).
    # Used to set a sensible initial viewer distance and orbit pivot.
    _pts0 = all_points[0] if (len(all_points) > 0 and all_points[0].size > 0) \
            else np.zeros((1, 3), dtype=np.float32)
    _dists0 = np.linalg.norm(_pts0, axis=1).astype(np.float64)
    scene_radius = float(np.percentile(_dists0, 75)) if _dists0.size > 0 else max(1.0, depth_avg)
    scene_radius = max(0.1, scene_radius)
    print(f"[app] scene_radius={scene_radius:.3f} (75th-pct point dist from centroid)")

    # ── Start Viser server ────────────────────────────────────────────────────
    server = viser.ViserServer(port=args.port, verbose=False)
    server.scene.set_up_direction("+y")
    server.scene.set_background_image(np.full((1, 1, 3), 255, dtype=np.uint8))
    print(f"[app] Viser server running on http://localhost:{args.port}")
    print("[app] Open the URL in your browser (or tunnel if on a remote server).")

    # Mutable shared state
    state = {
        "frame_idx": 0,
        "show_all":  False,
        "n_output":  render_frames_for_wan_output(min(snap_to_valid_wan_output(args.nframe), max_wan_frames_for_source(T))),
        "show_gizmos": False,
        "show_guide": True,
        "guide_mode": "Border only",
    }
    pc_handles: list[viser.PointCloudHandle | None] = [None] * T

    editor = CameraEditor(
        server,
        fov_deg=fov_deg,
        aspect=aspect,
        display_offset_viser=scene_center_viser,
    )
    # Source camera frustum is intentionally omitted: the synthetic source pose
    # does not yet correspond to true camera estimation and would be misleading.

    def _display_camera_pose_to_opencv(cam: viser.CameraHandle) -> np.ndarray:
        """Convert the current Viser browser camera to an OpenCV-convention c2w.

        Viser's camera convention: col0=right, col1=down, col2=forward (same
        local-axis convention as OpenCV).  The only difference is the world frame
        (Viser Y-up vs OpenCV Y-down), so we left-multiply by F = diag(1,-1,-1).
        """
        R_viser = _wxyz_to_rotation(cam.wxyz)
        # Position: display → Viser Y-up world → OpenCV world
        pos_display = np.asarray(cam.position, dtype=np.float32)
        pos_viser_world = pos_display + scene_center_viser   # undo recentering
        pos_opencv_world = pos_viser_world.copy()
        pos_opencv_world[1] = -pos_viser_world[1]            # undo Y flip
        pos_opencv_world[2] = -pos_viser_world[2]            # undo Z flip

        # Rotation: only world-frame flip needed (Viser already uses
        # col2=forward, col1=down — same camera-local axes as OpenCV).
        F = _OPENGL_TO_OPENCV[:3, :3]
        R_opencv = (F @ R_viser).astype(np.float32)

        c2w_opencv = np.eye(4, dtype=np.float32)
        c2w_opencv[:3, :3] = R_opencv
        c2w_opencv[:3, 3] = pos_opencv_world
        return c2w_opencv

    def _opencv_pose_to_display_viser(c2w_opencv: np.ndarray) -> np.ndarray:
        """Convert an OpenCV c2w to Viser display-space c2w.

        Position: OpenCV world → Viser Y-up world (flip Y,Z) → display (subtract center).
        Rotation: left-multiply F only (world-frame flip; camera-local axes unchanged).
        """
        F = _OPENGL_TO_OPENCV[:3, :3]
        R_opencv = c2w_opencv[:3, :3].astype(np.float32)
        R_viser = (F @ R_opencv).astype(np.float32)

        t = c2w_opencv[:3, 3].astype(np.float32)
        pos_display = np.array([t[0], -t[1], -t[2]], dtype=np.float32) - scene_center_viser

        c2w_viser = np.eye(4, dtype=np.float32)
        c2w_viser[:3, :3] = R_viser
        c2w_viser[:3, 3] = pos_display
        return c2w_viser

    # Per-client camera-attached framing-guide overlay handles.
    viewfinder_handles: dict[int, viser.ImageHandle] = {}
    # Settle timers: hide the guide immediately on camera move, show it after
    # _settle_delay seconds of stillness so it never flickers while orbiting.
    _settle_timers: dict[int, threading.Timer] = {}
    _settle_delay = 0.35   # seconds

    def _set_viewfinder_visible(client: viser.ClientHandle, visible: bool) -> None:
        h = viewfinder_handles.get(client.client_id)
        if h is not None:
            h.visible = bool(visible)

    def _cancel_settle_timer(client_id: int) -> None:
        t = _settle_timers.pop(client_id, None)
        if t is not None:
            t.cancel()

    def _sync_viewfinder_overlay(client: viser.ClientHandle) -> None:
        """Update guide position/orientation and make it visible (call when settled).

        The guide rectangle is placed AT the current orbit pivot (look_at) and
        sized using the OUTPUT camera's FOV and aspect ratio — not the viewer's.
        This means the box shows exactly the frame the output camera would capture
        at that scene depth, regardless of how the user has zoomed in the viewer.
        """
        if not state["show_guide"]:
            _set_viewfinder_visible(client, False)
            return

        try:
            cam = client.camera
            pos     = np.asarray(cam.position, dtype=np.float32)
            look    = np.asarray(cam.look_at,  dtype=np.float32)
            wxyz_cam = tuple(np.asarray(cam.wxyz, dtype=np.float32).tolist())
        except Exception:
            return

        if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(look)):
            return

        # Distance from viewer to orbit pivot — determines box size.
        d_to_lookat = float(np.linalg.norm(look - pos))
        if d_to_lookat < 0.01:
            return

        # Size the rectangle using the OUTPUT camera's FOV and aspect ratio.
        # fov_deg and aspect are captured from the run() closure (output camera params).
        output_fov_rad = float(np.radians(fov_deg))
        frame_h  = 2.0 * d_to_lookat * np.tan(output_fov_rad / 2.0)
        frame_w  = frame_h * aspect          # output camera W/H
        plane_pos = tuple(look.tolist())     # rectangle lives at the orbit pivot

        h = viewfinder_handles.get(client.client_id)
        # The overlay image has a fixed shape (output camera aspect); only create once.
        if h is None:
            overlay_rgba = _make_viewfinder_overlay(aspect=aspect, mode=state.get("guide_mode", "Border only"))
            if overlay_rgba is None:
                return
            h = client.scene.add_image(
                name="hud/viewfinder",
                image=overlay_rgba,
                render_width=1.0,
                render_height=1.0,
                format="png",
                cast_shadow=False,
                receive_shadow=False,
                position=plane_pos,
                wxyz=wxyz_cam,
                visible=False,   # made visible below once position is set
            )
            viewfinder_handles[client.client_id] = h

        h.position = plane_pos
        h.wxyz     = wxyz_cam
        h.scale    = (frame_w, frame_h, 1.0)
        h.visible  = True

    def _schedule_viewfinder_settle(client: viser.ClientHandle) -> None:
        """Hide guide immediately; re-show after _settle_delay s of camera stillness."""
        _cancel_settle_timer(client.client_id)
        _set_viewfinder_visible(client, False)
        if not state["show_guide"]:
            return

        def _on_settle() -> None:
            _settle_timers.pop(client.client_id, None)
            try:
                _sync_viewfinder_overlay(client)
            except Exception:
                pass

        t = threading.Timer(_settle_delay, _on_settle)
        _settle_timers[client.client_id] = t
        t.start()

    src_c2w_viser_display = _opencv_pose_to_display_viser(c2w_0)

    # Initial viewer position: use the source camera position in Viser display space.
    src_viewer_pos_display = src_c2w_viser_display[:3, 3].astype(np.float32)

    # Set initial camera. Both initial_camera and client.camera go through the
    # set_up_direction scene transform, so we use the raw display-space values.
    _init_cam_pos = tuple(src_viewer_pos_display.tolist())
    server.initial_camera.position = _init_cam_pos
    server.initial_camera.look_at = (0.0, 0.0, 0.0)
    server.initial_camera.up = (0.0, 1.0, 0.0)

    @server.on_client_connect
    def _on_client_connect(client: viser.ClientHandle) -> None:
        client.camera.position = _init_cam_pos
        client.camera.look_at = (0.0, 0.0, 0.0)
        client.camera.up_direction = (0.0, 1.0, 0.0)

        @client.camera.on_update
        def _on_client_camera_update(cam: viser.CameraHandle) -> None:
            _schedule_viewfinder_settle(cam.client)

        _schedule_viewfinder_settle(client)

    @server.on_client_disconnect
    def _on_client_disconnect(client: viser.ClientHandle) -> None:
        _cancel_settle_timer(client.client_id)
        viewfinder_handles.pop(client.client_id, None)

    # ── Scene transform gizmo ────────────────────────────────────────────────
    # Point clouds live under a Frame node so the gizmo can rotate the whole
    # scene without hiding children when the gizmo itself is toggled off.
    _scene_frame = server.scene.add_frame("scene_root", show_axes=False)
    _scene_gizmo = server.scene.add_transform_controls(
        "scene_gizmo",
        scale=scene_radius * 0.6,
        line_width=3.0,
        disable_sliders=True,
        visible=False,
    )

    @_scene_gizmo.on_update
    def _on_scene_gizmo_update(event: viser.TransformControlsEvent) -> None:
        _scene_frame.wxyz = event.target.wxyz
        _scene_frame.position = event.target.position

    _pc_lock = threading.Lock()

    # ── Helper: manage point cloud visibility ─────────────────────────────────
    def _ensure_pc(idx: int) -> None:
        with _pc_lock:
            if pc_handles[idx] is not None:
                return
            pc_handles[idx] = server.scene.add_point_cloud(
                name=f"scene_root/pointcloud/{idx:04d}",
                points=all_points[idx],
                colors=all_colors[idx],
                point_size=args.point_size,
            )

    def _show_frame(idx: int) -> None:
        for i, h in enumerate(pc_handles):
            if h is not None:
                h.visible = (state["show_all"] or i == idx)

    # Pre-load the first frame so the scene is not empty on open
    _ensure_pc(0)

    # Background preload: load remaining point clouds in a background thread
    # so frame scrubbing doesn't stutter on first visit.
    def _preload_all_pcs() -> None:
        for i in range(1, T):
            _ensure_pc(i)
            if pc_handles[i] is not None:
                pc_handles[i].visible = False
            time.sleep(0.05)  # throttle to avoid overwhelming the websocket
        print(f"[app] Background preload complete ({T} point clouds)")

    if T > 1:
        threading.Thread(target=_preload_all_pcs, daemon=True).start()

    # Playback FPS defaults to 16 (native rate of the video generation model).
    default_playback_fps = 16

    # ── Shutdown event (set by "Load New Video" button) ─────────────────────
    _shutdown_event = threading.Event()

    # ── Theme ─────────────────────────────────────────────────────────────────
    server.gui.configure_theme(
        control_layout="collapsible",
        dark_mode=True,
        show_logo=False,
        show_share_button=False,
    )

    # ── GUI — tabbed layout ──────────────────────────────────────────────────
    tab_group = server.gui.add_tab_group()

    # ── Scene tab ────────────────────────────────────────────────────────────
    with tab_group.add_tab("Scene", icon=viser.Icon.MOVIE):
        frame_slider = server.gui.add_slider(
            "Frame",
            min=0, max=T - 1, step=1,
            initial_value=0,
        )
        scene_play_btn = server.gui.add_button("Play", icon=viser.Icon.PLAYER_PLAY, color="teal")
        scene_fps_slider = server.gui.add_slider(
            "Playback FPS", min=1, max=30, step=1, initial_value=default_playback_fps
        )
        scene_loop_cb = server.gui.add_checkbox("Loop playback", initial_value=True)
        guide_dd = server.gui.add_dropdown(
            "Framing Guide", options=list(VIEWFINDER_MODES), initial_value="Border only",
        )
        with server.gui.add_folder("Display", expand_by_default=False):
            show_all_cb = server.gui.add_checkbox("Show all frames", initial_value=False)
            point_size_slider = server.gui.add_slider(
                "Point size", min=0.001, max=0.03, step=0.001,
                initial_value=args.point_size,
            )
            bg_color_dd = server.gui.add_dropdown(
                "Background",
                options=("White", "Light", "Dark", "Black"),
                initial_value="White",
            )
            scene_gizmo_cb = server.gui.add_checkbox(
                "Scene Gizmo (rotate/move scene)", initial_value=False,
            )
            btn_reset_cam = server.gui.add_button("Reset Camera", color="gray")

    # ── Camera tab ───────────────────────────────────────────────────────────
    with tab_group.add_tab("Camera", icon=viser.Icon.CAMERA):
        btn_add    = server.gui.add_button("Add Camera at Current View", color="green")
        btn_remove = server.gui.add_button("Remove Last Camera", color="yellow")
        btn_clear  = server.gui.add_button("Clear All Cameras", color="red")
        btn_undo   = server.gui.add_button("Undo", color="gray")
        btn_redo   = server.gui.add_button("Redo", color="gray")
        kf_count   = server.gui.add_text("Keyframes placed", initial_value="0",
                                          disabled=True)
        show_gizmos_cb = server.gui.add_checkbox("Show Keyframe Gizmos", initial_value=False)
        status_md   = server.gui.add_markdown(
            "_Place at least 2 cameras, then click Preview or Export._"
        )
        with server.gui.add_folder("Presets", expand_by_default=True):
            preset_dd = server.gui.add_dropdown(
                "Trajectory preset",
                options=list(PRESET_NAMES),
                initial_value=PRESET_NAMES[0],
            )
            btn_apply_preset = server.gui.add_button("Apply Preset", color="teal")
        with server.gui.add_folder("Path Settings", expand_by_default=False):
            _max_wan = max_wan_frames_for_source(T)
            _default_wan = min(snap_to_valid_wan_output(args.nframe), _max_wan)
            _default_render = render_frames_for_wan_output(_default_wan)
            nframe_num   = server.gui.add_number(
                "WAN output frames",
                initial_value=_default_wan,
                min=5, max=_max_wan, step=4,
            )
            render_frame_info = server.gui.add_text(
                "Render frames",
                initial_value=str(_default_render),
                disabled=True,
            )
            easing_dd = server.gui.add_dropdown(
                "Transition",
                options=EASING_MODES,
                initial_value="Linear",
            )
        btn_preview      = server.gui.add_button("Preview Path")
        btn_render_preview = server.gui.add_button("Render Video Preview", color="teal")
        # Video Preview (collapsed by default)
        with server.gui.add_folder("Video Preview", expand_by_default=False):
            btn_open_preview = server.gui.add_button(
                "Open Preview Window", color="teal", visible=False
            )
            preview_thumb = server.gui.add_image(
                np.zeros((H // 4, W // 4, 3), dtype=np.uint8),
                label="Last rendered frame",
                visible=False,
            )
            preview_info = server.gui.add_markdown("_No preview rendered yet._")

    # ── Export tab ────────────────────────────────────────────────────────────
    default_render_dir = str((Path(args.output).parent / "render_outputs").resolve())
    with tab_group.add_tab("Export", icon=viser.Icon.FILE_EXPORT):
        output_path_txt = server.gui.add_text(
            "Output path",
            initial_value=args.output,
        )
        btn_export = server.gui.add_button("Export cam_info.json", color="blue")

        with server.gui.add_folder("Render Assets", expand_by_default=True):
            render_out_dir_txt = server.gui.add_text(
                "Render output dir",
                initial_value=default_render_dir,
            )
            render_backend_dd = server.gui.add_dropdown(
                "Backend",
                options=("gpu_point", "numpy"),
                initial_value="gpu_point",
            )
            render_bg_txt = server.gui.add_text(
                "Backgrounds",
                initial_value="black,pink",
            )
            with server.gui.add_folder("Advanced", expand_by_default=False):
                render_device_txt = server.gui.add_text(
                    "Device",
                    initial_value="cuda",
                )
                render_python_txt = server.gui.add_text(
                    "Renderer Python",
                    initial_value="",
                    hint="Optional: path to a Python executable (e.g. py310 env with PyTorch3D)",
                )
                render_subsample_num = server.gui.add_number(
                    "Subsample", initial_value=1, min=1, max=16, step=1
                )
                render_radius_num = server.gui.add_number(
                    "Point Radius", initial_value=0.008, min=0.001, max=0.05, step=0.001
                )
                render_ppp_num = server.gui.add_number(
                    "Points / Pixel", initial_value=8, min=1, max=32, step=1
                )
                render_sobel_num = server.gui.add_number(
                    "Sobel Threshold", initial_value=0.35, min=0.05, max=1.0, step=0.01
                )
                render_nb_neighbors_num = server.gui.add_number(
                    "SOR Neighbors", initial_value=20, min=1, max=100, step=1
                )
                render_std_ratio_num = server.gui.add_number(
                    "SOR Std Ratio", initial_value=1.0, min=0.0, max=10.0, step=0.1
                )
            btn_render_assets = server.gui.add_button("Render Videos + Masks", color="teal")
        render_status_md = server.gui.add_markdown("")

    # ── Inference tab ────────────────────────────────────────────────────────
    with tab_group.add_tab("Inference", icon=viser.Icon.PLAYER_PLAY):
        server.gui.add_markdown(
            "_Generate the reshot video from a rendered condition pack._"
        )
        infer_pack_dir_txt = server.gui.add_text(
            "Condition pack dir",
            initial_value=default_render_dir,
        )
        infer_caption_txt = server.gui.add_text(
            "Caption",
            initial_value="",
        )
        infer_save_folder_txt = server.gui.add_text(
            "Output folder",
            initial_value="outputs",
        )
        with server.gui.add_folder("Model Settings", expand_by_default=False):
            infer_num_gpus = server.gui.add_number(
                "Num GPUs", initial_value=8, min=1, max=16, step=1,
            )
            infer_size_txt = server.gui.add_text("Resolution", initial_value="832*480")
            infer_steps_num = server.gui.add_number(
                "Sample steps", initial_value=40, min=1, max=100, step=1,
            )
            infer_high_lora_txt = server.gui.add_text(
                "High-noise LoRA", initial_value="",
            )
            infer_low_lora_txt = server.gui.add_text(
                "Low-noise LoRA", initial_value="",
            )
            infer_ckpt_dir_txt = server.gui.add_text(
                "Checkpoint dir (blank = auto-download)", initial_value="",
            )
        btn_gen_cmd = server.gui.add_button("Generate Command", color="blue")
        btn_launch_infer = server.gui.add_button("Launch Inference", color="green")
        infer_cmd_md = server.gui.add_markdown("")
        infer_status_md = server.gui.add_markdown("")

    # ── Results tab ───────────────────────────────────────────────────────────
    with tab_group.add_tab("Results", icon=viser.Icon.VIDEO):
        results_dir_txt = server.gui.add_text(
            "Results dir", initial_value=default_render_dir,
        )
        results_file_dd = server.gui.add_dropdown(
            "Video file",
            options=("render.mp4", "render_mask.mp4", "render_pink.mp4", "input.mp4"),
            initial_value="render.mp4",
        )
        btn_load_result = server.gui.add_button("Load Video", color="teal")
        results_thumb = server.gui.add_image(
            np.zeros((H // 4, W // 4, 3), dtype=np.uint8),
            label="Preview",
            visible=False,
        )
        results_frame_slider = server.gui.add_slider(
            "Frame", min=0, max=1, step=1, initial_value=0, visible=False,
        )
        results_status_md = server.gui.add_markdown("")

    _result_frames: list[np.ndarray] = []

    @btn_load_result.on_click
    def _on_load_result(event: viser.GuiEvent) -> None:
        import cv2
        rdir = Path(results_dir_txt.value.strip())
        fname = results_file_dd.value
        vpath = rdir / fname
        if not vpath.exists():
            # Check inference output dir too
            infer_dir = Path(infer_save_folder_txt.value.strip())
            candidates = list(infer_dir.glob("*.mp4")) if infer_dir.exists() else []
            if candidates:
                vpath = candidates[0]
            else:
                results_status_md.content = f"_Not found: {vpath}_"
                return
        try:
            cap = cv2.VideoCapture(str(vpath))
            _result_frames.clear()
            while True:
                ret, bgr = cap.read()
                if not ret:
                    break
                _result_frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            cap.release()
            if not _result_frames:
                results_status_md.content = "_No frames in video._"
                return
            results_frame_slider.max = len(_result_frames) - 1
            results_frame_slider.value = 0
            results_frame_slider.visible = True
            results_thumb.image = _result_frames[0]
            results_thumb.visible = True
            results_status_md.content = f"_Loaded {len(_result_frames)} frames from {vpath.name}_"
        except Exception as exc:
            results_status_md.content = f"_Error: {exc}_"

    @results_frame_slider.on_update
    def _on_result_frame(event: viser.GuiEvent) -> None:
        idx = int(event.target.value)
        if 0 <= idx < len(_result_frames):
            results_thumb.image = _result_frames[idx]

    # ── "Load New Video" button (below tabs) ────────────────────────────────
    server.gui.add_markdown("---")
    btn_new_video = server.gui.add_button(
        "Load New Video",
        color="gray",
        icon=viser.Icon.UPLOAD,
    )

    @btn_new_video.on_click
    def _on_new_video(event: viser.GuiEvent) -> None:
        upload_port = args.port + 1
        upload_url = f"http://localhost:{upload_port}/upload"
        # Redirect all connected clients to the upload page via injected JS,
        # with a small delay so the notification is visible first.
        for client in server.get_clients().values():
            client.add_notification(
                title="Returning to upload page",
                body="Redirecting to video upload...",
                loading=True,
                auto_close=3000,
            )
            # React's dangerouslySetInnerHTML doesn't execute <script> tags,
            # so we use an <img onerror> trick to run JS for the redirect.
            client.gui.add_html(
                f'<img src="x" onerror="setTimeout(function(){{window.location.href=\'{upload_url}\';}},2000)" style="display:none">'
            )

        # Shut down after a brief delay so the redirect JS is delivered.
        def _delayed_shutdown() -> None:
            time.sleep(1.5)
            _shutdown_event.set()

        threading.Thread(target=_delayed_shutdown, daemon=True).start()

    # ── Callbacks ─────────────────────────────────────────────────────────────
    # In viser 1.x all GUI callbacks receive a GuiEvent with .target = the handle.

    @frame_slider.on_update
    def _on_frame(event: viser.GuiEvent) -> None:
        idx = int(event.target.value)
        state["frame_idx"] = idx
        _ensure_pc(idx)
        _show_frame(idx)

    @show_all_cb.on_update
    def _on_show_all(event: viser.GuiEvent) -> None:
        state["show_all"] = bool(event.target.value)
        if state["show_all"]:
            for i in range(T):
                _ensure_pc(i)
        _show_frame(state["frame_idx"])

    @point_size_slider.on_update
    def _on_point_size(event: viser.GuiEvent) -> None:
        size = float(event.target.value)
        for h in pc_handles:
            if h is not None:
                h.point_size = size

    _BG_COLORS = {
        "Dark":  None,
        "White": (255, 255, 255),
        "Light": (220, 220, 220),
        "Black": (0, 0, 0),
    }

    @bg_color_dd.on_update
    def _on_bg_color(event: viser.GuiEvent) -> None:
        rgb = _BG_COLORS.get(event.target.value)
        if rgb is None:
            server.scene.set_background_image(None)
        else:
            server.scene.set_background_image(
                np.full((1, 1, 3), rgb, dtype=np.uint8)
            )

    @scene_gizmo_cb.on_update
    def _on_scene_gizmo(event: viser.GuiEvent) -> None:
        _scene_gizmo.visible = bool(event.target.value)

    @btn_reset_cam.on_click
    def _on_reset_cam(event: viser.GuiEvent) -> None:
        client = event.client
        if client is None:
            return
        client.camera.position = tuple(src_viewer_pos_display.tolist())
        client.camera.look_at = (0.0, 0.0, 0.0)
        client.camera.up_direction = (0.0, 1.0, 0.0)
        # Also reset scene gizmo + frame to identity
        _scene_gizmo.wxyz = (1.0, 0.0, 0.0, 0.0)
        _scene_gizmo.position = (0.0, 0.0, 0.0)
        _scene_frame.wxyz = (1.0, 0.0, 0.0, 0.0)
        _scene_frame.position = (0.0, 0.0, 0.0)

    @guide_dd.on_update
    def _on_guide_toggle(event: viser.GuiEvent) -> None:
        mode = event.target.value
        state["show_guide"] = mode != "Off"
        state["guide_mode"] = mode
        # Remove existing overlays so they get re-created with the new mode
        for cid, h in list(viewfinder_handles.items()):
            try:
                h.remove()
            except Exception:
                pass
        viewfinder_handles.clear()
        for c in server.get_clients().values():
            if state["show_guide"]:
                _schedule_viewfinder_settle(c)
            else:
                _cancel_settle_timer(c.client_id)

    scene_playback = {
        "playing": False,
        "fps": float(default_playback_fps),
        "loop": True,
        "thread": None,
    }

    def _stop_scene_playback() -> None:
        scene_playback["playing"] = False
        t = scene_playback["thread"]
        if t is not None and t.is_alive():
            t.join(timeout=1.0)
        scene_playback["thread"] = None
        try:
            scene_play_btn.label = "Play"
        except Exception:
            pass

    def _start_scene_playback_loop() -> None:
        while scene_playback["playing"]:
            cur = int(state["frame_idx"])
            nxt = cur + 1
            if nxt >= T:
                if scene_playback["loop"]:
                    nxt = 0
                else:
                    break
            frame_slider.value = int(nxt)
            try:
                scene_play_btn.label = f"Pause ({nxt + 1}/{T})"
            except Exception:
                pass
            time.sleep(1.0 / max(float(scene_playback["fps"]), 1.0))
        scene_playback["playing"] = False
        try:
            scene_play_btn.label = "Play"
        except Exception:
            pass
        scene_playback["thread"] = None

    @scene_fps_slider.on_update
    def _on_scene_fps(event: viser.GuiEvent) -> None:
        scene_playback["fps"] = float(event.target.value)

    @scene_loop_cb.on_update
    def _on_scene_loop(event: viser.GuiEvent) -> None:
        scene_playback["loop"] = bool(event.target.value)

    @scene_play_btn.on_click
    def _on_scene_play(event: viser.GuiEvent) -> None:
        if scene_playback["playing"]:
            _stop_scene_playback()
            return
        scene_playback["playing"] = True
        t = threading.Thread(target=_start_scene_playback_loop, daemon=True)
        scene_playback["thread"] = t
        t.start()

    # ── Preview playback state ────────────────────────────────────────────────
    preview_frames: list[np.ndarray] = []   # rendered frames (uint8 [H,W,3])
    playback = {
        "playing":   False,
        "frame_idx": 0,
        "fps":       10.0,
        "thread":    None,   # background playback Thread | None
    }

    def _set_status(msg: str) -> None:
        status_md.content = f"_{msg}_"

    editor.set_gizmos_visible(False)

    @show_gizmos_cb.on_update
    def _on_show_gizmos(event: viser.GuiEvent) -> None:
        state["show_gizmos"] = bool(event.target.value)
        editor.set_gizmos_visible(state["show_gizmos"])

    @btn_add.on_click
    def _on_add(event: viser.GuiEvent) -> None:
        client = event.client
        if client is None:
            _set_status("No browser client connected yet.")
            return

        c2w_opencv = _display_camera_pose_to_opencv(client.camera)

        editor.add_keyframe(c2w_opencv)
        editor.set_gizmos_visible(state["show_gizmos"])
        n = editor.n_keyframes
        kf_count.value = str(n)
        if state["show_gizmos"]:
            _set_status(f"Keyframe {n} added. Drag gizmo to edit, or click Export.")
        else:
            _set_status(
                f"Keyframe {n} added. Enable 'Show Keyframe Gizmos' to edit poses."
            )

    @btn_remove.on_click
    def _on_remove(event: viser.GuiEvent) -> None:
        removed = editor.remove_last_keyframe()
        n = editor.n_keyframes
        kf_count.value = str(n)
        if removed:
            _set_status(f"Removed last keyframe. {n} remaining.")
        else:
            _set_status("No keyframes to remove.")

    @btn_clear.on_click
    def _on_clear(event: viser.GuiEvent) -> None:
        editor.clear_all()
        kf_count.value = "0"
        _set_status("All keyframes cleared.")

    @btn_apply_preset.on_click
    def _on_apply_preset(event: viser.GuiEvent) -> None:
        preset_name = preset_dd.value
        scene_center_opencv = np.array(
            [scene_center_viser[0], -scene_center_viser[1], -scene_center_viser[2]],
            dtype=np.float32,
        )
        try:
            keyframes = generate_preset(preset_name, c2w_0, scene_center_opencv)
        except ValueError as exc:
            _set_status(f"Preset error: {exc}")
            return
        editor.clear_all()
        for kf in keyframes:
            editor.add_keyframe(kf)
        editor.set_gizmos_visible(state["show_gizmos"])
        kf_count.value = str(editor.n_keyframes)
        _set_status(f"Applied '{preset_name}' — {len(keyframes)} keyframes. Adjust gizmos or export.")

    @btn_undo.on_click
    def _on_undo(event: viser.GuiEvent) -> None:
        if editor.undo():
            editor.set_gizmos_visible(state["show_gizmos"])
            kf_count.value = str(editor.n_keyframes)
            _set_status(f"Undo — {editor.n_keyframes} keyframes.")
        else:
            _set_status("Nothing to undo.")

    @btn_redo.on_click
    def _on_redo(event: viser.GuiEvent) -> None:
        if editor.redo():
            editor.set_gizmos_visible(state["show_gizmos"])
            kf_count.value = str(editor.n_keyframes)
            _set_status(f"Redo — {editor.n_keyframes} keyframes.")
        else:
            _set_status("Nothing to redo.")

    @nframe_num.on_update
    def _on_nframe(event: viser.GuiEvent) -> None:
        raw = int(event.target.value)
        if is_valid_wan_frame_count(raw):
            n_render = render_frames_for_wan_output(raw)
            render_frame_info.value = str(n_render)
            state["n_output"] = n_render
        else:
            snapped = snap_to_valid_wan_output(raw)
            render_frame_info.value = f"(type {snapped} or {snapped+4})"
            state["n_output"] = render_frames_for_wan_output(snapped)

    @easing_dd.on_update
    def _on_easing(event: viser.GuiEvent) -> None:
        editor.set_easing_mode(event.target.value)

    @btn_preview.on_click
    def _on_preview(event: viser.GuiEvent) -> None:
        n = state["n_output"]
        path = editor.get_interpolated_path(n)
        client = event.client

        if path is None:
            _set_status("Need at least 2 keyframes to preview.")
            if client is not None:
                client.add_notification(
                    title="No path yet",
                    body="Place at least 2 camera keyframes first.",
                    color="red",
                    auto_close_seconds=3.0,
                )
            return

        # Force-redraw the path visualisation with the current output frame count
        editor.refresh_path(n)

        msg = (
            f"{editor.n_keyframes} keyframes → {n} output frames interpolated. "
            "Orange frustums show the trajectory."
        )
        _set_status(msg)
        if client is not None:
            client.add_notification(
                title="Path previewed",
                body=f"{editor.n_keyframes} keyframes → {n} output frames.",
                color="green",
                auto_close_seconds=4.0,
            )

    @btn_export.on_click
    def _on_export(event: viser.GuiEvent) -> None:
        n = state["n_output"]
        path = editor.get_interpolated_path(n)
        client = event.client

        if path is None:
            _set_status("Need at least 2 keyframes to export.")
            return

        w2c_targets = np.stack(
            [c2w_to_w2c(_keyframe_c2w_to_export_opencv(c2w)) for c2w in path], axis=0
        )
        out = output_path_txt.value.strip() or args.output

        try:
            export_cam_info(
                w2c_source=w2c_0,
                w2c_targets=w2c_targets,
                K=K,
                H=H,
                W=W,
                output_path=out,
            )
            wan_out = wan_consumed_frames(n)
            _set_status(f"Exported {n} render frames ({wan_out} WAN output) → `{out}`")
            if client is not None:
                client.add_notification(
                    title="Exported",
                    body=f"{n} render frames → {wan_out} WAN output frames\n{out}",
                    color="blue",
                    auto_close_seconds=6.0,
                )
        except Exception as exc:
            _set_status(f"Export error: {exc}")
            print(f"[export error] {exc}")

    render_assets_job = {"running": False, "thread": None}

    @btn_render_assets.on_click
    def _on_render_assets(event: viser.GuiEvent) -> None:
        client = event.client
        if render_assets_job["running"]:
            _set_status("Render job already running.")
            return

        n = state["n_output"]
        path = editor.get_interpolated_path(n)
        if path is None:
            _set_status("Need at least 2 keyframes to render assets.")
            if client is not None:
                client.add_notification(
                    title="No path yet",
                    body="Place at least 2 camera keyframes first.",
                    color="red",
                    auto_close_seconds=3.0,
                )
            return

        out_dir = render_out_dir_txt.value.strip()
        if not out_dir:
            _set_status("Render output dir cannot be empty.")
            return
        out_dir_path = Path(out_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)
        cam_info_path = str(out_dir_path / "cam_info.json")

        try:
            w2c_targets = np.stack(
                [c2w_to_w2c(_keyframe_c2w_to_export_opencv(c2w)) for c2w in path], axis=0
            )
            # Debug: check target camera z-values
            export_cam_info(
                w2c_source=w2c_0,
                w2c_targets=w2c_targets,
                K=K,
                H=H,
                W=W,
                output_path=cam_info_path,
            )
        except Exception as exc:
            _set_status(f"Prep render export failed: {exc}")
            print(f"[render prep error] {exc}")
            return

        params = {
            "video": args.video,
            "depth": args.depth,
            "cam_info": cam_info_path,
            "output_dir": str(out_dir_path),
            "max_frames": int(args.max_frames),
            "target_fps": float(args.target_fps) if args.target_fps is not None else None,
            "fps": 0.0,
            "subsample": int(render_subsample_num.value),
            "backend": str(render_backend_dd.value),
            "device": str(render_device_txt.value).strip() or "cuda",
            "radius": float(render_radius_num.value),
            "points_per_pixel": int(render_ppp_num.value),
            "sobel_threshold": float(render_sobel_num.value),
            "nb_neighbors": int(render_nb_neighbors_num.value),
            "std_ratio": float(render_std_ratio_num.value),
            "backgrounds": str(render_bg_txt.value).strip() or "black,pink",
        }
        render_python = str(render_python_txt.value).strip()

        def _render_status(msg: str) -> None:
            render_status_md.content = f"_{msg}_"

        def _worker() -> None:
            try:
                _render_status(f"Rendering assets to `{params['output_dir']}` …")
                if client is not None:
                    client.add_notification(
                        title="Render Started",
                        body=f"Rendering videos to:\n{params['output_dir']}",
                        color="teal",
                        auto_close_seconds=4.0,
                    )
                if render_python:
                    script_path = str((Path(__file__).resolve().parent.parent / "render_from_cam_info.py"))
                    cmd = [
                        render_python,
                        script_path,
                        "--video", params["video"],
                        "--depth", params["depth"],
                        "--cam-info", params["cam_info"],
                        "--output-dir", params["output_dir"],
                        "--max-frames", str(params["max_frames"]),
                        "--backend", str(params["backend"]),
                        "--device", str(params["device"]),
                        "--subsample", str(params["subsample"]),
                        "--radius", str(params["radius"]),
                        "--points-per-pixel", str(params["points_per_pixel"]),
                        "--sobel-threshold", str(params["sobel_threshold"]),
                        "--nb-neighbors", str(params["nb_neighbors"]),
                        "--std-ratio", str(params["std_ratio"]),
                        "--backgrounds", str(params["backgrounds"]),
                    ]
                    if params["target_fps"] is not None:
                        cmd.extend(["--target-fps", str(params["target_fps"])])
                    print("[render assets] subprocess:", " ".join(cmd))
                    proc = subprocess.run(
                        cmd,
                        cwd=str(Path(__file__).resolve().parent.parent),
                        capture_output=True,
                        text=True,
                    )
                    if proc.stdout:
                        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
                    if proc.stderr:
                        print(proc.stderr, end="" if proc.stderr.endswith("\n") else "\n")
                    if proc.returncode != 0:
                        raise RuntimeError(
                            f"Renderer subprocess failed (exit {proc.returncode}). "
                            "See terminal logs for details."
                        )
                    outputs = {
                        "render": str(Path(params["output_dir"]) / "render.mp4"),
                        "render_mask": str(Path(params["output_dir"]) / "render_mask.mp4"),
                    }
                else:
                    from render_from_cam_info import render_assets_from_paths

                    def _on_progress(done: int, total: int) -> None:
                        _render_status(f"Rendering frame {done}/{total} …")

                    outputs = render_assets_from_paths(**params, progress_cb=_on_progress)

                from pipeline_spec import validate_condition_pack
                pack_issues = validate_condition_pack(params["output_dir"])
                if pack_issues:
                    warnings = "; ".join(pack_issues)
                    _render_status(f"Render done with warnings: {warnings}")
                    if client is not None:
                        client.add_notification(
                            title="Render Complete (warnings)",
                            body="\n".join(pack_issues),
                            color="yellow",
                            auto_close_seconds=10.0,
                        )
                    return

                _render_status(f"Render complete → `{params['output_dir']}`")
                if client is not None:
                    client.add_notification(
                        title="Render Complete",
                        body=(
                            "Wrote render assets:\n"
                            f"{outputs.get('render', params['output_dir'])}"
                        ),
                        color="green",
                        auto_close_seconds=6.0,
                    )
            except Exception as exc:
                _render_status(f"Render error: {exc}")
                print(f"[render assets error] {exc}")
                if client is not None:
                    client.add_notification(
                        title="Render Failed",
                        body=str(exc),
                        color="red",
                        auto_close_seconds=8.0,
                    )
            finally:
                render_assets_job["running"] = False
                render_assets_job["thread"] = None

        render_assets_job["running"] = True
        t = threading.Thread(target=_worker, daemon=True)
        render_assets_job["thread"] = t
        t.start()

    # ── Inference handlers ────────────────────────────────────────────────────

    def _build_inference_cmd() -> str:
        pack = infer_pack_dir_txt.value.strip()
        n_gpus = int(infer_num_gpus.value)
        parts = [
            f"CUDA_VISIBLE_DEVICES={','.join(str(i) for i in range(n_gpus))}",
            f"{sys.executable} -m torch.distributed.run --master-port=29501 --nproc_per_node={n_gpus}",
            "inference_wan22_v2v_local.py",
            "--task=i2v-A14B",
            f"--size={infer_size_txt.value.strip()}",
            "--dit_fsdp --t5_fsdp",
            f"--ulysses_size={n_gpus}",
            f"--sample_steps={int(infer_steps_num.value)}",
            f'--save_folder="{infer_save_folder_txt.value.strip()}"',
            "--sample_solver=unipc --sample_shift=5",
            "--lora_alpha=512 --lora_rank=512",
            f'--base_folder="{pack}"',
            '--video_path="render.mp4"',
            '--mask_path="render_mask.mp4"',
            '--ref_path="input.mp4"',
            '--mask_pink_path="render_pink.mp4"',
            f'--caption="{infer_caption_txt.value.strip()}"',
            f'--video_id="output_{int(__import__("time").time())}.mp4"',
        ]
        ckpt = infer_ckpt_dir_txt.value.strip()
        if ckpt:
            parts.insert(5, f'--ckpt_dir="{ckpt}"')
        high_lora = infer_high_lora_txt.value.strip()
        if high_lora:
            parts.append(f'--high_noise_lora_weights="{high_lora}"')
        low_lora = infer_low_lora_txt.value.strip()
        if low_lora:
            parts.append(f'--low_noise_lora_weights="{low_lora}"')
        return " \\\n    ".join(parts)

    @btn_gen_cmd.on_click
    def _on_gen_cmd(event: viser.GuiEvent) -> None:
        cmd = _build_inference_cmd()
        infer_cmd_md.content = f"```bash\n{cmd}\n```"

    @btn_launch_infer.on_click
    def _on_launch_infer(event: viser.GuiEvent) -> None:
        pack = infer_pack_dir_txt.value.strip()
        if not pack or not Path(pack).exists():
            infer_status_md.content = "_Condition pack dir does not exist._"
            return

        if not infer_high_lora_txt.value.strip() or not infer_low_lora_txt.value.strip():
            infer_status_md.content = (
                "_LoRA weights required. Set both High-noise and Low-noise LoRA paths "
                "(local .safetensors or hf:// URI)._"
            )
            return

        from pipeline_spec import validate_condition_pack
        issues = validate_condition_pack(pack)
        if any("Missing" in i for i in issues):
            infer_status_md.content = f"_Condition pack incomplete: {'; '.join(issues)}_"
            return

        cmd_str = _build_inference_cmd()
        infer_cmd_md.content = f"```bash\n{cmd_str}\n```"
        infer_status_md.content = "_Launching inference…_"

        def _infer_worker() -> None:
            import time as _time
            try:
                cmd_flat = cmd_str.replace("\\\n    ", " ")
                print(f"[inference] Running: {cmd_flat}", flush=True)
                infer_status_md.content = "_Starting subprocess…_"
                proc = subprocess.Popen(
                    cmd_flat,
                    shell=True,
                    cwd=str(Path(__file__).resolve().parent.parent),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                last_update = 0.0
                _SKIP_PREFIXES = ("http request", "cas::", "xet", '"timestamp"', "resolve-cache", "reconstruct")
                try:
                    for line in proc.stdout:
                        line = line.rstrip()
                        if not line:
                            continue
                        print(f"[inference] {line}", flush=True)
                        low = line.lower()
                        is_error = "error" in low or "traceback" in low or "exception" in low
                        is_noise = not is_error and any(k in low for k in _SKIP_PREFIXES)
                        if is_noise:
                            continue
                        now = _time.monotonic()
                        if is_error or "%" in line or now - last_update > 3.0:
                            clean = line.strip().lstrip("[").split("]", 1)[-1].strip()
                            infer_status_md.content = f"_{clean[-120:]}_"
                            last_update = now
                except (BrokenPipeError, OSError):
                    pass
                rc = proc.wait()
                if rc == 0:
                    infer_status_md.content = "_Inference complete!_"
                    if event.client is not None:
                        event.client.add_notification(
                            title="Inference Complete",
                            body=f"Output saved to {infer_save_folder_txt.value.strip()}",
                            color="green",
                        )
                else:
                    msg = f"Inference failed (exit {rc}). Check terminal."
                    infer_status_md.content = f"_{msg}_"
                    print(f"[inference] {msg}", flush=True)
            except Exception as exc:
                msg = f"Inference error: {exc}"
                infer_status_md.content = f"_{msg}_"
                print(f"[inference] {msg}", flush=True)

        threading.Thread(target=_infer_worker, daemon=True).start()

    # ── Playback thread helpers ───────────────────────────────────────────────
    def _stop_playback() -> None:
        playback["playing"] = False
        t = playback["thread"]
        if t is not None and t.is_alive():
            t.join(timeout=1.0)
        playback["thread"] = None

    def _start_playback_loop(modal_image_handle, modal_slider_handle, modal_play_btn) -> None:
        """Background thread: advance frames at playback["fps"] until stopped."""
        while playback["playing"] and preview_frames:
            idx = (playback["frame_idx"] + 1) % len(preview_frames)
            playback["frame_idx"] = idx
            modal_image_handle.image = preview_frames[idx]
            modal_slider_handle.value = idx
            modal_play_btn.label = f"⏸ Pause  (frame {idx+1}/{len(preview_frames)})"
            time.sleep(1.0 / max(playback["fps"], 1.0))
        # Reset label when loop ends
        try:
            modal_play_btn.label = "▶ Play"
        except Exception:
            pass

    # ── Render preview callback ───────────────────────────────────────────────
    @btn_render_preview.on_click
    def _on_render_preview(event: viser.GuiEvent) -> None:
        client = event.client
        if client is None:
            _set_status("No browser client connected yet.")
            return

        path = editor.get_interpolated_path(state["n_output"])
        if path is None:
            _set_status("Need at least 2 keyframes to render preview.")
            client.add_notification(
                title="No path yet",
                body="Place at least 2 camera keyframes first.",
                color="red",
                auto_close_seconds=3.0,
            )
            return

        _stop_playback()
        _stop_scene_playback()

        n_out   = len(path)
        # Render at 3/4 resolution for quality; cap at 30 frames
        render_h = max(H * 3 // 4, 270)
        render_w = max(W * 3 // 4, 480)
        max_preview_frames = 30
        indices  = np.linspace(0, n_out - 1, min(n_out, max_preview_frames), dtype=int)

        _set_status(f"Rendering {len(indices)} frames at {render_w}x{render_h}…")
        preview_info.content = f"_Rendering {len(indices)} frames — please wait…_"

        rendered: list[np.ndarray] = []
        fov_rad = float(np.radians(fov_deg))

        # Save current scene visibility state to restore after rendering
        was_frame = state["frame_idx"]
        was_show_all = state["show_all"]
        was_guide_visible = state["show_guide"]

        from .camera_editor import _rotation_to_wxyz as _r2wxyz
        editor.set_scene_overlays_visible(False)
        editor.set_gizmos_visible(False)
        _cancel_settle_timer(client.client_id)
        _set_viewfinder_visible(client, False)
        try:
            for i, fi in enumerate(indices):
                # Map path frame index → corresponding video frame index
                video_fi = int(round(fi / max(n_out - 1, 1) * (T - 1)))
                video_fi = int(np.clip(video_fi, 0, T - 1))

                # Load this video frame's point cloud and show it exclusively
                _ensure_pc(video_fi)
                for j, h in enumerate(pc_handles):
                    if h is not None:
                        h.visible = (j == video_fi)

                # Brief pause so the browser receives the visibility update
                time.sleep(0.05)

                c2w_vis = _opencv_pose_to_display_viser(path[fi])
                img = client.get_render(
                    height=render_h,
                    width=render_w,
                    wxyz=_r2wxyz(c2w_vis[:3, :3]),
                    position=tuple(c2w_vis[:3, 3].tolist()),
                    fov=fov_rad,
                    transport_format="jpeg",
                )
                rendered.append(img)
                preview_thumb.image = img
                preview_thumb.visible = True
                preview_info.content = f"_Rendering frame {i+1}/{len(indices)}…_"
        finally:
            state["show_all"] = was_show_all
            editor.set_scene_overlays_visible(True)
            editor.set_gizmos_visible(state["show_gizmos"])
            # Camera is stationary after rendering — show guide directly (no flicker).
            if was_guide_visible:
                _sync_viewfinder_overlay(client)
            else:
                _set_viewfinder_visible(client, False)
            _show_frame(was_frame)

        preview_frames.clear()
        preview_frames.extend(rendered)
        playback["frame_idx"] = 0

        n_rendered = len(preview_frames)
        preview_thumb.image = preview_frames[0]
        preview_info.content = (
            f"**{n_rendered} frames** at {render_w}×{render_h}. "
            "Click **Open Preview Window** to play."
        )
        btn_open_preview.visible = True

        _set_status(f"Preview ready — {n_rendered} frames rendered.")
        client.add_notification(
            title="Preview ready",
            body="Click 'Open Preview Window' to play the animation.",
            color="teal",
            auto_close_seconds=4.0,
        )

    # ── Modal preview window ──────────────────────────────────────────────────
    @btn_open_preview.on_click
    def _on_open_preview(event: viser.GuiEvent) -> None:
        client = event.client
        if client is None or not preview_frames:
            return

        _stop_playback()
        n_frames = len(preview_frames)
        render_h, render_w = preview_frames[0].shape[:2]

        with client.gui.add_modal(f"Preview — {n_frames} frames  ({render_w}×{render_h})") as modal:
            close_btn  = client.gui.add_button("Close", color="gray")
            modal_img  = client.gui.add_image(preview_frames[0], label=None)
            modal_sl   = client.gui.add_slider(
                "Frame", min=0, max=n_frames - 1, step=1, initial_value=0
            )
            with client.gui.add_folder("Playback"):
                modal_fps  = client.gui.add_slider(
                    "FPS", min=1, max=30, step=1, initial_value=int(playback["fps"])
                )
                modal_play = client.gui.add_button("Play", color="green")

            @close_btn.on_click
            def _on_close(ev: viser.GuiEvent) -> None:
                _stop_playback()
                modal.close()

            @modal_sl.on_update
            def _on_modal_sl(ev: viser.GuiEvent) -> None:
                idx = int(ev.target.value)
                playback["frame_idx"] = idx
                modal_img.image = preview_frames[idx]

            @modal_fps.on_update
            def _on_modal_fps(ev: viser.GuiEvent) -> None:
                playback["fps"] = float(ev.target.value)

            @modal_play.on_click
            def _on_modal_play(ev: viser.GuiEvent) -> None:
                if playback["playing"]:
                    _stop_playback()
                    modal_play.label = "Play"
                    return
                playback["playing"] = True
                t = threading.Thread(
                    target=_start_playback_loop,
                    args=(modal_img, modal_sl, modal_play),
                    daemon=True,
                )
                playback["thread"] = t
                t.start()

    # ── Keep alive ────────────────────────────────────────────────────────────
    print("[app] Press Ctrl-C to quit.\n")
    try:
        while not _shutdown_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass

    new_video = _shutdown_event.is_set()
    reason = "new video requested" if new_video else "Ctrl-C"
    print(f"\n[app] Shutting down ({reason}).")
    _stop_scene_playback()
    _stop_playback()
    for _t in list(_settle_timers.values()):
        _t.cancel()
    _settle_timers.clear()
    server.stop()

    # If the user clicked "Load New Video", serve a redirect page on the same
    # port so the browser is sent to the upload page automatically.
    if new_video:
        time.sleep(1.0)  # wait for viser to release the port
        upload_url = f"http://localhost:{args.port + 1}/upload"
        _serve_redirect(args.port, upload_url, timeout=15.0)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="V2V 3D Camera Path Visualizer — place camera keyframes on a video point cloud."
    )
    p.add_argument("--video",  default=None, help="Input video (.mp4)")
    p.add_argument("--depth",  default=None, help="Metric depth maps (.npz with 'depths' key)")
    p.add_argument("--output", default="cam_info.json", help="Output JSON path (default: cam_info.json)")
    p.add_argument("--nframe", default=81, type=int,
                   help="Desired WAN output frames (snapped to nearest 4k+1; default: 81)")
    p.add_argument("--focal",  default=1.0, type=float, help="Focal length multiplier (default: 1.0)")
    p.add_argument("--start-elevation", dest="start_elevation", default=5.0, type=float,
                   help="Source camera elevation in degrees (default: 5.0)")
    p.add_argument("--port",  default=8080, type=int, help="Viser server port (default: 8080)")
    p.add_argument("--subsample", default=4, type=int,
                   help="Point cloud subsampling factor — keep every N-th pixel (default: 4)")
    p.add_argument("--point-size", dest="point_size", default=0.005, type=float,
                   help="Point size in world units (default: 0.005)")
    p.add_argument("--max-frames", dest="max_frames", default=200, type=int,
                   help="Maximum number of video frames to load (default: 200)")
    p.add_argument("--target-fps", dest="target_fps", default=None, type=float,
                   help="Downsample video to this FPS before loading (default: use all frames)")
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.video:
        parser.error("--video is required (or run via run_visualizer.sh for default example)")
    if not args.depth:
        parser.error("--depth is required (or run via run_visualizer.sh for default example)")

    if not Path(args.video).exists():
        print(f"[error] Video not found: {args.video}", file=sys.stderr)
        sys.exit(1)
    if not Path(args.depth).exists():
        print(f"[error] Depth file not found: {args.depth}", file=sys.stderr)
        sys.exit(1)

    run(args)


if __name__ == "__main__":
    main()
