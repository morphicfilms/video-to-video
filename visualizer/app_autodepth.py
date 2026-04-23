# Copyright 2025-2026 Morphic Inc. Licensed under Apache 2.0.
"""
app_autodepth.py — Viser visualizer launcher with automatic depth estimation.

Modes:
1) CLI video input:
    python -m visualizer.app_autodepth --video input.mp4
   (depth is auto-estimated if not provided)

2) UI launcher mode:
    python -m visualizer.app_autodepth
   Open the browser UI, then either:
     - set a remote path (on the machine running the app), or
     - upload a local video file from the browser.
   The app estimates depth and then launches the existing point-cloud visualizer.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import select
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlparse

from .app import run as run_visualizer


def _sanitize_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return safe or f"video_{int(time.time())}.mp4"


def _video_cache_key(video_path: str, method: str = "", max_res: int = 0, steps: int = 0) -> str:
    """
    Stable key for a specific video file + depth estimation config.

    Uses resolved path + file size + mtime ns + estimation params so we
    invalidate the cache when either the video or the estimation config changes.
    """
    p = Path(video_path).expanduser().resolve()
    st = p.stat()
    raw = f"{p}:{st.st_size}:{st.st_mtime_ns}:{method}:{max_res}:{steps}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _default_depth_output(
    video_path: str, depth_cache_dir: str,
    method: str = "", max_res: int = 0, steps: int = 0,
) -> Path:
    video = Path(video_path)
    stem = _sanitize_filename(video.stem)
    key = _video_cache_key(video_path, method=method, max_res=max_res, steps=steps)
    out_dir = Path(depth_cache_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{stem}_{key}_depths.npz"


def _cached_video_path(depth_path: Path) -> Path:
    """Return the companion cached-video path for a given depth cache file."""
    name = depth_path.name
    if name.endswith("_depths.npz"):
        return depth_path.parent / (name[: -len("_depths.npz")] + "_input.mp4")
    return depth_path.with_suffix(".mp4")


def _cache_input_video(video_path: str, depth_path: str) -> None:
    """Copy the source video into the cache directory alongside its depth file."""
    cached = _cached_video_path(Path(depth_path))
    if not cached.exists():
        shutil.copy2(video_path, cached)
        print(f"[launcher] Cached input video → {cached}")


def _estimate_depth(
    *,
    video_path: str,
    depth_out: str,
    args: argparse.Namespace,
    status_cb: Callable[[str], None] | None = None,
) -> None:
    def _status(msg: str) -> None:
        if status_cb is not None:
            status_cb(msg)  # status_cb handles printing
        else:
            print(msg)

    estimate_script = (Path(__file__).resolve().parent.parent / "estimate_depth.py").resolve()
    if not estimate_script.exists():
        raise FileNotFoundError(f"estimate_depth.py not found at {estimate_script}")

    depth_method = getattr(args, "depth_method", "depthcrafter")
    cmd = [
        sys.executable,
        "-u",
        str(estimate_script),
        "--video",
        str(video_path),
        "--output",
        str(depth_out),
        "--method",
        str(depth_method),
        "--dc-dir",
        str(args.dc_dir),
        "--gc-dir",
        str(args.gc_dir),
        "--max-res",
        str(args.depth_max_res),
        "--max-frames",
        str(args.depth_max_frames),
        "--steps",
        str(args.depth_steps),
        "--window-size",
        str(args.depth_window_size),
        "--overlap",
        str(args.depth_overlap),
        "--device",
        str(args.depth_device),
    ]
    if args.gc_cache_dir:
        cmd.extend(["--cache-dir", str(args.gc_cache_dir)])
    if args.depth_focal is not None:
        cmd.extend(["--focal", str(args.depth_focal)])

    _status("[launcher] Estimating depth...")
    _status(f"[launcher] Command: {' '.join(cmd)}")
    try:
        start_ts = time.time()
        last_heartbeat_sec = -1
        last_line = ""

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert proc.stdout is not None
        while True:
            # Non-blocking-ish stdout read + periodic heartbeat.
            ready, _, _ = select.select([proc.stdout], [], [], 1.0)
            if ready:
                line = proc.stdout.readline()
                if line:
                    msg = line.rstrip()
                    if msg:
                        last_line = msg
                        _status(msg)

            rc = proc.poll()
            if rc is not None:
                # Drain any remaining output.
                for line in proc.stdout:
                    msg = line.rstrip()
                    if msg:
                        last_line = msg
                        _status(msg)
                if rc != 0:
                    raise subprocess.CalledProcessError(rc, cmd)
                break

            elapsed = int(time.time() - start_ts)
            if elapsed >= 5 and elapsed - last_heartbeat_sec >= 5:
                hb = f"[launcher] Depth estimation running... {elapsed}s elapsed"
                if last_line:
                    hb += f" | last: {last_line[:120]}"
                _status(hb)
                last_heartbeat_sec = elapsed
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Depth estimation failed. Ensure depth dependencies are installed "
            "(run: `uv sync --group depth`) and check logs above for the root cause."
        ) from exc
    _status(f"[launcher] Depth ready: {depth_out}")


def _prepare_depth(
    *,
    video_path: str,
    depth_path: str | None,
    args: argparse.Namespace,
    status_cb: Callable[[str], None] | None = None,
) -> str:
    out = Path(depth_path).expanduser().resolve() if depth_path else _default_depth_output(
        video_path, args.depth_cache_dir,
        method=getattr(args, "depth_method", ""),
        max_res=getattr(args, "depth_max_res", 0),
        steps=getattr(args, "depth_steps", 0),
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists() and args.reuse_depth:
        msg = f"[launcher] Reusing existing depth: {out}"
        print(msg)
        if status_cb is not None:
            status_cb(msg)
    else:
        if not args.gc_dir or not Path(args.gc_dir).is_dir():
            raise ValueError(
                f"GeometryCrafter not found at '{args.gc_dir}'. "
                "Run ./install.sh to clone it automatically, or pass --gc-dir /path/to/GeometryCrafter."
            )
        _estimate_depth(video_path=video_path, depth_out=str(out), args=args, status_cb=status_cb)
        if not out.exists():
            raise FileNotFoundError(f"Depth estimation finished but output not found: {out}")

    # Always keep a copy of the source video next to the depth file so the
    # cache is self-contained (video + depth together).
    _cache_input_video(video_path, str(out))

    return str(out)


def _build_core_args(wrapper_args: argparse.Namespace, video_path: str, depth_path: str) -> argparse.Namespace:
    return argparse.Namespace(
        video=video_path,
        depth=depth_path,
        output=wrapper_args.output,
        nframe=wrapper_args.nframe,
        focal=wrapper_args.focal,
        start_elevation=wrapper_args.start_elevation,
        port=wrapper_args.port,
        subsample=wrapper_args.subsample,
        point_size=wrapper_args.point_size,
        max_frames=wrapper_args.max_frames,
        target_fps=wrapper_args.target_fps,
    )


def _launch_visualizer_with_video(
    *,
    video_path: str,
    depth_path: str | None,
    args: argparse.Namespace,
    status_cb: Callable[[str], None] | None = None,
) -> None:
    video_resolved = str(Path(video_path).expanduser().resolve())
    if not Path(video_resolved).exists():
        raise FileNotFoundError(f"Video not found: {video_resolved}")

    resolved_depth = _prepare_depth(
        video_path=video_resolved,
        depth_path=depth_path,
        args=args,
        status_cb=status_cb,
    )
    core_args = _build_core_args(args, video_resolved, resolved_depth)
    run_visualizer(core_args)


_HTML_UPLOAD_PAGE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>V2V Video Upload</title>
  <style>
    body  { font-family: system-ui, sans-serif; background: #1a1a2e; color: #eee;
            max-width: 560px; margin: 60px auto; padding: 24px; }
    h2   { margin-bottom: 4px; }
    p    { margin: 4px 0 16px; color: #aaa; font-size: .95em; }
    .row { display: flex; align-items: center; gap: 12px; margin: 10px 0; }
    label { min-width: 88px; color: #aaa; font-size: .9em; }
    select, input[type=file] { background: #2a2a3e; color: #eee;
      border: 1px solid #444; border-radius: 6px; padding: 6px 10px; cursor: pointer; }
    select:focus { outline: 1px solid #7c8cf8; }
    button { background: #4a6cf7; color: #fff; border: none; border-radius: 6px;
             padding: 10px 22px; cursor: pointer; font-size: .95em; margin-top: 8px; }
    button:disabled { background: #444; cursor: default; }
    button:hover:not(:disabled) { background: #3a5ce7; }
    #status { margin-top: 18px; padding: 12px 16px; background: #2a2a3e;
              border-radius: 6px; font-size: .9em; line-height: 1.5; min-height: 2.5em; }
    progress { display: block; width: 100%; height: 8px; margin-top: 10px;
               border-radius: 4px; appearance: none; }
    progress::-webkit-progress-bar   { background: #3a3a4e; border-radius: 4px; }
    progress::-webkit-progress-value { background: #4a6cf7; border-radius: 4px; }
    .done { color: #4ddb85; font-weight: 600; }
    .err  { color: #f76a6a; }
    #cached { margin-top: 28px; }
    #cached h3 { font-size: .95em; color: #aaa; margin: 0 0 10px; font-weight: 500; }
    .cv { display: flex; align-items: center; gap: 10px; padding: 8px 12px;
          background: #2a2a3e; border-radius: 6px; margin: 6px 0; }
    .cv-info { flex: 1; min-width: 0; }
    .cv-name { display: block; font-size: .88em; white-space: nowrap;
               overflow: hidden; text-overflow: ellipsis; }
    .cv-meta { font-size: .78em; color: #888; margin-top: 2px; }
    .cv-depth { color: #4ddb85; }
    .cv-btn { background: #2e4a9e; padding: 5px 13px; font-size: .85em;
              margin-top: 0; flex-shrink: 0; }
    .cv-btn:hover:not(:disabled) { background: #1e3a8e; }
  </style>
</head>
<body>
  <h2>V2V Video Upload</h2>
  <p>Clips are trimmed to <strong>5 s</strong> at <strong>16 fps</strong>
     and downsampled <em>in your browser</em> before uploading.</p>
  <div class="row">
    <label>Video file</label>
    <input type="file" id="fileInput" accept="video/*">
  </div>
  <div class="row">
    <label>Resolution</label>
    <select id="quality">
      <option value="480">480p</option>
      <option value="720">720p</option>
    </select>
  </div>
  <button id="processBtn" disabled>Process &amp; Upload</button>
  <div id="status">Select a video file to begin.</div>
  <progress id="progress" value="0" max="100" style="display:none"></progress>
  <div id="cached" style="display:none">
    <h3>Cached videos</h3>
    <div id="cachedList"></div>
  </div>

  <script>
  const fileInput  = document.getElementById('fileInput');
  const processBtn = document.getElementById('processBtn');
  const qualitySel = document.getElementById('quality');
  const statusEl   = document.getElementById('status');
  const progressEl = document.getElementById('progress');

  function setStatus(msg, cls) {
    statusEl.textContent = msg;
    statusEl.className = cls || '';
  }
  function fmtSize(b) {
    if (b < 1024)        return b + ' B';
    if (b < 1024*1024)   return (b/1024).toFixed(0) + ' KB';
    return (b/1024/1024).toFixed(1) + ' MB';
  }

  fileInput.addEventListener('change', () => {
    processBtn.disabled = !fileInput.files.length;
    if (fileInput.files.length)
      setStatus('Ready: ' + fileInput.files[0].name + ' (' + fmtSize(fileInput.files[0].size) + ')');
  });

  processBtn.addEventListener('click', async () => {
    const file = fileInput.files[0];
    if (!file) return;
    processBtn.disabled = true;
    progressEl.style.display = 'block';
    progressEl.value = 0;
    try {
      const targetH = parseInt(qualitySel.value);
      setStatus('Loading video\\u2026');
      const blob = await processVideo(file, targetH);
      setStatus('Uploading ' + fmtSize(blob.size) + '\\u2026');
      await uploadBlob(blob, file.name);
      progressEl.style.display = 'none';
      setStatus('Estimating depth\u2026 this may take a few minutes.');
      pollStatus();
    } catch (e) {
      setStatus('Error: ' + e.message, 'err');
      processBtn.disabled = false;
    }
  });

  // ── Video processing (Canvas + MediaRecorder) ──────────────────────────────
  function processVideo(file, targetH) {
    return new Promise((resolve, reject) => {
      const video = document.createElement('video');
      video.muted = true;
      video.playsInline = true;
      const url = URL.createObjectURL(file);
      video.src = url;

      video.onloadedmetadata = () => {
        const srcW = video.videoWidth, srcH = video.videoHeight;
        // Cap short side to targetH (e.g. 480 or 720), maintain aspect, round to even
        let dstW, dstH;
        const shortSide = Math.min(srcW, srcH);
        if (shortSide <= targetH) {
          dstW = srcW;
          dstH = srcH;
        } else if (srcH <= srcW) {
          // Landscape or square: height is the short side
          dstH = targetH;
          dstW = Math.round(targetH * srcW / srcH);
        } else {
          // Portrait: width is the short side
          dstW = targetH;
          dstH = Math.round(targetH * srcH / srcW);
        }
        dstW = Math.ceil(dstW / 2) * 2;
        dstH = Math.ceil(dstH / 2) * 2;

        const canvas = document.createElement('canvas');
        canvas.width  = dstW;
        canvas.height = dstH;
        const ctx = canvas.getContext('2d');

        const FPS = 16, DURATION = 5.0;
        const codecs = [
          'video/webm;codecs=vp9',
          'video/webm;codecs=vp8',
          'video/webm',
        ];
        const mime = codecs.find(c => {
          try { return MediaRecorder.isTypeSupported(c); } catch { return false; }
        }) || 'video/webm';

        const recorder = new MediaRecorder(canvas.captureStream(FPS),
                                           { mimeType: mime, videoBitsPerSecond: 2_500_000 });
        const chunks = [];
        recorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
        recorder.onstop = () => {
          URL.revokeObjectURL(url);
          resolve(new Blob(chunks, { type: mime }));
        };

        let frameTimer = null;
        function stopCapture() {
          if (frameTimer) { clearInterval(frameTimer); frameTimer = null; }
          video.pause();
          ctx.drawImage(video, 0, 0, dstW, dstH);
          setTimeout(() => recorder.stop(), 300);
        }

        video.onseeked = () => {
          video.play().then(() => {
            recorder.start(250);
            const t0 = performance.now();
            frameTimer = setInterval(() => {
              ctx.drawImage(video, 0, 0, dstW, dstH);
              const s = ((performance.now() - t0) / 1000).toFixed(1);
              setStatus('Processing\\u2026 ' + s + 's / ' + DURATION + 's  ('
                        + dstW + '\\xd7' + dstH + ' @ ' + FPS + ' fps)');
            }, 1000 / FPS);
            setTimeout(stopCapture, DURATION * 1000 + 300);
            video.onended = stopCapture;
          }).catch(reject);
        };

        video.onerror = () => reject(new Error('Cannot load video'));
        video.currentTime = 0;
      };

      video.onerror = () => reject(new Error('Cannot load video'));
      video.load();
    });
  }

  // ── Upload ─────────────────────────────────────────────────────────────────
  function uploadBlob(blob, originalName) {
    return new Promise((resolve, reject) => {
      const ext  = blob.type.includes('mp4') ? '.mp4' : '.webm';
      const dot  = originalName.lastIndexOf('.');
      const base = dot >= 0 ? originalName.substring(0, dot) : originalName;
      const fname = base + '_processed' + ext;
      const xhr = new XMLHttpRequest();
      xhr.upload.onprogress = e => {
        if (e.lengthComputable) {
          const pct = Math.round(e.loaded / e.total * 100);
          progressEl.value = pct;
          setStatus('Uploading\\u2026 ' + pct + '%  ('
                    + fmtSize(e.loaded) + ' / ' + fmtSize(e.total) + ')');
        }
      };
      xhr.onload  = () => xhr.status === 200 ? resolve()
                                              : reject(new Error('HTTP ' + xhr.status));
      xhr.onerror = () => reject(new Error('Network error'));
      xhr.open('POST', '/upload?filename=' + encodeURIComponent(fname));
      xhr.setRequestHeader('Content-Type', 'application/octet-stream');
      xhr.send(blob);
    });
  }

  // ── Cached video list ───────────────────────────────────────────────────────
  const _cache = [];   // indexed by position; avoids quote-escaping in onclick

  function loadCached() {
    fetch('/cached')
      .then(r => r.json())
      .then(items => {
        if (!items.length) return;
        _cache.length = 0;
        items.forEach((item, i) => _cache.push(item));
        const section = document.getElementById('cached');
        const list    = document.getElementById('cachedList');
        section.style.display = 'block';
        list.innerHTML = items.map((item, i) => {
          const meta = fmtSize(item.size)
                     + (item.has_depth ? ' \u00b7 <span class="cv-depth">\u2713 depth cached</span>'
                                       : ' \u00b7 needs depth estimation');
          return '<div class="cv">'
               + '<div class="cv-info">'
               + '<span class="cv-name" title="' + item.filename + '">' + item.filename + '</span>'
               + '<div class="cv-meta">' + meta + '</div>'
               + '</div>'
               + '<button class="cv-btn" onclick="useCached(' + i + ')">Use</button>'
               + '</div>';
        }).join('');
      })
      .catch(() => {});
  }

  function useCached(i) {
    const item = _cache[i];
    processBtn.disabled = true;
    progressEl.style.display = 'none';
    setStatus('Starting\u2026 ' + item.filename);
    fetch('/use_cached', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: item.path, filename: item.filename }),
    })
      .then(r => r.ok ? r.json() : Promise.reject(new Error('HTTP ' + r.status)))
      .then(() => pollStatus())
      .catch(e => {
        setStatus('Error: ' + e.message, 'err');
        processBtn.disabled = false;
      });
  }

  loadCached();

  // ── Status polling (runs after upload completes) ────────────────────────────
  function pollStatus() {
    fetch('/status')
      .then(r => r.json())
      .then(data => {
        if (data.error) {
          setStatus('Error: ' + data.error, 'err');
          processBtn.disabled = false;
          return;
        }
        if (data.redirect) {
          setStatus('Done! Opening visualizer\u2026', 'done');
          setTimeout(() => { window.location.href = data.redirect_url; }, 800);
          return;
        }
        setStatus(data.message || 'Processing\u2026');
        setTimeout(pollStatus, 1000);
      })
      .catch(() => setTimeout(pollStatus, 2000));
  }
  </script>
</body>
</html>
"""


def _make_upload_handler(
    upload_dir: str,
    on_file_received: Callable[[str, str], None],
    status_state: dict,
    depth_cache_dir: str,
):
    """Return an HTTPRequestHandler class bound to the given upload dir and callback."""

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            import json

            path = urlparse(self.path).path
            if path in ("", "/", "/upload"):
                body = _HTML_UPLOAD_PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/status":
                body = json.dumps(status_state).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/cached":
                items = []
                ud = Path(upload_dir).expanduser().resolve()
                if ud.is_dir():
                    for f in sorted(ud.iterdir(), key=lambda x: -x.stat().st_mtime):
                        if f.suffix.lower() in (".mp4", ".mov", ".mkv"):
                            has_depth = _default_depth_output(str(f), depth_cache_dir).exists()
                            items.append({
                                "filename": f.name,
                                "path": str(f),
                                "has_depth": has_depth,
                                "size": f.stat().st_size,
                            })
                body = json.dumps(items).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/use_cached":
                try:
                    import json as _json
                    length = int(self.headers.get("Content-Length", 0))
                    data = _json.loads(self.rfile.read(length))
                    video_path = data["path"]
                    filename = data["filename"]
                    if not Path(video_path).is_file():
                        self.send_error(404, "Cached file not found")
                        return
                    on_file_received(video_path, filename)
                    resp = b'{"status":"ok"}'
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(resp)))
                    self.end_headers()
                    self.wfile.write(resp)
                except Exception as exc:
                    self.send_error(500, str(exc))
                return
            if parsed.path != "/upload":
                self.send_error(404)
                return
            params = parse_qs(parsed.query)
            filename = (params.get("filename") or ["upload.webm"])[0]
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                self.send_error(400, "Empty body")
                return
            try:
                content = self.rfile.read(content_length)
                out_dir = Path(upload_dir).expanduser().resolve()
                out_dir.mkdir(parents=True, exist_ok=True)
                ext = Path(filename).suffix.lower()
                if ext not in (".mp4", ".webm", ".mkv", ".mov"):
                    ext = ".webm"
                # Key by sanitized filename so the same source video always maps
                # to the same cache file (MediaRecorder output is non-deterministic,
                # so content hashing would create duplicates on every upload).
                stem = _sanitize_filename(Path(filename).stem)
                out_path = out_dir / f"{stem}{ext}"
                if not out_path.exists():
                    out_path.write_bytes(content)
                    print(f"[launcher] Upload saved → {out_path} ({len(content):,} bytes)")
                else:
                    print(f"[launcher] Upload reused → {out_path}")

                # MediaRecorder webm files lack duration metadata; decord (and many
                # other decoders) reject them.  Remux to mp4 using the bundled ffmpeg.
                if ext == ".webm":
                    mp4_path = out_path.with_suffix(".mp4")
                    if not mp4_path.exists():
                        try:
                            import imageio_ffmpeg
                            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
                            result = subprocess.run(
                                [ffmpeg, "-i", str(out_path),
                                 "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                                 "-an", str(mp4_path), "-y"],
                                capture_output=True, text=True,
                            )
                            if result.returncode == 0 and mp4_path.exists() and mp4_path.stat().st_size > 0:
                                out_path.unlink(missing_ok=True)
                                out_path = mp4_path
                                filename = Path(filename).stem + ".mp4"
                                print(f"[launcher] Converted to mp4 → {out_path}")
                            else:
                                print(f"[launcher] ffmpeg error:\n{result.stderr}")
                        except Exception as conv_exc:
                            print(f"[launcher] Warning: mp4 conversion failed: {conv_exc}")
                    else:
                        out_path.unlink(missing_ok=True)
                        out_path = mp4_path
                        filename = Path(filename).stem + ".mp4"

                on_file_received(str(out_path), filename)
                resp = b'{"status":"ok"}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                self.send_error(500, str(exc))

        def log_message(self, fmt: str, *args: object) -> None:
            pass  # suppress per-request logs

    return _Handler


def _run_ui_launcher(args: argparse.Namespace) -> None:
    upload_port = args.port + 1

    print(f"[launcher] Open http://localhost:{upload_port}/upload in your browser.")
    print(f"[launcher] After depth estimation, the 3D viewer will start on port {args.port}.")

    launch_state: dict = {"video": None, "depth": None, "error": None, "worker_running": False}
    _status_state: dict = {
        "message": "Waiting for upload\u2026",
        "redirect": False,
        "redirect_url": f"http://localhost:{args.port}",
        "error": None,
    }
    launch_event = threading.Event()
    lock = threading.Lock()

    def _set_status(msg: str) -> None:
        _status_state["message"] = msg
        print(msg)

    def _start_with_video(video_path: str) -> None:
        with lock:
            if launch_state["worker_running"]:
                return
            launch_state["worker_running"] = True

        def _worker() -> None:
            try:
                depth_path = _prepare_depth(
                    video_path=video_path,
                    depth_path=str(args.depth) if args.depth else None,
                    args=args,
                    status_cb=_set_status,
                )
                with lock:
                    launch_state["video"] = str(Path(video_path).expanduser().resolve())
                    launch_state["depth"] = depth_path
                    launch_state["error"] = None
                _status_state["redirect"] = True
                _set_status("Starting visualizer\u2026")
                launch_event.set()
            except Exception as exc:
                _status_state["error"] = str(exc)
                with lock:
                    launch_state["error"] = str(exc)
                _set_status(f"Error: {exc}")
            finally:
                with lock:
                    launch_state["worker_running"] = False

        threading.Thread(target=_worker, daemon=True).start()

    def _on_file_received(video_path: str, filename: str) -> None:
        _set_status(f"File received: {filename}. Starting depth estimation\u2026")
        _start_with_video(video_path)

    _handler_cls = _make_upload_handler(
        args.upload_dir, _on_file_received, _status_state, args.depth_cache_dir
    )
    _httpd = HTTPServer(("", upload_port), _handler_cls)
    threading.Thread(target=_httpd.serve_forever, daemon=True).start()

    try:
        while not launch_event.is_set():
            time.sleep(0.25)
        # Keep serving /status long enough for the browser to poll the redirect
        # flag and begin navigation before we tear down the HTTP server.
        time.sleep(4.0)
    finally:
        _httpd.shutdown()
        time.sleep(0.5)

    with lock:
        err = launch_state["error"]
        video = launch_state["video"]
        depth = launch_state["depth"]

    if err:
        raise RuntimeError(err)
    if not video or not depth:
        raise RuntimeError("Launcher finished without selected video/depth.")

    return video, depth


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "V2V visualizer launcher with auto-depth support. "
            "If --video is omitted, a UI launcher opens for video selection/upload."
        )
    )

    # Existing visualizer args
    p.add_argument("--video", default=None, help="Input video path (.mp4). Optional in launcher mode.")
    p.add_argument("--depth", default=None, help="Depth .npz path. If omitted/missing, depth is estimated.")
    p.add_argument("--output", default="cam_info.json", help="Output cam_info.json path")
    p.add_argument("--nframe", default=81, type=int, help="Number of output frames")
    p.add_argument("--focal", default=1.0, type=float, help="Focal multiplier")
    p.add_argument("--start-elevation", dest="start_elevation", default=5.0, type=float, help="Source camera elevation")
    p.add_argument("--port", default=8080, type=int, help="Viser server port")
    p.add_argument("--subsample", default=4, type=int, help="Point cloud subsample factor")
    p.add_argument("--point-size", dest="point_size", default=0.005, type=float, help="Point size")
    p.add_argument("--max-frames", dest="max_frames", default=200, type=int, help="Max frames for visualization")
    p.add_argument("--target-fps", dest="target_fps", default=None, type=float, help="Downsample FPS for visualization")

    # Auto-depth options
    p.add_argument("--reuse-depth", dest="reuse_depth", action="store_true", default=True,
                   help="Reuse existing depth file if present")
    p.add_argument("--no-reuse-depth", dest="reuse_depth", action="store_false",
                   help="Always regenerate depth even if output path exists")
    p.add_argument("--depth-cache-dir", default=".cache/visualizer_depths",
                   help="Directory for auto-generated depth outputs")
    p.add_argument("--upload-dir", default=".cache/visualizer_uploads",
                   help="Where browser-uploaded videos are stored on the server")

    p.add_argument("--depth-method", dest="depth_method", default="depthcrafter",
                   choices=["depthcrafter", "gc_moge"],
                   help="Depth estimation method")
    _default_dc_dir = str(Path(__file__).resolve().parent.parent / ".deps" / "DepthCrafter")
    p.add_argument("--dc-dir", dest="dc_dir", default=_default_dc_dir,
                   help="DepthCrafter repo root (for depthcrafter method)")
    _default_gc_dir = str(Path(__file__).resolve().parent.parent / ".deps" / "GeometryCrafter")
    p.add_argument("--gc-dir", dest="gc_dir", default=_default_gc_dir,
                   help="GeometryCrafter repo root (for gc_moge method)")
    p.add_argument("--gc-cache-dir", dest="gc_cache_dir", default=None,
                   help="Optional model cache dir for GeometryCrafter")
    p.add_argument("--depth-device", dest="depth_device", default="cuda", help="Depth estimation device")
    p.add_argument("--depth-max-res", dest="depth_max_res", default=1024, type=int,
                   help="Depth estimation longest-side cap")
    p.add_argument("--depth-max-frames", dest="depth_max_frames", default=-1, type=int,
                   help="Depth estimation max frames (-1 = all)")
    p.add_argument("--depth-steps", dest="depth_steps", default=5, type=int,
                   help="GeometryCrafter diffusion steps")
    p.add_argument("--depth-window-size", dest="depth_window_size", default=110, type=int,
                   help="GeometryCrafter sliding window size")
    p.add_argument("--depth-overlap", dest="depth_overlap", default=25, type=int,
                   help="GeometryCrafter sliding window overlap")
    p.add_argument("--depth-focal", dest="depth_focal", default=None, type=float,
                   help="Optional fixed focal (pixels) for depth estimation")
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Direct mode (video passed in CLI): launch immediately with auto depth if needed.
    if args.video:
        try:
            _launch_visualizer_with_video(
                video_path=args.video,
                depth_path=args.depth,
                args=args,
            )
        except Exception as exc:
            print(f"[error] {exc}", file=sys.stderr)
            sys.exit(1)
        return

    # No --video: upload loop — after the visualizer closes, go back to upload page.
    while True:
        try:
            video, depth = _run_ui_launcher(args)
        except Exception as exc:
            print(f"[error] {exc}", file=sys.stderr)
            sys.exit(1)

        print(f"[launcher] Launching visualizer with video={video} depth={depth}")
        core_args = _build_core_args(args, video, depth)
        try:
            run_visualizer(core_args)
        except Exception as exc:
            print(f"[error] Visualizer exited with error: {exc}", file=sys.stderr)

        print("[launcher] Visualizer closed. Restarting upload page…")
        print(f"[launcher] Open http://localhost:{args.port + 1}/upload to load a new video.")


if __name__ == "__main__":
    main()
