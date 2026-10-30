"""Convex decomposition (CoACD) of a mesh file, isolated in a child process.

CoACD can abort() in C++ on some meshes; running it in a child keeps the
caller (an export batch, a worker pool) alive. Used by the MuJoCo evaluator
(true-geometry collision) and by the exporter (concave primitives).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def _decompose(mesh_path: str, out_dir: str, threshold: float) -> None:
    import coacd
    import trimesh
    coacd.set_log_level("error")
    mesh = trimesh.load(mesh_path, force="mesh")
    last = None
    for mode in ("auto", "on", "off"):
        try:
            pieces = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces),
                                     threshold=threshold, preprocess_mode=mode)
            break
        except Exception as e:     # noqa: BLE001
            last = e
    else:
        raise RuntimeError(f"coacd failed: {last}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, (v, f) in enumerate(pieces):
        trimesh.Trimesh(v, f).export(out / f"part_{i:03d}.obj")


def decompose_file(mesh_path: str | Path, out_dir: str | Path,
                   threshold: float = 0.05, timeout: float = 600) -> list[Path]:
    """Decompose ``mesh_path`` into ``out_dir/part_<i>.obj`` (single-threaded
    child process; the directory appears atomically). Returns the part paths,
    reusing an existing ``out_dir``. Raises RuntimeError on failure."""
    out_dir = Path(out_dir)
    parts = sorted(out_dir.glob("part_*.obj")) if out_dir.is_dir() else []
    if parts:
        return parts
    tmp = out_dir.with_name(f"{out_dir.name}.tmp{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    env = dict(os.environ, OMP_NUM_THREADS="1")
    r = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                        str(mesh_path), str(tmp), str(threshold)],
                       capture_output=True, env=env, timeout=timeout)
    if r.returncode != 0 or not any(tmp.glob("part_*.obj")):
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError(f"coacd failed (exit {r.returncode})")
    try:
        tmp.rename(out_dir)              # atomic publish
    except OSError:                      # a racing worker won; use its parts
        shutil.rmtree(tmp, ignore_errors=True)
    return sorted(out_dir.glob("part_*.obj"))


if __name__ == "__main__":
    _decompose(sys.argv[1], sys.argv[2], float(sys.argv[3]))
