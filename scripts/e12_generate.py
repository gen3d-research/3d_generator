#!/usr/bin/env python3
"""E12: text to raw mesh with a neural model, for the same prompts per seed. Runs in ~/venv/gen3d_neural,
under `flock ~/.gpu.lock` (torch stays out of the generator's venv). Weights come from the Hugging Face cache.

    flock ~/.gpu.lock python scripts/e12_generate.py --model shap_e --seeds 65 66 --root output/e12
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def shap_e():
    from diffusers import ShapEPipeline
    from diffusers.utils import export_to_ply
    pipe = ShapEPipeline.from_pretrained("openai/shap-e", torch_dtype=torch.float16).to("cuda")

    def run(prompt, seed, out):
        g = torch.Generator("cuda").manual_seed(seed)
        im = pipe(prompt, generator=g, guidance_scale=15.0, num_inference_steps=64,
                  frame_size=64, output_type="mesh").images[0]
        path = str(out.with_suffix(".ply"))
        export_to_ply(im, path)
        return path
    return run


def trellis():
    os.environ.setdefault("SPCONV_ALGO", "native")
    os.environ.setdefault("ATTN_BACKEND", "xformers")
    sys.path.insert(0, str(ROOT / "output/e12/src/TRELLIS"))
    from trellis.pipelines import TrellisTextTo3DPipeline
    import trimesh
    pipe = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-large")
    pipe.cuda()

    def run(prompt, seed, out):
        res = pipe.run(prompt, seed=seed, formats=["mesh"])
        m = res["mesh"][0]
        path = str(out.with_suffix(".ply"))
        trimesh.Trimesh(m.vertices.detach().cpu().numpy(), m.faces.detach().cpu().numpy(),
                        process=False).export(path)
        return path
    return run


def hunyuan():
    sys.path.insert(0, str(ROOT / "output/e12/src/Hunyuan3D-2"))
    from hy3dgen.rembg import BackgroundRemover
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dgen.text2image import HunyuanDiTPipeline
    t2i = HunyuanDiTPipeline("Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled")
    rembg = BackgroundRemover()
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2",
                                                            subfolder="hunyuan3d-dit-v2-0", variant="fp16")

    def run(prompt, seed, out):
        image = rembg(t2i(prompt, seed=seed).convert("RGB"))
        g = torch.Generator("cuda").manual_seed(seed)
        mesh = pipe(image=image, num_inference_steps=30, generator=g, output_type="trimesh")[0]
        path = str(out.with_suffix(".ply"))
        mesh.export(path)
        return path
    return run


BACKENDS = {"shap_e": shap_e, "trellis": trellis, "hunyuan3d": hunyuan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=BACKENDS)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--root", type=Path, default=ROOT / "output/e12")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    run = BACKENDS[args.model]()
    for s in args.seeds:
        d = args.root / args.model / f"seed_{s}"
        (d / "raw").mkdir(parents=True, exist_ok=True)
        prompts = json.load(open(args.root / "prompts" / f"seed_{s}.json"))
        if args.limit:
            prompts = prompts[:args.limit]
        log_path = d / "generate.json"
        log = json.load(open(log_path)) if log_path.exists() else {}
        for i, p in enumerate(prompts):
            key = f"{i:03d}_{p['name']}"
            if key in log:
                continue
            t0 = time.time()
            rec = {"prompt": p["prompt"], "name": p["name"], "seed": s * 1000 + i}
            try:
                torch.cuda.reset_peak_memory_stats()
                rec["path"] = run(p["prompt"], rec["seed"], d / "raw" / key)
            except Exception as e:     # noqa: BLE001 - a model failure is a result: the object has no mesh
                rec["error"] = f"{type(e).__name__}: {e}"
                traceback.print_exc()
            rec["seconds"] = time.time() - t0
            rec["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
            log[key] = rec
            log_path.write_text(json.dumps(log, indent=1))
            print(key, f"{rec['seconds']:.1f}s", f"{rec['peak_gb']:.1f} GB", rec.get("error", "ok")[:100], flush=True)


if __name__ == "__main__":
    main()
