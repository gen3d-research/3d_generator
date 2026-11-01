#!/usr/bin/env python3
"""Downloads the weights E12 needs, only the files it uses, into the Hugging Face cache. Prints sizes."""
from huggingface_hub import snapshot_download
from pathlib import Path

REPOS = [
    ("openai/shap-e", ["*.json", "*.txt", "*fp16.safetensors", "text_encoder/*", "tokenizer/*"]),
    ("microsoft/TRELLIS-text-large", None),
    ("microsoft/TRELLIS-image-large", ["pipeline.json", "ckpts/slat_dec_mesh*", "ckpts/ss_dec*"]),
    ("tencent/Hunyuan3D-2", ["hunyuan3d-dit-v2-0/*.yaml", "hunyuan3d-dit-v2-0/model.fp16.safetensors",
                             "hunyuan3d-vae-v2-0/*", "LICENSE", "NOTICE"]),
    ("Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled", None),
]
total = 0
for repo, patterns in REPOS:
    path = Path(snapshot_download(repo, allow_patterns=patterns))
    size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    total += size
    print(f"{repo} {size / 1e9:.1f} GB {path}", flush=True)
print(f"TOTAL {total / 1e9:.1f} GB", flush=True)
