#!/usr/bin/env python
"""
Fix ComfyUI Anima LoRA key naming bug:

Some LoRA converters incorrectly split `llm_adapter` into `llm.adapter`, producing keys like:
  diffusion_model.llm.adapter.blocks.0.self_attn.q_proj.lora_down.weight
ComfyUI expects:
  diffusion_model.llm_adapter.blocks.0.self_attn.q_proj.lora_down.weight

This script rewrites keys in-place to the correct prefix and writes a new .safetensors file.

Usage:
  python fix_comfyui_lora_keys.py INPUT.safetensors [OUTPUT.safetensors]
  python fix_comfyui_lora_keys.py --dir DIR [--pattern "*.safetensors"]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


BAD = "diffusion_model.llm.adapter."
GOOD = "diffusion_model.llm_adapter."


def _load(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    weights: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        meta = dict(f.metadata() or {})
        for k in f.keys():
            weights[k] = f.get_tensor(k)
    return weights, meta


def fix_file(inp: Path, out: Path) -> dict[str, int]:
    w, meta = _load(inp)
    rewrites = 0
    kept = 0
    new_w: dict[str, torch.Tensor] = {}
    for k, t in w.items():
        nk = k
        if nk.startswith(BAD):
            nk = GOOD + nk[len(BAD) :]
            rewrites += 1
        else:
            kept += 1
        if nk in new_w:
            raise RuntimeError(f"Key collision after rewrite: {nk}")
        new_w[nk] = t

    # Preserve metadata, but mark as fixed
    meta = dict(meta)
    meta["fixed_keys"] = "llm.adapter->llm_adapter"

    save_file(new_w, str(out), metadata=meta)
    return {"rewritten": rewrites, "kept": kept, "total": len(w)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Fix ComfyUI Anima LoRA key naming (llm.adapter -> llm_adapter)")
    ap.add_argument("input", nargs="?", help="Input .safetensors file")
    ap.add_argument("output", nargs="?", help="Output .safetensors file (optional)")
    ap.add_argument("--dir", dest="dir_", help="Batch mode: directory containing .safetensors")
    ap.add_argument("--pattern", default="*.safetensors", help="Batch glob pattern (default: *.safetensors)")
    args = ap.parse_args()

    if args.dir_:
        d = Path(args.dir_)
        files = sorted(d.glob(args.pattern))
        files = [p for p in files if p.is_file() and p.suffix.lower() == ".safetensors"]
        if not files:
            raise SystemExit(f"No files matched: {d / args.pattern}")
        for p in files:
            if p.stem.endswith("_fixedkeys"):
                continue
            out = p.with_name(p.stem + "_fixedkeys.safetensors")
            stats = fix_file(p, out)
            print(f"{p.name} -> {out.name} | rewritten={stats['rewritten']} total={stats['total']}")
        return

    if not args.input:
        raise SystemExit("Provide INPUT.safetensors or use --dir")
    inp = Path(args.input)
    if args.output:
        out = Path(args.output)
    else:
        out = inp.with_name(inp.stem + "_fixedkeys.safetensors")
    stats = fix_file(inp, out)
    print(f"wrote {out} | rewritten={stats['rewritten']} total={stats['total']}")


if __name__ == "__main__":
    main()

