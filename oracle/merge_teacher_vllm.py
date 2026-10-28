#!/usr/bin/env python3
"""Merge a completed single-layer SFT teacher LoRA for vLLM offline prompt-embed sampling.

Run in the EXISTING training venv, not the separately installed vLLM venv.
Merged ~9 GB model is intentionally on ephemeral local disk, never FUSE.
The source checkpoint and persistent network-volume artifacts are unchanged.
"""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from sample_distill_teacher import check_adapter_live, sha256


def config_contract(teacher_dir):
    if not (teacher_dir / "COMPLETE").is_file():
        raise ValueError("Teacher checkpoint is incomplete")
    meta = json.loads((teacher_dir / "meta.json").read_text())
    if meta.get("layer") not in (26, 30):
        raise ValueError("Supported teacher layers: 26, 30")
    required = {"stage": f"l{meta['layer']}_ao_sft_fresh_pool_v1",
                "model": "Qwen/Qwen3.5-4B", "layer": meta["layer"],
                "transform": "unit", "prompt_slot": "last token of rendered prompt",
                "source": "frozen_AR_of_fresh_nonoverlap_AO_crops"}
    if any(meta.get(k) != v for k, v in required.items()):
        raise ValueError("Teacher does not match frozen SFT injection contract")
    if float(meta.get("alpha", 0)) <= 0 or not meta.get("prompt"):
        raise ValueError("Missing injection alpha or prompt")
    return meta


def weight_files_and_embed_key(directory):
    """Find the one token embedding in a saved HF safetensors index."""
    from safetensors import safe_open
    index = directory / "model.safetensors.index.json"
    if index.exists():
        mapping = json.loads(index.read_text())["weight_map"]
    elif (directory / "model.safetensors").exists():
        with safe_open(str(directory / "model.safetensors"), framework="pt") as f:
            mapping = {key: "model.safetensors" for key in f.keys()}
    else:
        raise RuntimeError("Saved merged model lacks safetensors weights")
    matches = [k for k in mapping if k.endswith("embed_tokens.weight")]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one text token embedding weight; found {matches}")
    return sorted(set(mapping.values())), matches[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True,
                   help="Local EPHEMERAL dir, e.g. /root/olens-merged/l26-step500")
    p.add_argument("--cache-dir", type=Path, default=Path("/root/.cache/huggingface/hub"))
    p.add_argument("--min-free-gib", type=float, default=12)
    args = p.parse_args()
    output = args.output.resolve()
    if output.exists() or output == Path("/root") or output == Path("/"):
        p.error("Output already exists or is not a new leaf directory")
    if output == Path("/workspace") or Path("/workspace") in output.parents:
        p.error("Never merge large HF models onto the RunPod FUSE volume")
    output.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output.parent).free
    if free < args.min_free_gib * 1024**3:
        p.error(f"Only {free/1024**3:.1f} GiB free locally; need >= {args.min_free_gib}")
    meta = config_contract(args.teacher)
    adapter_path = args.teacher / "lora" / "adapter_model.safetensors"
    if not adapter_path.is_file():
        p.error("Missing LoRA adapter")
    tok = AutoTokenizer.from_pretrained(args.teacher / "tokenizer")
    prompt_ids = tok(meta["prompt"], add_special_tokens=False).input_ids
    if not prompt_ids:
        p.error("Teacher prompt tokenizes to empty IDs")
    if not torch.cuda.is_available():
        p.error("CUDA required for the one-time merge")
    base = AutoModelForCausalLM.from_pretrained(
        meta["model"], revision=meta.get("revision"),
        dtype=torch.bfloat16, device_map="cuda", cache_dir=str(args.cache_dir))
    base.config.use_cache = False
    peft = PeftModel.from_pretrained(base, args.teacher / "lora",
                                    is_trainable=False).eval()
    check_adapter_live(peft, adapter_path)
    # A one-step, no-soft-token parity gate isolates PEFT merge correctness.
    ids = torch.tensor([prompt_ids], device="cuda")
    with torch.inference_mode():
        logits_before = peft(input_ids=ids).logits[:, -1, :].float()
    merged = peft.merge_and_unload(safe_merge=True).eval()
    with torch.inference_mode():
        logits_after = merged(input_ids=ids).logits[:, -1, :].float()
    cosine = float(F.cosine_similarity(logits_before, logits_after).mean())
    argmax_match = bool(torch.equal(logits_before.argmax(-1), logits_after.argmax(-1)))
    max_abs = float((logits_before-logits_after).abs().max())
    if cosine < 0.999 or not argmax_match:
        raise RuntimeError(f"Failed merged-adapter parity: cos={cosine:.6f} "
                           f"argmax={argmax_match} maxabs={max_abs:.4f}")
    staging = output.with_name(output.name + ".partial")
    if staging.exists():
        p.error(f"Staging dir already exists (inspect/remove manually): {staging}")
    staging.mkdir()
    merged.save_pretrained(staging, safe_serialization=True, max_shard_size="4GB")
    tok.save_pretrained(staging)
    config = AutoConfig.from_pretrained(staging)
    files, embkey = weight_files_and_embed_key(staging)
    manifest = {"schema":f"l{meta['layer']}_vllm_merged_teacher_v1",
                "source_checkpoint":str(args.teacher),
                "source_adapter_sha256":sha256(adapter_path),
                "model":meta["model"],"revision":meta.get("revision"),
                "layer":meta["layer"],"alpha":meta["alpha"],
                "prompt":meta["prompt"],"prompt_slot":meta["prompt_slot"],
                "transform":meta["transform"],"merge_cosine":cosine,
                "merge_argmax_match":argmax_match,"merge_max_abs":max_abs,
                "config_model_type":config.model_type,
                "embedding_key":embkey,
                "weights":{name:sha256(staging/name) for name in files},
                "notes":"Ephemeral merged model; all authoritative data remains on /workspace"}
    (staging / "olens_merge_manifest.json").write_text(
        json.dumps(manifest,indent=2)+"\n")
    os.replace(staging,output)
    print(json.dumps(manifest,indent=2),flush=True)
    print("MERGE COMPLETE",output,flush=True)


if __name__=="__main__":
    main()
