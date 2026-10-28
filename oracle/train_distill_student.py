#!/usr/bin/env python3
"""Single-GPU, warm-started single-layer multi-concept distillation SFT (not RL).

Use frozen NNOMP/assembly bullet targets and TRUE single-layer residual vectors.
Student prompt uses Agam's concepts_raw wording and a verified single-token
injection slot. The parent AO-SFT used a different prompt with the slot at
its final token; that *intentional* student-stage change is recorded.
"""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_TEMPLATE = (
    "An activation vector from layer {layer} of a language model is enclosed in activation "
    "tags: <activation>{char}</activation>. Produce distinct concepts that encode this "
    "activation, each as a '- ' bullet on its own line."
)
STAGE = "l26_distill_student_sft_v1"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def render_concepts_prompt(tokenizer, layer):
    """Match upstream's chat template and ensure a unique 1-token slot."""
    for code in range(0x3200, 0x3400):
        char = chr(code)
        alone = tokenizer(char, add_special_tokens=False).input_ids
        if len(alone) != 1:
            continue
        body = PROMPT_TEMPLATE.format(layer=layer, char=char)
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": body}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False)
        ids = rendered if isinstance(rendered, list) else rendered["input_ids"]
        matches = [i for i, token in enumerate(ids) if token == alone[0]]
        if len(matches) == 1:
            return torch.tensor(ids, dtype=torch.long), matches[0], char
    raise RuntimeError("No unique single-token concepts_raw injection slot")


def load_targets(path, expected_split):
    if not (path / "COMPLETE").is_file():
        raise ValueError(f"Incomplete distillation assembly: {path}")
    meta = json.loads((path / "meta.json").read_text())
    if (meta.get("schema") != f"l{meta.get('layer')}_distill_bullets_v1"
        or meta.get("source") != "true_GT_residual"
        or meta.get("prompt_mode") != "concepts_raw"
        or meta.get("split") != expected_split
        or meta.get("layer") not in (26, 30)
        or meta.get("filter_policy") not in ("upstream", "no_unigram")):
        raise ValueError(f"Incorrect assembly source contract: {path}")
    data_file = path / "targets.safetensors"
    if not data_file.is_file() or sha256(data_file) != meta.get("targets_sha256"):
        raise ValueError(f"Missing or SHA-mismatched targets: {path}")
    data = load_file(str(data_file))
    n = len(data["vec"])
    lengths = data["offsets"]
    if (n < 1 or len(lengths) != n + 1
        or lengths[0].item() != 0 or lengths[-1].item() != len(data["target_ids"])
        or (lengths[1:] <= lengths[:-1]).any()
        or len(data["src_pair_row"]) != n
        or len(data["src_item_idx"]) != n
        or data["vec"].ndim != 2 or data["vec"].shape[1] != 2560):
        raise ValueError(f"Malformed packed bullet targets: {path}")
    if n != meta.get("kept_rows") or not torch.isfinite(data["vec"]).all():
        raise ValueError(f"Invalid vectors or item count: {path}")
    return meta, data


def row(data, idx):
    lo, hi = (int(x) for x in data["offsets"][idx:idx+2])
    return data["vec"][idx].float(), data["target_ids"][lo:hi].long()


def pack(rows, pad_id):
    """Targets ALREADY include an EOS; mask padding and prompt loss."""
    vectors = torch.stack([r[0] for r in rows])
    width = max(len(r[1]) for r in rows)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    masks = torch.zeros_like(ids)
    for i, (_, target) in enumerate(rows):
        ids[i, :len(target)] = target
        labels[i, :len(target)] = target
        masks[i, :len(target)] = 1
    return vectors, ids, labels, masks


def conditioned_batch(model, prompt_ids, slot, vectors, target_ids, labels, masks, alpha):
    """Replace exactly the single prompt embedding, with raw true h normalized."""
    embed = model.get_input_embeddings()
    b = len(vectors)
    prompts = prompt_ids[None, :].expand(b, -1)
    pe = embed(prompts).clone()
    if vectors.ndim != 2 or vectors.shape[-1] != pe.shape[-1]:
        raise ValueError("Activation/input embedding width mismatch")
    if not torch.isfinite(vectors).all() or (vectors.float().norm(dim=1) < 1e-8).any():
        raise ValueError("Nonfinite or zero true residual")
    pe[:, slot, :] = (F.normalize(vectors.float(), dim=-1) * alpha).to(pe.dtype)
    joined = torch.cat((pe, embed(target_ids)), dim=1)
    attn = torch.cat((torch.ones(b, len(prompt_ids), device=masks.device, dtype=masks.dtype),
                      masks), dim=1)
    loss_labels = torch.cat((torch.full((b, len(prompt_ids)), -100,
                                        device=labels.device, dtype=labels.dtype),
                             labels), dim=1)
    return joined, attn, loss_labels


def verified_contract(train_path, val_path, parent_path):
    train_meta, train = load_targets(train_path, "train")
    val_meta, val = load_targets(val_path, "val")
    parent_meta = json.loads((parent_path / "meta.json").read_text())
    if not (parent_path / "COMPLETE").exists():
        raise ValueError("Warm-start AO-SFT checkpoint incomplete")
    adapter = parent_path / "lora" / "adapter_model.safetensors"
    if not adapter.is_file():
        raise ValueError("Parent LoRA adapter missing")
    if (parent_meta.get("stage") != f"l{parent_meta.get('layer')}_ao_sft_fresh_pool_v1"
        or parent_meta.get("layer") not in (26, 30)
        or parent_meta.get("source") != "frozen_AR_of_fresh_nonoverlap_AO_crops"
        or parent_meta.get("transform") != "unit"
        or float(parent_meta.get("alpha", -1)) != 8000
        or parent_meta.get("prompt_slot") != "last token of rendered prompt"):
        raise ValueError("Unexpected parent AO-SFT warm-start contract")
    for meta in (train_meta, val_meta):
        if (meta["model"] != parent_meta["model"]
            or meta["layer"] != parent_meta["layer"]
            or Path(meta["teacher"]).resolve() != parent_path.resolve()):
            raise ValueError("Target/parent mismatch")
    if train_meta["items_sha256"] != val_meta["items_sha256"]:
        try:
            from .distill_expansion import verify_expansion, verify_targets
        except ImportError:
            from distill_expansion import verify_expansion, verify_targets
        _, expanded, original = verify_expansion(train_meta["items_path"],
            train_meta["items_sha256"], val_meta["items_sha256"])
        verify_targets(train_meta, train, expanded, "train")
        verify_targets(val_meta, val, original, "val")
    if set(int(v) for v in train["src_pair_row"]).intersection(
        int(v) for v in val["src_pair_row"]):
        raise ValueError("Train/validation source pair rows overlap")
    return train_meta, train, val_meta, val, parent_meta, sha256(adapter)


def next_indices(n, seed, step, micro, batch_size):
    """Deterministic per optimizer-step schedule; no DataLoader resume state."""
    rng = random.Random(f"{seed}:{step}:{micro}")
    return [rng.randrange(n) for _ in range(batch_size)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--val", type=Path, required=True,
                   help="Use same fixed validation target artifact across variants")
    p.add_argument("--init-from", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="CPU-only manifest and rendered prompt/target inspection")
    a = p.parse_args()
    if (min(a.max_steps,a.save_every,a.batch_size,a.grad_accum) <= 0
        or not 0 < a.lr < 1):
        p.error("Invalid training parameters")
    if a.train.resolve() == a.val.resolve():
        p.error("Train/validation artifact must be different")
    if a.resume and a.dry_run:
        p.error("Cannot resume in dry-run")
    if a.output.exists() and not a.resume and not a.dry_run:
        p.error("Output directory exists; specify --resume or a fresh output")

    trm, tr, vam, va, parent, adapter_sha = verified_contract(a.train, a.val, a.init_from)
    tokenizer = AutoTokenizer.from_pretrained(a.init_from / "tokenizer")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token_id is None:
        p.error("Tokenizer has no EOS")
    for data in (tr,va):
        ends = data["target_ids"][data["offsets"][1:].long() - 1]
        if not bool((ends == tokenizer.eos_token_id).all()):
            p.error("Target EOS mismatch")
    prompt, slot, char = render_concepts_prompt(tokenizer, parent["layer"])
    info = {
        "stage": f"l{parent['layer']}_distill_student_sft_v1", "model": parent["model"], "revision": parent.get("revision"),
        "layer": parent["layer"], "transform": "unit", "alpha": parent["alpha"],
        "prompt_mode": "concepts_raw", "prompt": PROMPT_TEMPLATE.format(layer=parent["layer"],char=char),
        "prompt_slot": slot, "prompt_char": char,
        "rendered_prompt_ids_sha256": hashlib.sha256(prompt.numpy().tobytes()).hexdigest(),
        "init_from": str(a.init_from.resolve()), "init_adapter_sha256": adapter_sha,
        "train": str(a.train.resolve()), "train_targets_sha256": trm["targets_sha256"],
        "train_filter_policy": trm["filter_policy"], "train_rows": len(tr["vec"]),
        "val": str(a.val.resolve()), "val_targets_sha256": vam["targets_sha256"],
        "val_filter_policy": vam["filter_policy"], "val_rows": len(va["vec"]),
        "items_sha256": trm["items_sha256"], "source": "true_GT_residual",
        "training": {"max_steps":a.max_steps, "save_every":a.save_every,
                     "batch_size":a.batch_size,"grad_accum":a.grad_accum,
                     "lr":a.lr,"seed":a.seed},
        "evaluation_status": "pilot validation, NOT independent final test",
        "upstream_contract_delta": (
            "Student uses chat-templated concepts_raw prompt with interior single-token "
            "activation slot, unlike parent's short plain-text final-token prompt; "
            "both inject unit true residual at alpha=8000.")
    }
    print(json.dumps(info,indent=2,ensure_ascii=False),flush=True)
    print("Example target:",tokenizer.decode(row(tr,0)[1].tolist())[:360],flush=True)
    if a.dry_run:
        print("DRY RUN: no CUDA, model or output",flush=True)
        return
    if not torch.cuda.is_available():
        p.error("CUDA required")
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    if a.resume:
        latest_path = a.output / "latest.json"
        if not latest_path.is_file() or not (a.output / "meta.json").is_file():
            p.error("Cannot resume without complete run metadata")
        if json.loads((a.output / "meta.json").read_text()) != info:
            p.error("Resume manifest differs; new output required")
        latest = json.loads(latest_path.read_text())
        cp = a.output / "checkpoints" / f"step_{int(latest['step']):06d}"
        if cp.resolve() != (a.output / latest["checkpoint"]).resolve() or not (cp / "COMPLETE").is_file():
            p.error("Resume checkpoint pointer incomplete or invalid")
    else:
        a.output.mkdir(parents=True)
        atomic_json(a.output / "meta.json", info)

    base = AutoModelForCausalLM.from_pretrained(parent["model"],
             revision=parent.get("revision"),dtype=torch.bfloat16).cuda()
    base.config.use_cache = False
    if base.get_input_embeddings().weight.shape[1] != 2560:
        raise RuntimeError("Base Qwen3.5 embedding width mismatch")
    lora_path = cp / "lora" if a.resume else a.init_from / "lora"
    model = PeftModel.from_pretrained(base, lora_path, is_trainable=True)
    trainable = sum(t.numel() for t in model.parameters() if t.requires_grad)
    if not trainable:
        raise RuntimeError("Parent LoRA has no trainable parameters")
    print(f"Warm-started {trainable} LoRA parameters from {lora_path}",flush=True)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    opt = torch.optim.AdamW((x for x in model.parameters() if x.requires_grad),lr=a.lr)
    device = torch.device("cuda")
    prompt = prompt.to(device)

    def batch(data, indices):
        return (t.to(device) for t in pack(
            [row(data,i) for i in indices], tokenizer.pad_token_id))

    def evaluate():
        model.eval()
        loss_sum=0.0
        tokens=0
        with torch.no_grad():
            for idx in range(0,len(va["vec"]),a.batch_size):
                v,ids,labels,mask=batch(va,list(range(idx,min(idx+a.batch_size,len(va["vec"])))))
                embeddings,attn,full_labels=conditioned_batch(
                    model,prompt,slot,v,ids,labels,mask,parent["alpha"])
                loss=model(inputs_embeds=embeddings,attention_mask=attn,labels=full_labels).loss
                n=int((full_labels!=-100).sum())
                loss_sum+=float(loss)*n
                tokens+=n
        if tokens==0:
            raise RuntimeError("Validation has no supervised tokens")
        return {"val_token_ce":loss_sum/tokens,"val_supervised_tokens":tokens}

    def save(step, metrics):
        parent_dir=a.output / "checkpoints"
        parent_dir.mkdir(exist_ok=True)
        path=parent_dir / f"step_{step:06d}"
        staging=parent_dir / f".step_{step:06d}.partial"
        if path.exists() or staging.exists():
            raise RuntimeError(f"Refusing overwrite: {path} or {staging}")
        staging.mkdir()
        model.save_pretrained(staging / "lora")
        tokenizer.save_pretrained(staging / "tokenizer")
        atomic_json(staging / "metrics.json",metrics)
        torch.save({"optimizer":opt.state_dict(),"step":step,
                    "torch_rng":torch.get_rng_state(),
                    "cuda_rng":torch.cuda.get_rng_state_all()},staging / "state.pt")
        (staging / "COMPLETE").write_text("ok\n")
        os.replace(staging,path)
        atomic_json(a.output / "latest.json",
                    {"step":step,"checkpoint":str(path.relative_to(a.output))})
        print("SAVED",json.dumps(metrics),flush=True)

    start=0
    if a.resume:
        state=torch.load(cp / "state.pt",map_location="cpu",weights_only=False)
        if int(state["step"])!=int(latest["step"]):
            raise ValueError("Optimizer checkpoint step differs from latest pointer")
        opt.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        start=int(state["step"])
    if start == a.max_steps:
        print("TRAINING ALREADY COMPLETE",a.output)
        return
    if start > a.max_steps:
        p.error("Resume step exceeds target")

    # An initial shared-val CE is useful for verifying warm-start and prompt
    # transition, particularly when two filter policies use the same holdout.
    if start==0:
        print("INITIAL VALIDATION",json.dumps(evaluate()),flush=True)
    for step in range(start+1,a.max_steps+1):
        model.train()
        opt.zero_grad(set_to_none=True)
        train_ce=0.0
        for micro in range(a.grad_accum):
            chosen=next_indices(len(tr["vec"]),a.seed,step,micro,a.batch_size)
            v,ids,labels,mask=batch(tr,chosen)
            embeddings,attn,full_labels=conditioned_batch(
                model,prompt,slot,v,ids,labels,mask,parent["alpha"])
            loss=model(inputs_embeds=embeddings,attention_mask=attn,labels=full_labels).loss
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite training loss at step {step}")
            (loss/a.grad_accum).backward()
            train_ce+=float(loss.detach())/a.grad_accum
        torch.nn.utils.clip_grad_norm_((x for x in model.parameters() if x.requires_grad),1.0)
        opt.step()
        print(f"step={step} train_ce={train_ce:.4f}",flush=True)
        if step%a.save_every==0 or step==a.max_steps:
            metrics={"step":step,"train_token_ce":train_ce,**evaluate()}
            save(step,metrics)
    print("DISTILL STUDENT COMPLETE",a.output,flush=True)


if __name__=="__main__":
    main()

