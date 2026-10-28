#!/usr/bin/env python3
"""Prepare true single-layer residuals for Agam-style distillation teacher sampling.

This is a new GT-activation source, not the AO-SFT AR-output pool. Draw one
nontrivial span per rollout from the AR-held-out partition; hash-split rollouts
between distillation train and validation. All choices are deterministic.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "legacy"))
sys.path.insert(0, str(ROOT / "oracle"))
from train_ar import load_corpus  # noqa: E402
from train_gt_sft import PROMPT  # noqa: E402


def degenerate(text):
    """Port of upstream r2_filter's F1 heuristic for selecting teacher items."""
    from collections import Counter
    p = text.strip()
    if len("".join(p.split())) < 8:
        return True
    if sum(c.isalnum() for c in p) / max(1, len(p)) < 0.30:
        return True
    words = p.split()
    if len(words) >= 4 and Counter(words).most_common(1)[0][1] / len(words) >= .5:
        return True
    if len(p) >= 20:
        grams = Counter(p[i:i+4] for i in range(len(p)-3))
        if grams and 4 * grams.most_common(1)[0][1] / len(p) > .5 and grams.most_common(1)[0][1] >= 4:
            return True
    return False


def partition(shard, rollout, seed, val_frac):
    payload = f"distill-v1:{seed}:{shard}:{rollout}".encode()
    u = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64
    return "val" if u < val_frac else "train"


def choose_items(corpus, tok, seed, n_train, n_val, val_frac=0.15, max_span=64, exclude=()):
    rng = random.Random(seed)
    groups = {}
    for i in corpus.val_indices.tolist():
        key = (int(corpus.shard_ids[i]), int(corpus.rollout_ids[i]))
        groups.setdefault(key, []).append(i)
    keys = sorted(set(groups) - set(exclude))
    rng.shuffle(keys)
    selection = {"train": [], "val": []}
    counts = {"groups": len(keys), "rejected_degenerate": 0,
              "rejected_zero_norm": 0, "rejected_short": 0}
    for shard, rollout in keys:
        split = partition(shard, rollout, seed, val_frac)
        cap = n_train if split == "train" else n_val
        if len(selection[split]) >= cap:
            continue
        positions = list(groups[(shard, rollout)])
        rng.shuffle(positions)
        for i in positions:
            lo, hi = int(corpus.span_offsets[i]), int(corpus.span_offsets[i + 1])
            ids = corpus.span_token_ids[lo:min(hi,lo+max_span)].tolist()
            if len(ids) < 2:
                counts["rejected_short"] += 1
                continue
            if not torch.isfinite(corpus.activations[i]).all() or corpus.activations[i].norm() <= 1e-8:
                counts["rejected_zero_norm"] += 1
                continue
            if degenerate(tok.decode(ids, skip_special_tokens=True)):
                counts["rejected_degenerate"] += 1
                continue
            selection[split].append((i, shard, rollout, ids))
            break
        if len(selection["train"]) >= n_train and len(selection["val"]) >= n_val:
            break
    if len(selection["train"]) != n_train or len(selection["val"]) != n_val:
        raise ValueError(f"Insufficient valid distinct rollouts: { {k: len(v) for k,v in selection.items()} }")
    assert {(x[1],x[2]) for x in selection["train"]}.isdisjoint(
        {(x[1],x[2]) for x in selection["val"]})
    return selection, counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("/workspace/o-lens-qwen35-4b/data/olens_200k_all32"))
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n-train", type=int, default=512)
    p.add_argument("--n-val", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--layer", type=int, default=26)
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--base-items", type=Path, help="Append to this bank; retain its validation tensors exactly")
    p.add_argument("--fresh-data", type=Path, help="Dedicated prompt-disjoint harvest; all new rollouts eligible")
    p.add_argument("--reserve-test", type=int, default=256)
    p.add_argument("--source-config", type=Path, help="Require exact historical corpus inventory/counts")
    p.add_argument("--dry-run", action="store_true", help="Select and audit capacity without writing")
    args = p.parse_args()
    if args.fresh_data and not args.base_items:
        p.error("--fresh-data requires --base-items")
    if min(args.n_train, args.n_val) < 1 or args.output.exists():
        p.error("Need positive counts and a new output directory")
    if not (args.teacher / "COMPLETE").is_file():
        p.error("Teacher checkpoint is incomplete")
    teacher = json.loads((args.teacher / "meta.json").read_text())
    if (teacher.get("stage") != f"l{args.layer}_ao_sft_fresh_pool_v1" or
        teacher.get("layer") != args.layer or teacher.get("model") != args.model or
        teacher.get("prompt") != PROMPT or teacher.get("transform") != "unit" or
        teacher.get("source") != "frozen_AR_of_fresh_nonoverlap_AO_crops"):
        p.error("Teacher model, prompt or injection contract mismatch")
    ns = argparse.Namespace(layer=args.layer, pair_dir=args.data/"pairs",
         rollout_dir=args.data/"rollouts", pair_glob=None,
         rollout_glob="rollouts_*.jsonl", split_seed=42, val_frac=.10)
    expected = None
    if args.source_config:
        expected = json.loads(args.source_config.read_text())
        if expected["target_layer"] != args.layer or expected["model_id"] != args.model:
            p.error("Source configuration model/layer mismatch")
        for directory, pattern, key in (("pairs", f"pairs_l{args.layer}_*.safetensors", "pair_shards"),
                                         ("rollouts", "rollouts_*.jsonl", "rollout_shards")):
            if sorted(x.name for x in (args.data/directory).glob(pattern)) != sorted(expected[key]):
                p.error("Corpus listing differs from AR inventory; stage a verified local copy")
    corpus = load_corpus(ns)
    if expected and (len(corpus.train_indices) != expected["train_pairs"] or
                     len(corpus.val_indices) != expected["val_pairs"]):
        p.error("Corpus split counts differ from historical AR source")
    if corpus.d_model != 2560:
        raise RuntimeError(f"Unexpected activation width: {corpus.d_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher/"tokenizer")
    old = None
    fresh_audit = None
    if args.base_items:
        from distill_expansion import bank, groups
        base_meta, old = bank(args.base_items)
        if base_meta.get("schema") != "gt_distill_items_v1" or base_meta.get("expansion"):
            raise ValueError("Expected original pilot bank, not a nested expansion")
        if args.seed != base_meta["seed"]:
            raise ValueError("Preserve original partition seed")
        if any(base_meta[k] != v for k, v in (("model", args.model), ("layer", args.layer), ("teacher", str(args.teacher)))):
            raise ValueError("Base bank teacher/model mismatch")
        if args.n_train <= len(old["train_vec"]) or args.reserve_test < 1:
            raise ValueError("Expansion needs more training items and a positive reservation")
        # Verify row IDs still refer to exactly the same corpus, not a reordered harvest.
        for split in ("train", "val"):
            for j, i in enumerate(old[split + "_pair_rows"].tolist()):
                lo = int(corpus.span_offsets[i])
                n = int(old[split + "_lengths"][j])
                if (not bool(corpus.is_val[i]) or
                    int(corpus.shard_ids[i]) != int(old[split + "_shards"][j]) or
                    int(corpus.rollout_ids[i]) != int(old[split + "_rollouts"][j]) or
                    not torch.equal(corpus.activations[i], old[split + "_vec"][j]) or
                    not torch.equal(corpus.span_token_ids[lo:lo+n], old[split + "_ids"][j,:n])):
                    raise ValueError("Base bank no longer matches corpus")
        if args.fresh_data:
            from fresh_distill_corpus import load_fresh
            corpus, fresh_audit = load_fresh(args.fresh_data, corpus, ns,
                args.model, args.layer, teacher.get("revision"))
        selection, counts = choose_items(corpus, tokenizer, args.seed,
            args.n_train - len(old["train_vec"]), args.reserve_test,
            exclude=groups(old, "train") | groups(old, "val"))
        selection["test"] = selection.pop("val")
    else:
        selection, counts = choose_items(corpus, tokenizer, args.seed, args.n_train, args.n_val)
    if args.dry_run:
        print(json.dumps({"new_rows": {k:len(v) for k,v in selection.items()}, "counts":counts, "fresh_audit":fresh_audit}))
        return
    tensors = {}
    meta = {"schema":"gt_distill_items_v1","layer":args.layer,"model":args.model,
            "teacher":str(args.teacher),"teacher_alpha":float(teacher["alpha"]),
            "teacher_revision":teacher.get("revision"),"prompt":teacher["prompt"],
            "seed":args.seed,"source_partition":"AR-held-out-rollout-v1",
            "distill_partition":"rollout-hash-15pct-val-v1","selection_counts":counts,
            "note":"Pilot distill splits, not untouched final evaluation"}
    for split, rows in selection.items():
        vec = torch.stack([corpus.activations[i] for i,*_ in rows]).contiguous()
        ids = torch.zeros((len(rows),64),dtype=torch.int32)
        lengths = []
        for j,(_,_,_,span) in enumerate(rows):
            ids[j,:len(span)] = torch.tensor(span,dtype=torch.int32)
            lengths.append(len(span))
        tensors[split+"_vec"] = vec
        tensors[split+"_ids"] = ids
        tensors[split+"_lengths"] = torch.tensor(lengths,dtype=torch.int32)
        tensors[split+"_pair_rows"] = torch.tensor([row[0] for row in rows],dtype=torch.int64)
        tensors[split+"_shards"] = torch.tensor([row[1] for row in rows],dtype=torch.int32)
        tensors[split+"_rollouts"] = torch.tensor([row[2] for row in rows],dtype=torch.int32)
        meta[split+"_rows"] = len(rows)
    if fresh_audit:
        for split in selection:
            tensors[split + "_pair_rows"] += fresh_audit["pair_row_offset"]
        meta["fresh_corpus"] = fresh_audit
        meta["source_partition"] = "original-prefix-plus-dedicated-fresh-harvest-v1"
        meta["distill_partition"] = "fresh-rollout-hash-15pct-reserved-test-v1"
    if old is not None:
        from distill_expansion import digest
        for key, value in old.items():
            tensors[key] = torch.cat((value, tensors[key]), dim=0) if key.startswith("train_") else value.clone()
        meta["train_rows"] = len(tensors["train_vec"])
        meta["val_rows"] = len(tensors["val_vec"])
        meta["expansion"] = {"base_items":str(args.base_items.resolve()),
            "base_sha256":base_meta["items_sha256"],
            "original_train_rows":len(old["train_vec"]),
            "reservation":"Prospective only; historical AO exposure not audited"}
        meta["corpus_sha256"] = {str(p):digest(p) for p in corpus.pair_paths + corpus.rollout_paths}
    args.output.mkdir(parents=True)
    file = args.output/"items.safetensors"
    save_file(tensors,str(file))
    meta["items_sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
    (args.output/"meta.json").write_text(json.dumps(meta, indent=2)+"\n")
    (args.output/"COMPLETE").write_text("ok\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

