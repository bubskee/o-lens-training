#!/usr/bin/env python3
"""CPU-only Agam-style NNOMP bullet target assembly; no model inference."""
import argparse
import hashlib
import json
import os
import re
from collections import Counter
from itertools import pairwise
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

WORD = re.compile(r"[A-Za-z0-9À-ɏЀ-ӿ]+")
CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")
FILTER_POLICIES = ("upstream", "no_unigram")

LEAD = "'" + chr(96) + "\".,;:!?)]}>* \t\n-" + "".join(
    map(chr, (0x2019,0x3002,0xFF0C,0x3001,0xFF1B,0xFF1A,
              0xFF09,0x3011,0x300D,0x201D,0x2026,0x2013,0x2014)))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def units(text):
    return WORD.findall(text) + CJK.findall(text)


def keys(text):
    c = CJK.findall(text)
    return {x.lower() for x in WORD.findall(text)} | {a+b for a,b in pairwise(c)}


def ngram_loop(text, min_n=1):
    """Upstream checks n=1,2,3; exploratory alternative checks n=2,3."""
    if min_n not in (1,2):
        raise ValueError("ngram min_n must be 1 (upstream) or 2 (no_unigram)")
    u = [x.lower() for x in units(text)]
    for n in range(min_n,4):
        if len(u) >= 3*n:
            c = Counter(tuple(u[i:i+n]) for i in range(len(u)-n+1))
            if c and c.most_common(1)[0][1] >= 3:
                return True
    return False


def reject_reason(picks, n_cand, min_picks=4, filter_policy="upstream"):
    """First-rejection rule; only the n-gram minimum differs by policy."""
    if filter_policy not in FILTER_POLICIES:
        raise ValueError(f"Unknown assembly filter policy: {filter_policy}")
    if n_cand < 25:
        return "n_cand<25"
    if not min_picks <= len(picks) <= 4:
        return "n_sel!=4" if min_picks == 4 else f"n_sel<{min_picks}"
    k = [keys(x) for x in picks]
    for i,p in enumerate(picks):
        if len(units(p)) < 2:
            return "min_units<2"
        if "\ufffd" in p:
            return "replchar"
        if (ngram_loop(p) if filter_policy == "upstream"
            else ngram_loop(p, min_n=2)):
            return "ngram_loop"
        compact = "".join(p.split())
        if not compact or sum(x.isalpha() for x in compact)/len(compact) < .30:
            return "low_letter"
        if any(line.lstrip().startswith("- ") for line in p.split("\n")):
            return "bullet_collision"
        for j in range(i+1,len(picks)):
            a,b = p.strip(),picks[j].strip()
            if a.startswith(b) or b.startswith(a):
                return "prefix_redundant"
            union = k[i] | k[j]
            if union and len(k[i] & k[j])/len(union) >= .45:
                return "jaccard>=0.45"
    return ""


def assemble(selection, bank, tokenizer, min_picks=4, filter_policy="upstream"):
    if filter_policy not in FILTER_POLICIES:
        raise ValueError(f"Unknown assembly filter policy: {filter_policy}")
    accepted, audit, seen = [], [], set()
    split = selection["split"]
    for entry in selection["rows"]:
        idx = int(entry["item_idx"])
        if idx in seen or not 0 <= idx < len(bank[split+"_vec"]):
            raise ValueError(f"Duplicate/out-of-range item_idx {idx}")
        seen.add(idx)
        picks = entry.get("omp_sel") or []
        texts = [p["text"].lstrip(LEAD) for p in picks]
        strict_reason = reject_reason(texts,int(entry["n_candidates"]),min_picks)
        reason = (strict_reason if filter_policy == "upstream"
                  else reject_reason(texts,int(entry["n_candidates"]),min_picks,
                                     filter_policy=filter_policy))
        audit.append({"item_idx":idx,"reason":reason or "accepted",
                      "upstream_reason":strict_reason or "accepted",
                      "n_candidates":entry["n_candidates"],"n_picks":len(picks)})
        if reason:
            continue
        ordered = sorted(zip(picks,texts),key=lambda z: (int(z[0]["len"]),len(z[1])))
        target = "\n".join("- "+txt.strip() for _,txt in ordered)
        tokens = tokenizer(target,add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
        if not tokens or any(not isinstance(t,int) for t in tokens):
            raise ValueError("Bad target tokenization")
        accepted.append({"item_idx":idx,"target_ids":tokens,"text":target,
                         "src_pair_row":int(bank[split+"_pair_rows"][idx]),
                         "vec":bank[split+"_vec"][idx].float()})
    return accepted,audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection",type=Path,required=True)
    p.add_argument("--items",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--min-picks",type=int,choices=(3,4),default=4)
    p.add_argument("--filter-policy",choices=FILTER_POLICIES,default="upstream",
                   help="upstream = original 1/2/3-gram rule; "
                        "no_unigram = exploratory 2/3-gram-only variant")
    a=p.parse_args()
    if a.output.exists():
        p.error("Refusing existing output directory")
    if not (a.selection/"COMPLETE").is_file() or not (a.items/"COMPLETE").is_file():
        p.error("Incomplete selection or true-activation bank")
    sel_path=a.selection/"report.json"
    source_path=a.items/"items.safetensors"
    selection=json.loads(sel_path.read_text())
    info=json.loads((a.items/"meta.json").read_text())
    if (selection.get("schema")!=f"l{info['layer']}_nnomp_smoke_v1"
        or selection.get("source")!="true_GT_residual"
        or selection.get("items_sha256")!=info["items_sha256"]
        or selection.get("layer")!=info["layer"]
        or selection.get("split") not in ("train","val")):
        p.error("NNOMP/GT source mismatch")
    if digest(source_path)!=info["items_sha256"]:
        p.error("GT bank checksum mismatch")
    teacher=Path(info["teacher"])
    if not (teacher/"COMPLETE").is_file():
        p.error("Teacher incomplete")
    teacher_meta=json.loads((teacher/"meta.json").read_text())
    if teacher_meta.get("model")!=info["model"] or teacher_meta.get("layer")!=info["layer"]:
        p.error("Teacher model/layer mismatch")
    bank=load_file(str(source_path))
    tokenizer=AutoTokenizer.from_pretrained(teacher/"tokenizer")
    if tokenizer.eos_token_id is None:
        p.error("Tokenizer has no EOS")
    rows,audit=assemble(selection,bank,tokenizer,a.min_picks,
                        filter_policy=a.filter_policy)
    counts=Counter(item["reason"] for item in audit)
    upstream_counts=Counter(item["upstream_reason"] for item in audit)
    meta={"schema":f"l{info['layer']}_distill_bullets_v1","model":info["model"],
          "layer":selection["layer"],"split":selection["split"],
          "teacher":str(teacher),"source":"true_GT_residual",
          "prompt_mode":"concepts_raw","min_picks":a.min_picks,
          "filter_policy":a.filter_policy,
          "filter_policy_description":(
              "upstream original: 1-, 2-, or 3-gram repeats >=3 times"
              if a.filter_policy == "upstream"
              else "exploratory no-unigram: only 2-/3-gram repeats >=3 times"),
          "items_path":str(a.items.resolve()),
          "selection_sha256":digest(sel_path),"items_sha256":info["items_sha256"],
          "ar_checkpoint":selection["ar_checkpoint"],
          "input_rows":len(audit),"kept_rows":len(rows),
          "reject_counts":dict(sorted(counts.items())),
          "upstream_reject_counts":dict(sorted(upstream_counts.items())),
          "newly_accepted_vs_upstream":sum(
              row["reason"] == "accepted" and row["upstream_reason"] != "accepted"
              for row in audit),
          "status":"assembly only, not an independently tested student"}
    a.output.mkdir(parents=True)
    (a.output/"audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    if rows:
        flat=[t for row in rows for t in row["target_ids"]]
        offsets=[0]
        for row in rows:
            offsets.append(offsets[-1]+len(row["target_ids"]))
        payload={"target_ids":torch.tensor(flat,dtype=torch.int32),
                 "offsets":torch.tensor(offsets,dtype=torch.int64),
                 "vec":torch.stack([row["vec"] for row in rows]),
                 "src_pair_row":torch.tensor([row["src_pair_row"] for row in rows],dtype=torch.int64),
                 "src_item_idx":torch.tensor([row["item_idx"] for row in rows],dtype=torch.int64)}
        staging=a.output/"targets.partial.safetensors"
        save_file(payload,str(staging))
        os.replace(staging,a.output/"targets.safetensors")
        meta["targets_sha256"]=digest(a.output/"targets.safetensors")
        (a.output/"previews.json").write_text(json.dumps(
            [{"item_idx":row["item_idx"],"text":row["text"]} for row in rows],
            ensure_ascii=False,indent=2)+"\n")
    (a.output/"meta.json").write_text(json.dumps(meta,indent=2)+"\n")
    (a.output/"COMPLETE").write_text("ok\n")
    print(json.dumps(meta,indent=2),flush=True)

if __name__=="__main__":
    main()

