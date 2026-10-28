#!/usr/bin/env python3
"""Sample multiple teacher continuations conditioned on true single-layer residuals.

Follows Agam's distillation teacher sampling shape: GT activations, 64 draws,
temperature=1, top_p=.95, max_new_tokens=100. Deliberately uses THIS teacher's
own prompt and unit/alpha contract, rather than substituting his 27B contract.
Each item is saved atomically; existing completed items are skipped on resume.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_adapter_live(model, adapter_path):
    """Guard against a quietly inert PEFT load, as reported upstream."""
    from safetensors.torch import load_file
    saved = load_file(str(adapter_path))
    def canon(k):
        return k.replace(".default.", ".").replace("._orig_mod.", ".")
    live = {canon(k): v for k,v in model.state_dict().items() if "lora_" in k}
    matched = [k for k,v in saved.items() if canon(k) in live and
               live[canon(k)].shape == v.shape and
               torch.allclose(live[canon(k)].float().cpu(), v.float().cpu(), rtol=1e-3, atol=1e-4)]
    coverage = len(matched) / max(1,len(saved))
    if coverage < 0.9:
        raise RuntimeError(f"Live LoRA tensor coverage {len(matched)}/{len(saved)}={coverage:.1%}, refusing teacher sampling")
    print(f"adapter verified: {len(matched)}/{len(saved)} tensors live",flush=True)


def load_contract(teacher_path, items_path, split, start, count, n_samples, max_new, sample_batch, seed):
    if not (teacher_path/"COMPLETE").exists() or not (items_path/"COMPLETE").exists():
        raise ValueError("Teacher or true-activation items incomplete")
    teacher = json.loads((teacher_path/"meta.json").read_text())
    items = json.loads((items_path/"meta.json").read_text())
    if teacher.get("stage") != f"l{teacher.get('layer')}_ao_sft_fresh_pool_v1":
        raise ValueError("Wrong teacher checkpoint lineage")
    if items["teacher"] != str(teacher_path) or items["model"] != teacher["model"] or items["layer"] != teacher["layer"]:
        raise ValueError("Teacher/GT source mismatch")
    if items["teacher_alpha"] != teacher["alpha"] or items["prompt"] != teacher["prompt"]:
        raise ValueError("Teacher injection contract mismatch")
    if teacher.get("transform")!="unit" or teacher.get("prompt_slot")!="last token of rendered prompt":
        raise ValueError("Unexpected teacher injection format")
    if split not in ("train","val") or start < 0 or count < 1 or start+count > items[split+"_rows"]:
        raise ValueError("Invalid split/range")
    if min(n_samples,max_new,sample_batch) < 1:
        raise ValueError("Invalid sampling parameters")
    if sha256(items_path/"items.safetensors") != items["items_sha256"]:
        raise ValueError("GT item file hash mismatch")
    adapter = teacher_path/"lora"/"adapter_model.safetensors"
    if not adapter.exists():
        raise ValueError(f"Missing LoRA weights: {adapter}")
    manifest = {
        "format":"gt_distill_teacher_samples_v1",
        "checkpoint":str(teacher_path),
        "adapter_sha256":sha256(adapter),
        "items_path":str(items_path),
        "items_sha256":items["items_sha256"],
        "model":teacher["model"],"revision":teacher.get("revision"),
        "layer":teacher["layer"],"alpha":teacher["alpha"],
        "transform":"unit","prompt":teacher["prompt"],
        "slot":"last token of rendered prompt",
        "split":split,"start":start,"count":count,
        "samples_per_item":n_samples,"max_new_tokens":max_new,
        "sample_batch":sample_batch,"temperature":1.0,"top_p":.95,"seed":seed,
        "evaluation_status":"distill construction only, not independent test"
    }
    return teacher, items, manifest


@torch.inference_mode()
def generate_samples(model, tokenizer, prompt_ids, vector, alpha, n_samples, max_new, sample_batch, seed):
    embedding = model.get_input_embeddings()
    ids = prompt_ids[None,:].to(embedding.weight.device)
    pe = embedding(ids).clone()
    if vector.ndim != 1 or vector.numel() != pe.shape[-1]:
        raise ValueError("Input residual width doesn't match token embedding width")
    pe[:, -1, :] = (F.normalize(vector[None,:].float().to(pe.device),dim=-1,eps=1e-9)*alpha).to(pe.dtype)
    all_samples = []
    for base in range(0,n_samples,sample_batch):
        batch_size = min(sample_batch,n_samples-base)
        torch.manual_seed(seed+base)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed+base)
        embeds = pe.expand(batch_size,-1,-1).contiguous()
        mask = torch.ones(embeds.shape[:2], device=embeds.device, dtype=torch.long)
        generated = model.generate(
            inputs_embeds=embeds,attention_mask=mask,
            do_sample=True,temperature=1.0,top_p=.95,max_new_tokens=max_new,
            use_cache=True, pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        if generated.shape[0] != batch_size:
            raise RuntimeError(f"Unexpected generate batch shape: {tuple(generated.shape)}")
        for row in generated:
            toks = row.tolist()
            # With inputs_embeds-only generation, HF returns generated token IDs.
            all_samples.append({"ids":toks,"text":tokenizer.decode(toks,skip_special_tokens=True)})
    assert len(all_samples)==n_samples
    return all_samples


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher",type=Path,required=True)
    p.add_argument("--items",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--split",choices=["train","val"],default="train")
    p.add_argument("--start",type=int,default=0)
    p.add_argument("--count",type=int,default=8)
    p.add_argument("--n-samples",type=int,default=64)
    p.add_argument("--max-new-tokens",type=int,default=100)
    p.add_argument("--sample-batch",type=int,default=8)
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--dry-run",action="store_true")
    args=p.parse_args()
    teacher, items, manifest = load_contract(
        args.teacher,args.items,args.split,args.start,args.count,
        args.n_samples,args.max_new_tokens,args.sample_batch,args.seed)
    if args.dry_run:
        print(json.dumps(manifest,indent=2))
        print("DRY RUN: no model loaded, no files written")
        return
    if not torch.cuda.is_available():
        p.error("CUDA required for teacher sampling")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path=args.output/"manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            p.error("Existing sampling manifest differs: refusing to mix experiments")
    else:
        if list(args.output.iterdir()):
            p.error("Existing nonempty directory without manifest")
        tmp=args.output/"manifest.json.tmp"
        tmp.write_text(json.dumps(manifest,indent=2)+"\n")
        os.replace(tmp,manifest_path)
    tensors=load_file(str(args.items/"items.safetensors"))
    vecs=tensors[args.split+"_vec"]
    pair_rows=tensors[args.split+"_pair_rows"]
    tokenizer=AutoTokenizer.from_pretrained(args.teacher/"tokenizer")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token=tokenizer.eos_token
    # Explicit cache_dir avoids stale HF_HOME pointing to the FUSE volume.
    # Override only if intentionally provisioning another local model cache.
    model_cache = os.environ.get("OLENS_HF_CACHE_DIR", "/root/.cache/huggingface/hub")
    base=AutoModelForCausalLM.from_pretrained(teacher["model"],revision=teacher.get("revision"),
                                             dtype=torch.bfloat16,device_map="cuda",
                                             cache_dir=model_cache)
    base.config.use_cache=True
    model=PeftModel.from_pretrained(base,args.teacher/"lora",is_trainable=False).eval()
    check_adapter_live(model,args.teacher/"lora"/"adapter_model.safetensors")
    prompts=torch.tensor(tokenizer(teacher["prompt"],add_special_tokens=False).input_ids,dtype=torch.long)
    if not len(prompts):
        raise RuntimeError("Empty teacher prompt")
    print(f"teacher α={teacher['alpha']}, layer={teacher['layer']}, slot=last prompt token, sample={args.n_samples}x{args.max_new_tokens}",flush=True)
    for i in range(args.start,args.start+args.count):
        dest=args.output/f"item_{i:06d}.json"
        if dest.exists():
            prior=json.loads(dest.read_text())
            if prior.get("pair_row") != int(pair_rows[i]) or len(prior.get("samples",[]))!=args.n_samples:
                raise RuntimeError(f"Inconsistent prior item: {dest}")
            print(f"skip existing item {i}",flush=True)
            continue
        generated=generate_samples(model,tokenizer,prompts,vecs[i],
                                   teacher["alpha"],args.n_samples,args.max_new_tokens,
                                   args.sample_batch,args.seed+i*1000003)
        payload={"item_idx":i,"split":args.split,"pair_row":int(pair_rows[i]),
                 "n_samples":len(generated),"samples":generated}
        temp=dest.with_suffix(".json.partial")
        temp.write_text(json.dumps(payload,ensure_ascii=False)+"\n")
        os.replace(temp,dest)
        print(f"item={i} samples={len(generated)} preview={generated[0]['text'][:120]!r}",flush=True)
    print("SAMPLING RANGE COMPLETE",args.start,args.start+args.count,flush=True)


if __name__=="__main__":
    main()
