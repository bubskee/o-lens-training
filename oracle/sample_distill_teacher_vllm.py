#!/usr/bin/env python3
"""Bulk single-layer distillation teacher generation via vLLM prompt embeddings.

Run in /root/olens-vllm-venv, NOT the HF/PEFT training environment.
Outputs are byte-schema compatible with oracle/select_distill_nnomp.py.
Source activations are TRUE single-layer residuals; prompt, slot and alpha come from
the exact completed SFT checkpoint used for the merged teacher.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoTokenizer


def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def configure_vllm_multiprocessing():
    """Use CUDA-safe spawn; this CLI is protected by the __main__ guard."""
    mode = os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    if mode != "spawn":
        raise RuntimeError(
            "This sampler requires VLLM_WORKER_MULTIPROC_METHOD=spawn "
            "to prevent CUDA reinitialization in forked workers"
        )


def shard_bounds(start,count,n_shards,shard):
    if n_shards < 1 or not 0 <= shard < n_shards or count < n_shards:
        raise ValueError("Need 1 <= n_shards <= count and 0 <= shard < n_shards")
    lo=start + count*shard//n_shards
    hi=start + count*(shard+1)//n_shards
    return lo,hi


def contract(teacher,items,merged,split,start,count,n_shards,shard,
             n_samples,max_new,chunk,seed,max_num_seqs):
    if min(n_samples,max_new,chunk,max_num_seqs) < 1:
        raise ValueError("Invalid positive sampling parameter")
    if not (teacher/"COMPLETE").is_file() or not (items/"COMPLETE").is_file():
        raise ValueError("Incomplete SFT teacher or true-activation bank")
    tm=json.loads((teacher/"meta.json").read_text())
    im=json.loads((items/"meta.json").read_text())
    mm=json.loads((merged/"olens_merge_manifest.json").read_text())
    if tm.get("stage")!=f"l{tm.get('layer')}_ao_sft_fresh_pool_v1" or im.get("schema")!="gt_distill_items_v1":
        raise ValueError("Wrong source lineage")
    if (mm.get("schema")!=f"l{tm.get('layer')}_vllm_merged_teacher_v1"
        or Path(mm["source_checkpoint"]).resolve()!=teacher.resolve()
        or mm["source_adapter_sha256"]!=sha256(teacher/"lora"/"adapter_model.safetensors")
        or mm["model"]!=tm["model"] or im["model"]!=tm["model"]
        or mm["layer"]!=tm["layer"] or im["layer"]!=tm["layer"]
        or mm["transform"]!="unit" or tm["transform"]!="unit"
        or mm["prompt"]!=tm["prompt"] or im["prompt"]!=tm["prompt"]
        or mm["prompt_slot"]!=tm["prompt_slot"]
        or mm["alpha"]!=tm["alpha"] or im["teacher_alpha"]!=tm["alpha"]
        or Path(im["teacher"]).resolve()!=teacher.resolve()):
        raise ValueError("Merged teacher / true-activation contract mismatch")
    if sha256(items/"items.safetensors")!=im["items_sha256"]:
        raise ValueError("GT item artifact SHA mismatch")
    if split not in ("train","val") or start<0 or count<1 or start+count>im[split+"_rows"]:
        raise ValueError("Invalid sample range")
    lo,hi=shard_bounds(start,count,n_shards,shard)
    manifest={"format":"gt_distill_teacher_samples_v1", # downstream NNOMP contract
        "engine":"vllm_offline_prompt_embeds","merged_manifest_sha256":sha256(merged/"olens_merge_manifest.json"),
        "checkpoint":str(teacher),"adapter_sha256":mm["source_adapter_sha256"],
        "items_path":str(items),"items_sha256":im["items_sha256"],
        "model":tm["model"],"revision":tm.get("revision"),
        "layer":tm["layer"],"alpha":tm["alpha"],"transform":"unit",
        "prompt":tm["prompt"],"slot":tm["prompt_slot"],
        "split":split,"start":lo,"count":hi-lo,"overall_start":start,
        "overall_count":count,"shard":shard,"n_shards":n_shards,
        "samples_per_item":n_samples,"max_new_tokens":max_new,
        "sample_batch":n_samples,"chunk_items":chunk,"max_num_seqs":max_num_seqs,
        "temperature":1.0,"top_p":.95,"seed":seed,
        "evaluation_status":"distill construction only, not independent test"}
    return tm,im,mm,manifest


def read_embedding(merged,merge_meta):
    """Read only token embeddings from the indexed merged safetensors model."""
    index=merged/"model.safetensors.index.json"
    key=merge_meta["embedding_key"]
    if index.exists():
        filename=json.loads(index.read_text())["weight_map"][key]
    else:
        filename="model.safetensors"
    if filename not in merge_meta["weights"]:
        raise ValueError("Embedding shard absent from merge manifest")
    shard=merged/filename
    if sha256(shard)!=merge_meta["weights"][filename]:
        raise ValueError("Merged embedding shard checksum mismatch")
    with safe_open(str(shard),framework="pt",device="cpu") as file:
        weight=file.get_tensor(key)
    if weight.ndim!=2 or weight.shape[1]!=2560:
        raise ValueError(f"Expected Qwen3.5-4B input embedding width 2560; got {tuple(weight.shape)}")
    return weight


def inject_prompt(weight,prompt_ids,activation,alpha):
    """Exactly replace the last prompt embedding, with unit-L2 residual."""
    ids=torch.tensor(prompt_ids,dtype=torch.long)
    if not ids.numel() or int(ids.max())>=weight.shape[0]:
        raise ValueError("Invalid prompt IDs")
    if activation.ndim!=1 or activation.numel()!=weight.shape[1]:
        raise ValueError("Activation dimension mismatch")
    if not bool(torch.isfinite(activation).all()) or float(activation.float().norm())<1e-8:
        raise ValueError("Nonfinite or near-zero true residual")
    pe=weight[ids].clone()
    pe[-1]=(F.normalize(activation.float(),dim=0,eps=1e-9)*alpha).to(pe.dtype)
    return pe.contiguous()


def materialize_requests(indices,tensors,split,weight,prompt_ids,alpha):
    prompts=[]
    for i in indices:
        vec=tensors[split+"_vec"][i]
        prompts.append({"prompt_embeds":inject_prompt(weight,prompt_ids,vec,alpha)})
    return prompts


def sample_chunk(llm,params_cls,prompts,indices,n_samples,max_new,seed):
    """Purely adapt vLLM's ordered completion outputs to existing NNOMP JSON."""
    params=[params_cls(n=n_samples,temperature=1.0,top_p=.95,
                       max_tokens=max_new,seed=(seed+int(i)*1000003)%(2**31-1))
            for i in indices]
    generated=llm.generate(prompts,sampling_params=params,use_tqdm=False)
    if len(generated)!=len(indices):
        raise RuntimeError("vLLM output count differs from input count")
    results=[]
    for i,row in zip(indices,generated,strict=True):
        outputs=sorted(row.outputs,key=lambda x:x.index)
        if len(outputs)!=n_samples or [o.index for o in outputs]!=list(range(n_samples)):
            raise RuntimeError(f"Item {i}: expected {n_samples} indexed completions")
        samples=[]
        for o in outputs:
            ids=list(o.token_ids)
            if not ids or len(ids)>max_new or not all(isinstance(x,int) for x in ids):
                raise RuntimeError(f"Item {i}: invalid generated token IDs")
            samples.append({"ids":ids,"text":o.text})
        results.append(samples)
    return results


def atomic_json(path,content):
    tmp=path.with_suffix(".json.partial")
    if tmp.exists():
        raise RuntimeError(f"Partial output exists; inspect before reattempt: {tmp}")
    tmp.write_text(json.dumps(content,ensure_ascii=False)+"\n")
    os.replace(tmp,path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher",type=Path,required=True)
    p.add_argument("--items",type=Path,required=True)
    p.add_argument("--merged",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--split",choices=("train","val"),default="train")
    p.add_argument("--start",type=int,default=0)
    p.add_argument("--count",type=int,default=2)
    p.add_argument("--n-shards",type=int,default=1)
    p.add_argument("--shard",type=int,default=0)
    p.add_argument("--n-samples",type=int,default=64)
    p.add_argument("--max-new-tokens",type=int,default=100)
    p.add_argument("--chunk-items",type=int,default=2)
    p.add_argument("--max-num-seqs",type=int,default=128)
    p.add_argument("--gpu-memory-utilization",type=float,default=.85)
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--language-model-only",action="store_true",
                   help="Use vLLM's text-only Qwen3.5 model mode, when supported")
    a=p.parse_args()
    tm,im,mm,manifest=contract(a.teacher,a.items,a.merged,a.split,
        a.start,a.count,a.n_shards,a.shard,a.n_samples,a.max_new_tokens,
        a.chunk_items,a.seed,a.max_num_seqs)
    print(json.dumps(manifest,indent=2),flush=True)
    if a.dry_run:
        print("DRY RUN: no model loaded or output created",flush=True)
        return
    # Do not call torch.cuda.is_available() before launching vLLM. Depending
    # on the PyTorch/runtime check, that can initialize CUDA in the parent
    # and poison the default forked engine process. The engine itself checks
    # the availability of a suitable CUDA device.
    configure_vllm_multiprocessing()
    if not 0 < a.gpu_memory_utilization <= 1:
        p.error("GPU memory fraction must be in (0,1]")
    a.output.mkdir(parents=True,exist_ok=True)
    destination=a.output/"manifest.json"
    if destination.exists():
        if json.loads(destination.read_text())!=manifest:
            p.error("Existing sampling manifest differs; refusing mixed run")
    else:
        if any(a.output.iterdir()):
            p.error("Nonempty destination without manifest")
        atomic_json(destination,manifest)
    data=load_file(str(a.items/"items.safetensors"))
    lo,hi=manifest["start"],manifest["start"]+manifest["count"]
    if (a.output/"COMPLETE").is_file():
        for i in range(lo,hi):
            file=a.output/f"item_{i:06d}.json"
            if not file.is_file():
                raise RuntimeError(f"COMPLETE but missing item {i}")
            saved=json.loads(file.read_text())
            if (saved.get("item_idx")!=i or saved.get("split")!=a.split or
                saved.get("pair_row")!=int(data[a.split+"_pair_rows"][i]) or
                len(saved.get("samples",[]))!=a.n_samples):
                raise RuntimeError(f"COMPLETE but inconsistent item {i}")
        print("ALL ITEMS ALREADY COMPLETE; no vLLM engine start",flush=True)
        return
    # Verify every merged weight shard, not only the embedding shard, before
    # bringing up the engine; vLLM must read exactly the audited merged model.
    for filename,expected in mm["weights"].items():
        if sha256(a.merged/filename)!=expected:
            raise RuntimeError(f"Merged model weight SHA mismatch: {filename}")
    tokenizer=AutoTokenizer.from_pretrained(a.teacher/"tokenizer")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token=tokenizer.eos_token
    prompt_ids=tokenizer(tm["prompt"],add_special_tokens=False).input_ids
    merged_tok=AutoTokenizer.from_pretrained(a.merged)
    if merged_tok(tm["prompt"],add_special_tokens=False).input_ids!=prompt_ids:
        p.error("Merged model and teacher tokenize prompt differently")
    weight=read_embedding(a.merged,mm)
    # A vLLM prompt_embeds request holds the ENTIRE already-rendered prompt.
    if len(prompt_ids)+a.max_new_tokens>2048:
        p.error("Max sequence length exceeds 2048")
    from vllm import LLM, SamplingParams
    kwargs={"model":str(a.merged),"enable_prompt_embeds":True,
            "tensor_parallel_size":1,"max_model_len":2048,
            "max_num_seqs":a.max_num_seqs,
            "gpu_memory_utilization":a.gpu_memory_utilization,
            "dtype":"bfloat16","seed":a.seed}
    if a.language_model_only:
        kwargs["language_model_only"]=True
    start_loading=time.monotonic()
    engine=LLM(**kwargs)
    load_seconds=time.monotonic()-start_loading
    print(f"vLLM loaded in {load_seconds:.1f}s; prompt tokens={len(prompt_ids)}",flush=True)
    lo,hi=manifest["start"],manifest["start"]+manifest["count"]
    gen_tokens=0
    generation_seconds=0.0
    resumed=0
    for chunk_start in range(lo,hi,a.chunk_items):
        chunk_end=min(hi,chunk_start+a.chunk_items)
        indices=list(range(chunk_start,chunk_end))
        pending=[]
        for i in indices:
            dest=a.output/f"item_{i:06d}.json"
            if dest.exists():
                prior=json.loads(dest.read_text())
                if (prior.get("item_idx")!=i or prior.get("split")!=a.split
                    or prior.get("pair_row")!=int(data[a.split+"_pair_rows"][i])
                    or len(prior.get("samples",[]))!=a.n_samples):
                    raise RuntimeError(f"Existing item mismatches source: {dest}")
                resumed+=1
            else:
                pending.append(i)
        if not pending:
            print(f"skip complete items {chunk_start}:{chunk_end}",flush=True)
            continue
        prompts=materialize_requests(pending,data,a.split,weight,prompt_ids,tm["alpha"])
        t0=time.monotonic()
        generated=sample_chunk(engine,SamplingParams,prompts,pending,
                               a.n_samples,a.max_new_tokens,a.seed)
        elapsed=time.monotonic()-t0
        generation_seconds+=elapsed
        for i,samples in zip(pending,generated,strict=True):
            dest=a.output/f"item_{i:06d}.json"
            atomic_json(dest,{"item_idx":i,"split":a.split,
                              "pair_row":int(data[a.split+"_pair_rows"][i]),
                              "n_samples":len(samples),"samples":samples})
            gen_tokens+=sum(len(row["ids"]) for row in samples)
        print(f"items {chunk_start}:{chunk_end}, new={len(pending)}, "
              f"time={elapsed:.1f}s tokens={sum(sum(len(x['ids']) for x in g) for g in generated)} "
              f"throughput={sum(sum(len(x['ids']) for x in g) for g in generated)/max(elapsed,1e-9):.1f}tok/s",
              flush=True)
    report={"status":"complete","engine":"vllm","loaded_seconds":load_seconds,
            "generated_seconds":generation_seconds,"new_tokens":gen_tokens,
            "new_tokens_per_second":gen_tokens/max(generation_seconds,1e-9),
            "already_complete_items":resumed,"items":hi-lo}
    atomic_json(a.output/"throughput.json",report)
    (a.output/"COMPLETE").write_text("ok\n")
    print("BULK TEACHER COMPLETE",json.dumps(report),flush=True)


if __name__=="__main__":
    main()
