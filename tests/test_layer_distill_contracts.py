"""Exercise actual pure contract functions without importing GPU dependencies."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

def functions(filename, names):
    tree = ast.parse((ROOT / "oracle" / filename).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = dict(Path=Path, json=json, hashlib=hashlib)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec"), ns)
    return ns

bulk = functions("sample_distill_teacher_vllm.py", {"sha256", "shard_bounds", "contract"})
contract = bulk["contract"]

class LayerContracts(unittest.TestCase):
    def check_layer(self, layer):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            teacher=root/"teacher"
            items=root/"items"
            merged=root/"merged"
            (teacher/"lora").mkdir(parents=True)
            items.mkdir()
            merged.mkdir()
            (teacher/"COMPLETE").write_text("ok")
            (items/"COMPLETE").write_text("ok")
            adapter=b"adapterbytes"
            (teacher/"lora"/"adapter_model.safetensors").write_bytes(adapter)
            (items/"items.safetensors").write_bytes(b"itemsbytes")
            tm={"stage":f"l{layer}_ao_sft_fresh_pool_v1","model":"Qwen/Qwen3.5-4B",
                "revision":"revision","layer":layer,"alpha":8000.,
                "prompt":"Read an activation!","prompt_slot":"last token of rendered prompt",
                "transform":"unit"}
            im={"schema":"gt_distill_items_v1","model":tm["model"],"layer":layer,
                "teacher":str(teacher),"teacher_alpha":8000.,
                "prompt":tm["prompt"],"train_rows":40,"val_rows":10,
                "items_sha256":hashlib.sha256(b"itemsbytes").hexdigest()}
            mm={"schema":f"l{layer}_vllm_merged_teacher_v1",
                "source_checkpoint":str(teacher),
                "source_adapter_sha256":hashlib.sha256(adapter).hexdigest(),
                "model":tm["model"],"layer":layer,"transform":"unit",
                "prompt":tm["prompt"],"prompt_slot":tm["prompt_slot"],"alpha":8000.}
            (teacher/"meta.json").write_text(json.dumps(tm))
            (items/"meta.json").write_text(json.dumps(im))
            (merged/"olens_merge_manifest.json").write_text(json.dumps(mm))
            _,_,_,manifest=contract(teacher,items,merged,"train",0,40,4,2,64,100,2,5,128)
            self.assertEqual((manifest["start"],manifest["count"]),(20,10))
            self.assertEqual(manifest["format"],"gt_distill_teacher_samples_v1")
            self.assertEqual(manifest["samples_per_item"],64)
            mm["layer"] = 30 if layer == 26 else 26
            (merged/"olens_merge_manifest.json").write_text(json.dumps(mm))
            with self.assertRaisesRegex(ValueError,"contract mismatch"):
                contract(teacher,items,merged,"train",0,40,4,2,64,100,2,5,128)
            mm["layer"] = layer
            mm["alpha"]=16000.
            (merged/"olens_merge_manifest.json").write_text(json.dumps(mm))
            with self.assertRaisesRegex(ValueError,"contract mismatch"):
                contract(teacher,items,merged,"train",0,40,4,2,64,100,2,5,128)

    def test_l26_compatibility(self):
        self.check_layer(26)

    def test_l30_and_cross_layer_rejection(self):
        self.check_layer(30)

    def test_student_l30_parent_and_mismatch(self):
        ns = functions("train_distill_student.py", {"sha256", "verified_contract"})
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root/"lora").mkdir()
            (root/"COMPLETE").write_text("ok")
            (root/"lora/adapter_model.safetensors").write_bytes(b"adapter")
            parent = dict(stage="l30_ao_sft_fresh_pool_v1", layer=30,
                model="Qwen/Qwen3.5-4B", source="frozen_AR_of_fresh_nonoverlap_AO_crops",
                transform="unit", alpha=8000, prompt_slot="last token of rendered prompt")
            (root/"meta.json").write_text(json.dumps(parent))
            meta = dict(model=parent["model"], layer=30, teacher=str(root), items_sha256="same")
            ns["load_targets"] = lambda path, split: (dict(meta), {"src_pair_row":[1 if split=="train" else 2]})
            self.assertEqual(ns["verified_contract"](root/"tr", root/"va", root)[4]["layer"], 30)
            meta["layer"] = 26
            with self.assertRaisesRegex(ValueError, "Target/parent mismatch"):
                ns["verified_contract"](root/"tr", root/"va", root)

if __name__ == "__main__":
    unittest.main()
