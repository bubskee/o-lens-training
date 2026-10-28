# L30 distillation handoff

AO teacher completed at step 500: validation token CE 2.1521184924433623,
63,031 supervised tokens. Source pool: 24,973 train / 2,679 validation crops.
Pool SHA256: d31a96b606f5ec105422835f51ea73d49cc087bc7b8f124c7d0a5749200a8cbd.
AR-output SHA256: 2f476021d969459a9b357a9bcfdf7ed6bea445dd10cb31e254485893fa2d4555.
These are user-reported completed artifacts on /workspace.

Reuse L26's algorithms with L30-specific contracts, prompt, teacher, vectors,
and frozen AR. No new harvest yet. Start with 128 training / 32 validation raw
items, one per eligible AR-held-out rollout, fixed seed 42. This is a small
pipeline pilot, not a matched comparison with L26's 2,928-target student.
Capacity is checked before writing; do not reduce counts silently on failure.
The separate pain-axis examples remain outside fitting.

All commands run from /root/o-lens-training after merging/pulling.
Use `set -o pipefail`; tee each stage to /workspace/o-lens-qwen35-4b/logs.
Stages are explicit, not automatically chained:

1. `bash scripts/run_l30_distill.sh capacity` (CPU; no artifacts).
2. `bash scripts/run_l30_distill.sh prepare` (CPU; new true-residual bank).
3. `bash scripts/run_l30_distill.sh merge` (GPU; ephemeral /root merged base).
4. `bash scripts/run_l30_distill.sh sample-smoke train` (vLLM, two items).
   Inspect completion and candidate texts before bulk.
5. `bash scripts/run_l30_distill.sh sample train`, then `sample val`.
6. `bash scripts/run_l30_distill.sh select train`, then `select val`.
7. `bash scripts/run_l30_distill.sh assemble train`, then `assemble val`.
   Inspect accepted counts; empty assembly cannot train.
8. `bash scripts/run_l30_distill.sh student-dry`, then `student-smoke`.

Sampling retains 64 draws, T=1, top-p=.95, 100-token cap. NNOMP selects up to
four prefixes using the frozen L30 whitener; assembly retains four with the
settled no_unigram policy. L30 NNOMP now freezes adapter/head hashes at selection.
Student uses its layer-specific concepts_raw prompt, unit/alpha8000, LR5e-5.

Do not run the L26 workout or HF exporter on L30: their release/evaluation
contracts remain L26-specific and are a subsequent task. No long student run is
launched by this runner. GPU validation of this adaptation is still pending.
