# Profiling CLIP on AWS Inferentia2

Compile OpenAI's [CLIP](https://openai.com/research/clip) (`openai/clip-vit-base-patch32`) to an AWS Inferentia2 (Inf2) accelerator, capture a hardware execution trace, and read the timeline to find where inference time actually goes.

The goal is a concrete systems story: how a dynamic PyTorch model becomes a statically-compiled accelerator binary, and what GPU profiling intuition does — and doesn't — transfer to compiled ML hardware.

## Why CLIP, and why Neuron

CLIP is a dual-encoder model. It takes an image and a list of text captions and scores which caption matches by computing cosine similarity in a shared embedding space (it is not generative).

- **Vision encoder:** ViT-B/32 — 12 transformer layers, 32×32 patches, 224×224 input → 512-dim embedding (~87M params)
- **Text encoder:** 12-layer BERT-like transformer, BPE tokenization, max 77 tokens → 512-dim embedding (~63M params)
- **Total:** ~151M params, ~600MB on disk

Inferentia2 differs from a GPU in the way that matters for profiling: it uses a **systolic array** (like a TPU), and inference is **ahead-of-time (AOT) compiled**. `neuronx-cc` lowers the model to a NEFF (NeuronCore Executable File Format) binary with a static, baked-in instruction schedule — there is no dynamic runtime scheduler like CUDA. The profiler shows you *what the compiler decided*, not what a runtime scheduler chose at execution time.

## Hardware

| | |
|---|---|
| Instance | `inf2.xlarge` |
| Accelerator | 1 Inferentia2 chip, **2 NeuronCores**, 32 GB HBM |
| AMI | Deep Learning AMI Neuron (Ubuntu) — ships `torch_neuronx`, `neuronx-cc`, Neuron profiler |
| Storage | 100 GB gp3 (room for weights, NEFF, and profile output) |

## Scripts

### `clip_profile.py` — compile + profiled run

Compiles CLIP to a single NEFF graph, runs warmup + 5 profiled iterations, and prints CLIP similarity scores. Writes `.ntff` hardware trace files for the Neuron profile viewer.

```bash
python3 clip_profile.py
```

Key details baked into the script:
- `NEURON_PROFILE` is set **before** `import torch_neuronx` — the runtime reads it at initialization. Setting it after import is ignored.
- `return_dict=False` — XLA tracing can't handle dataclass/dict outputs; the model must return plain tuples.
- Trace signature is `(input_ids, pixel_values)` — no `attention_mask` (AWS's exact pattern).
- Compiler flag `--enable-saturate-infinity` prevents NaN from `inf * 0` in attention softmax.
- Text is padded with `padding="max_length"` to exactly 77 tokens so the runtime shape matches the compiled shape (`[4, 77]`). A static NEFF rejects any shape mismatch.

The compiled NEFF is saved to `./clip_neuron_compiled/clip_neuron.pt`; traces land in `./clip_profile_results/`.

### `benchmark_clip_baseline.py` — CPU vs Neuron sweep

Compiles a separate static NEFF per image batch size and benchmarks CPU PyTorch against compiled Neuron execution. Each batch size is a distinct compiled artifact — that is expected on AOT hardware.

```bash
python3 benchmark_clip_baseline.py --batches 1 2 4 8 16 --cpu-iters 20 --neuron-iters 100
```

Writes `./clip_baseline_results.csv` with p50/p90/p99 latency, throughput, speedup, and `max_abs_logit_diff` (CPU-vs-Neuron output drift). It pops `NEURON_PROFILE` so profiling overhead doesn't distort the latency numbers.

Sample results:

| image batch | CPU p50 (ms) | Neuron p50 (ms) | speedup | max abs logit diff |
|---|---|---|---|---|
| 1 | 252.8 | 14.7 | 17.9× | 1.9e-05 |
| 2 | 317.1 | 4.8 | 66.1× | 2.7e-05 |
| 4 | 426.8 | 6.5 | 65.9× | 2.6e-05 |

Batch 1 has the worst Neuron latency because fixed runtime overhead isn't amortized. The tiny logit drift (~1e-05) confirms the compiled model preserves outputs.

## Setup on the instance

```bash
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
export PATH=/opt/aws/neuron/bin:$PATH

# The Neuron venv ships torch_neuronx but not HuggingFace transformers
pip install transformers requests Pillow -q

# Confirm the hardware: 1 device, 2 cores, 32 GB
neuron-ls
```

## Viewing the profile

The profile viewer needs an **InfluxDB 1.x** backend (line protocol + InfluxQL), which the AMI does not ship:

```bash
wget https://dl.influxdata.com/influxdb/releases/influxdb_1.8.10_amd64.deb
sudo dpkg -i influxdb_1.8.10_amd64.deb
sudo systemctl enable --now influxdb
```

`neuron-profile` is deprecated in favor of `neuron-explorer`. Start the viewer on the instance (it must keep running):

```bash
neuron-explorer view \
  -n ./clip_profile_results/<id>.neff \
  -s ./clip_profile_results/<id>-1.ntff \
  --display-name clip-vit-b32-bs4 \
  --port 3001
```

Then tunnel **both** ports from your laptop (3001 = UI, 3002 = profile API):

```powershell
ssh -i "C:\Users\varun\.ssh\neuron-key.pem" -N `
  -L 3001:localhost:3001 -L 3002:localhost:3002 `
  ubuntu@<current-public-ip>
```

Open `http://localhost:3001/`.

## What the timeline answers

- Which encoder is the bottleneck — vision or text?
- Is there an idle sync gap between NeuronCore 0 and 1? (On CUDA both branches share one scheduler, so this gap is invisible — on Neuron it shows up explicitly.)
- What % of latency is the similarity op?
- At what batch size does throughput stop scaling — i.e. when does the systolic array saturate?

In one captured run, the device timeline showed **~15.96 ms** of device activity while the Python timer measured **~21.8 ms** end-to-end. That ~6 ms gap is host/runtime overhead (Python calls, runtime launch, I/O movement) — exactly the kind of thing a hardware profile surfaces that a wall-clock timer cannot.

> **Note on profiling overhead:** With `NEURON_PROFILE` enabled, per-iteration latency jumps from ~21 ms to ~550 ms because the runtime records nanosecond timestamps for every op. The ~21 ms figure is real production latency; the ~550 ms is instrumentation-only.

## Cost & lifecycle

- **Stop** (don't Terminate) when done — stopped ≈ $0.40/day for EBS storage only.
- Terminate deletes the compiled NEFF; recompiling costs 5–10 min. In production: compile once → upload NEFF to S3 → download on cold start.
- Public IP changes on every start (no Elastic IP) — note the new IP from the EC2 console after restarting.
- Total expected project cost: ~$6–8.

## Files

| File | Purpose |
|---|---|
| `clip_profile.py` | Compile CLIP → NEFF, run profiled inference, emit `.ntff` traces |
| `benchmark_clip_baseline.py` | CPU vs Neuron latency/throughput sweep across batch sizes |
| `CLIP_PROFILING_PLAN.txt` | The original 8–10 hour project plan |
| `WHAT_IM_DOING.md` | Live engineering log — every error, root cause, and fix |
| `BLOG_DRAFT.md` | Draft write-up of the findings |
