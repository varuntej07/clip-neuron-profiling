import os
import time
import torch
import torch_neuronx
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import requests

MODEL_ID = "openai/clip-vit-base-patch32"
COMPILED_DIR = "./clip_neuron_compiled"
PROFILE_DIR = "./clip_profile_results"
os.makedirs(COMPILED_DIR, exist_ok=True)
os.makedirs(PROFILE_DIR, exist_ok=True)

print("[1/5] Loading CLIP model...")

processor = CLIPProcessor.from_pretrained(MODEL_ID)

# return_dict=False is required for torch_neuronx.trace()
# XLA tracing cannot handle dict/dataclass outputs; with False the model returns plain tuples instead.
model = CLIPModel.from_pretrained(MODEL_ID, return_dict=False).eval()

# AOT (Ahead of time) compilation requires fixed input shapes known at compile time (unlike GPU JIT).
# AWS official example used (input_ids, pixel_values) — two inputs, no attention_mask.
# Shapes: [batch=4, seq_len=77] and [batch=4, channels=3, H=224, W=224].
dummy_input_ids = torch.zeros(4, 77, dtype=torch.long)
dummy_pixel_values = torch.zeros(4, 3, 224, 224)
# torch_neuronx.trace() runs a single forward pass to capture the computation graph as
# HLO (XLA's High Level Operations IR), then hands it to neuronx-cc (AOT compiler).
# neuronx-cc lowers HLO -> NEFF (NeuronCore Executable File Format): a static instruction
# stream scheduled for the systolic array. This bakes the op order into the binary.
# --enable-saturate-infinity: prevents NaN propagation from inf*0 in attention softmax.
# Passing model directly (not a wrapper)
# First compile: 5-10 min. Subsequent loads from .pt skip this entirely.
print("[2/5] Compiling to Neuron (5-10 min first run)...")
neuron_clip = torch_neuronx.trace(
    model,
    (dummy_input_ids, dummy_pixel_values),
    compiler_args="--enable-saturate-infinity"
)

# Serializes the TorchScript wrapper + embedded NEFF binary to disk.
# On reload, the NEFF bypasses recompilation entirely.
torch.jit.save(neuron_clip, os.path.join(COMPILED_DIR, "clip_neuron.pt"))
print("[2/5] Saved to:", COMPILED_DIR)

# Real JPEG images — PIL decodes into RGB arrays; processor normalizes to 224x224.
# Using COCO val2017 images: direct JPEG URLs so PIL can decode the stream.
print("[3/5] Fetching images and tokenizing text...")
img_urls = [
    "http://images.cocodataset.org/val2017/000000039769.jpg",  # cats
    "http://images.cocodataset.org/val2017/000000397133.jpg",  # train
]
images = [Image.open(requests.get(url, stream=True).raw).convert("RGB") for url in img_urls]
texts = [
    "a photo of a cat",
    "a photo of a train",
    "a photo of a dog",
    "two cats sleeping on a couch",
]

# processor tokenizes text (BPE, max 77 tokens) + normalizes images into pixel_values.
# images*2 repeats the 2 images to match batch=4.
inputs = processor(text=texts, images=images * 2,
                   return_tensors="pt", padding=True, truncation=True, max_length=77)
input_ids = inputs["input_ids"]       # [4, 77]
pixel_values = inputs["pixel_values"]    # [4, 3, 224, 224]

# First inference DMA-transfers the NEFF binary from EBS (host DRAM) into NeuronCore
# on-chip SRAM and fills the instruction pipeline. Subsequent calls skip this.
# Without warmup, iter-0 latency includes NEFF load time (~100-500ms) — an outlier.
print("[4/5] Warmup (loads NEFF onto NeuronCore SRAM)...")
with torch.no_grad():
    _ = neuron_clip(input_ids, pixel_values)

# Profiler instruments NeuronCore hardware counters for the duration of the context.
# profile_type="trace": records per-op hardware execution timeline as a .ntff file.
# .ntff captures which NeuronCore ran each op, start/end timestamps (ns), DMA events.
# 5 iterations: first 1-2 may still have pipeline fill artifacts; later iters are stable.
print("[5/5] Profiled run...")
with torch_neuronx.experimental.profiler.profile(
    profile_dir=PROFILE_DIR,
    profile_type="trace",
    neuron_profile_args="--nn-batch-size=1"
) as prof:
    with torch.no_grad():
        for i in range(5):
            t0 = time.perf_counter()
            output = neuron_clip(input_ids, pixel_values)
            t1 = time.perf_counter()
            print(f"  iter {i+1}: {(t1 - t0) * 1000:.1f} ms")

print(f"\nProfile saved to: {PROFILE_DIR}")
print(f"View timeline: neuron-profile view -d {PROFILE_DIR} --port 3001")

# output[0] = logits_per_image (return_dict=False returns a tuple;
# index 0 is logits_per_image per CLIPModel source).
with torch.no_grad():
    output = neuron_clip(input_ids, pixel_values)
    
logits = output[0]
probs = logits.softmax(dim=1)
print("\nCLIP similarity (image x text):")

for i, url in enumerate(img_urls * 2):
    print(f"  Image {i}: {[f'{p:.3f}' for p in probs[i].tolist()]}")
