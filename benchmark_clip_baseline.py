import argparse
import csv
import os
import statistics
import time
from pathlib import Path

os.environ.pop("NEURON_PROFILE", None)

import requests
import torch
import torch_neuronx
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


MODEL_ID = "openai/clip-vit-base-patch32"
TEXTS = [
    "a photo of a cat",
    "a photo of a train",
    "a photo of a dog",
    "two cats sleeping on a couch",
]
IMAGE_URLS = [
    "http://images.cocodataset.org/val2017/000000039769.jpg",
    "http://images.cocodataset.org/val2017/000000397133.jpg",
]


def percentile(values, percent):
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * percent / 100
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def load_images():
    images = []
    for url in IMAGE_URLS:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        images.append(Image.open(response.raw).convert("RGB"))
    return images


def make_inputs(processor, images, image_batch):
    repeated_images = [images[index % len(images)] for index in range(image_batch)]
    inputs = processor(
        text=TEXTS,
        images=repeated_images,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=77,
    )
    return inputs["input_ids"], inputs["pixel_values"]


def summarize(times_ms, image_batch):
    mean_ms = statistics.mean(times_ms)
    return {
        "mean_ms": mean_ms,
        "p50_ms": percentile(times_ms, 50),
        "p90_ms": percentile(times_ms, 90),
        "p99_ms": percentile(times_ms, 99),
        "throughput_images_per_sec": image_batch / (mean_ms / 1000),
    }


def benchmark_callable(callable_model, input_ids, pixel_values, warmup, iters):
    with torch.no_grad():
        for _ in range(warmup):
            _ = callable_model(input_ids, pixel_values)

        times_ms = []
        for _ in range(iters):
            start = time.perf_counter()
            output = callable_model(input_ids, pixel_values)
            end = time.perf_counter()
            times_ms.append((end - start) * 1000)

    return output, times_ms


def compile_or_load_neuron(model, input_ids, pixel_values, compiled_path, force_compile):
    compile_seconds = 0.0
    if compiled_path.exists() and not force_compile:
        return torch.jit.load(str(compiled_path)), compile_seconds, "loaded"

    compiled_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    neuron_model = torch_neuronx.trace(
        model,
        (input_ids, pixel_values),
        compiler_args="--enable-saturate-infinity",
    )
    compile_seconds = time.perf_counter() - start
    torch.jit.save(neuron_model, str(compiled_path))
    return neuron_model, compile_seconds, "compiled"


def main():
    parser = argparse.ArgumentParser(
        description="Compare CPU PyTorch CLIP against Inferentia2/Neuron CLIP."
    )
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--cpu-iters", type=int, default=20)
    parser.add_argument("--neuron-iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--compiled-dir", default="./clip_neuron_compiled_baselines")
    parser.add_argument("--out", default="./clip_baseline_results.csv")
    parser.add_argument("--force-compile", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(os.cpu_count() or 1)

    print("[1/4] Loading CLIP and inputs...")
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    cpu_model = CLIPModel.from_pretrained(MODEL_ID, return_dict=False).eval()
    images = load_images()

    rows = []
    for image_batch in args.batches:
        print(f"\n[2/4] Batch={image_batch}: preparing fixed-shape inputs...")
        input_ids, pixel_values = make_inputs(processor, images, image_batch)
        compiled_path = Path(args.compiled_dir) / f"clip_vit_b32_images_bs{image_batch}_texts4.pt"

        print(f"[3/4] Batch={image_batch}: CPU baseline...")
        cpu_output, cpu_times = benchmark_callable(
            cpu_model,
            input_ids,
            pixel_values,
            warmup=args.warmup,
            iters=args.cpu_iters,
        )
        cpu_stats = summarize(cpu_times, image_batch)

        print(f"[4/4] Batch={image_batch}: Neuron compile/load + benchmark...")
        neuron_model, compile_seconds, compile_status = compile_or_load_neuron(
            cpu_model,
            input_ids,
            pixel_values,
            compiled_path,
            force_compile=args.force_compile,
        )
        neuron_output, neuron_times = benchmark_callable(
            neuron_model,
            input_ids,
            pixel_values,
            warmup=args.warmup,
            iters=args.neuron_iters,
        )
        neuron_stats = summarize(neuron_times, image_batch)

        cpu_logits = cpu_output[0].detach().float()
        neuron_logits = neuron_output[0].detach().float()
        max_abs_diff = (cpu_logits - neuron_logits).abs().max().item()

        row = {
            "image_batch": image_batch,
            "text_count": len(TEXTS),
            "compile_status": compile_status,
            "compile_seconds": round(compile_seconds, 3),
            "cpu_mean_ms": round(cpu_stats["mean_ms"], 3),
            "cpu_p50_ms": round(cpu_stats["p50_ms"], 3),
            "cpu_p90_ms": round(cpu_stats["p90_ms"], 3),
            "cpu_p99_ms": round(cpu_stats["p99_ms"], 3),
            "cpu_throughput_images_per_sec": round(cpu_stats["throughput_images_per_sec"], 3),
            "neuron_mean_ms": round(neuron_stats["mean_ms"], 3),
            "neuron_p50_ms": round(neuron_stats["p50_ms"], 3),
            "neuron_p90_ms": round(neuron_stats["p90_ms"], 3),
            "neuron_p99_ms": round(neuron_stats["p99_ms"], 3),
            "neuron_throughput_images_per_sec": round(
                neuron_stats["throughput_images_per_sec"], 3
            ),
            "speedup_cpu_mean_over_neuron_mean": round(
                cpu_stats["mean_ms"] / neuron_stats["mean_ms"], 3
            ),
            "max_abs_logit_diff": round(max_abs_diff, 6),
            "compiled_path": str(compiled_path),
        }
        rows.append(row)

        print(
            f"batch={image_batch} cpu_p50={row['cpu_p50_ms']} ms "
            f"neuron_p50={row['neuron_p50_ms']} ms "
            f"speedup={row['speedup_cpu_mean_over_neuron_mean']}x "
            f"max_abs_logit_diff={row['max_abs_logit_diff']}"
        )

    with open(args.out, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    main()
