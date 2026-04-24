"""shared/profiler.py

Hardware-agnostic profiling functions for all three tiers.
The same four public functions run unchanged across Tier 1 (CPU), Tier 2 (T4),
and Tier 3 (A100). Only the inference_fn and inputs passed to them differ.

Energy estimation note:
    estimate_energy() uses the TDP (thermal design power) of the device as a
    proxy for actual power draw. This gives a conservative upper-bound estimate
    rather than a measured value. On real hardware, a power meter or RAPL
    interface should be used for accurate readings. The estimate is still
    useful for cross-tier comparisons when the same methodology is applied
    consistently.

CSV output:
    save_results_to_csv() writes one row per profiling run to a CSV file under
    RESULTS_DIR. The schema is fixed so that all three tiers produce files that
    can be concatenated directly for the D2 benchmark report.
"""

import csv
import os
import time
from functools import wraps

import numpy as np
import psutil


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_process_ram_mb() -> float:
    """Return current process RSS memory usage in megabytes.

    Args:
        None

    Returns:
        RSS memory in MB as a float.
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)


def _time_single_call(inference_fn: callable, inputs) -> float:
    """Time a single call to inference_fn and return wall-clock ms.

    Uses time.perf_counter() which has sub-millisecond resolution on all
    supported platforms.

    Args:
        inference_fn: A callable that accepts inputs and returns any value.
        inputs: The input to pass to inference_fn.

    Returns:
        Wall-clock elapsed time in milliseconds as a float.
    """
    start = time.perf_counter()
    inference_fn(inputs)
    end   = time.perf_counter()
    return (end - start) * 1000.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def measure_latency(
    inference_fn: callable,
    inputs,
    n_runs: int,
) -> dict:
    """Measure per-call latency of inference_fn over n_runs repeated calls.

    The first call is treated as a warm-up and excluded from statistics.
    This avoids counting JIT compilation, model cache loading, or OS page
    faults in the reported latency, which would not reflect steady-state
    performance.

    Args:
        inference_fn: A callable that accepts inputs and performs inference.
            For Tier 1, use train_tier1.infer_joblib or train_tier1.infer_onnx
            wrapped in a lambda that closes over the model.
            For Tier 2, a lambda wrapping the HuggingFace model forward pass.
            For Tier 3, a lambda wrapping tier3_llm.run_inference.
        inputs: The input batch to pass to inference_fn on each call.
            CONFIGURABLE: use a representative batch size for the target tier.
        n_runs: Number of timed calls (excluding the warm-up call).
            CONFIGURABLE: set via N_LATENCY_RUNS in the notebook config cell.

    Returns:
        A dict with keys:
            'mean_ms': float, mean latency in milliseconds.
            'std_ms':  float, standard deviation in milliseconds.
            'p95_ms':  float, 95th percentile latency in milliseconds.
            'all_ms':  list[float], all individual run times (for plotting).
    """
    # Warm-up call: excluded from statistics
    inference_fn(inputs)

    times_ms = [_time_single_call(inference_fn, inputs) for _ in range(n_runs)]

    return {
        "mean_ms": float(np.mean(times_ms)),
        "std_ms":  float(np.std(times_ms)),
        "p95_ms":  float(np.percentile(times_ms, 95)),
        "all_ms":  times_ms,
    }


def measure_peak_ram(
    inference_fn: callable,
    inputs,
) -> float:
    """Measure peak RSS RAM consumed during a single inference call.

    Samples process RSS memory before and after the call, then reports
    the delta. This approach captures the allocation caused by inference
    without requiring the memory_profiler decorator, which can interfere
    with some GPU runtimes.

    Note: RSS measures physical memory pages held by the process. It
    does not include GPU VRAM. For GPU tiers, measure VRAM separately
    using torch.cuda.max_memory_allocated().

    Args:
        inference_fn: A callable that accepts inputs and performs inference.
        inputs: The input to pass to inference_fn.

    Returns:
        Peak RAM increase in MB as a float. Returns 0.0 if memory usage
        decreased during the call (can occur due to GC).
    """
    ram_before_mb = _get_process_ram_mb()
    inference_fn(inputs)
    ram_after_mb  = _get_process_ram_mb()

    delta_mb = ram_after_mb - ram_before_mb
    return float(max(delta_mb, 0.0))


def measure_peak_vram_mb() -> float:
    """Measure peak GPU VRAM allocated since the last reset, in megabytes.

    This function is a no-op on CPU (Tier 1) and returns 0.0 safely.
    On GPU tiers (Tier 2 and Tier 3), it reads from PyTorch's CUDA
    memory tracker.

    Call torch.cuda.reset_peak_memory_stats() before the inference call
    whose VRAM you want to measure, then call this function after.

    Args:
        None

    Returns:
        Peak VRAM allocated in MB. Returns 0.0 if CUDA is unavailable.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    except ImportError:
        return 0.0


def estimate_energy(
    latency_ms: float,
    tdp_watts: float,
) -> float:
    """Estimate energy consumed during one inference call.

    Uses the device TDP as a proxy for actual power draw. This gives a
    conservative upper bound. The formula is:
        energy_mJ = latency_s * power_W * 1000

    Args:
        latency_ms: Mean latency in milliseconds from measure_latency().
        tdp_watts: Thermal design power of the device in watts.
            CONFIGURABLE: set via CPU_TDP_WATTS or GPU_TDP_WATTS in the
            notebook config cell.

    Returns:
        Estimated energy consumption in millijoules (mJ) as a float.
    """
    latency_s  = latency_ms / 1000.0
    energy_mj  = latency_s * tdp_watts * 1000.0
    return float(energy_mj)


def measure_load_time(load_fn: callable) -> float:
    """Time a single call to load_fn, which loads model artefacts from disk.

    This measures cold-load time: the time to deserialise and initialise the
    model from scratch, including any JIT compilation or graph optimisation
    performed by the runtime on first load.

    Args:
        load_fn: A zero-argument callable that loads and returns the model.
            For Tier 1 joblib: lambda: load_artefacts_joblib(p1, p2, p3)
            For Tier 1 ONNX:   lambda: load_artefacts_onnx(p1, p2, p3)
            For Tier 2:        lambda: load_pretrained_model(model_name)
            For Tier 3:        lambda: AutoModelForCausalLM.from_pretrained(...)

    Returns:
        Load time in milliseconds as a float.
    """
    start = time.perf_counter()
    load_fn()
    end   = time.perf_counter()
    return float((end - start) * 1000.0)


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------

# Fixed CSV column schema shared across all three tiers.
# All tiers write to this schema so results files can be concatenated.
RESULTS_CSV_COLUMNS = [
    "tier",
    "model_name",
    "inference_mode",    # 'joblib', 'onnx', 'hf_pytorch', 'onnx_gpu', 'llm_hf'
    "n_samples",
    "mean_latency_ms",
    "std_latency_ms",
    "p95_latency_ms",
    "peak_ram_mb",
    "peak_vram_mb",
    "energy_mj",
    "load_time_ms",
    "accuracy",
    "f1_macro",
    "notes",             # CONFIGURABLE: free-text field for run notes
]


def save_results_to_csv(results: dict, path: str) -> None:
    """Append one row of profiling results to a CSV file.

    Creates the file and writes the header row if the file does not yet
    exist. Appends without overwriting if the file already exists, so
    multiple runs accumulate in the same file.

    Args:
        results: A dict whose keys are a subset of RESULTS_CSV_COLUMNS.
            Missing keys are written as empty strings.
        path: Full file path for the CSV. Should be under RESULTS_DIR.

    Returns:
        None
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    file_exists = os.path.isfile(path)

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=RESULTS_CSV_COLUMNS,
            extrasaction="ignore",
        )
        if not file_exists:
            writer.writeheader()

        row = {col: results.get(col, "") for col in RESULTS_CSV_COLUMNS}
        writer.writerow(row)


def run_full_profile(
    inference_fn: callable,
    load_fn: callable,
    inputs,
    tdp_watts: float,
    n_runs: int,
    tier: int,
    model_name: str,
    inference_mode: str,
    n_samples: int,
    accuracy: float,
    f1_macro: float,
    results_csv_path: str,
    notes: str = "",
) -> dict:
    """Run all profiling measurements and save results to CSV in one call.

    This is the convenience function called from the profiling stub cell
    in each notebook. It calls all four measurement functions in sequence,
    assembles a results dict, prints a summary, and writes to CSV.

    Args:
        inference_fn: A callable that accepts inputs and performs inference.
        load_fn: A zero-argument callable that loads the model from disk.
        inputs: The input batch passed to inference_fn on each call.
        tdp_watts: Device TDP in watts for energy estimation.
        n_runs: Number of latency measurement runs.
        tier: Integer tier number (1, 2, or 3).
        model_name: Human-readable model identifier string.
        inference_mode: One of 'joblib', 'onnx', 'hf_pytorch', 'onnx_gpu',
            'llm_hf'.
        n_samples: Number of samples in the inputs batch.
        accuracy: Overall accuracy from evaluate_model().
        f1_macro: Macro F1 from evaluate_model().
        results_csv_path: Full path to the results CSV file.
        notes: Optional free-text notes for this run.

    Returns:
        A dict containing all measured values, matching RESULTS_CSV_COLUMNS.
    """
    print(f"[profile] Starting full profile: Tier {tier} | {model_name} | {inference_mode}")

    latency_stats = measure_latency(inference_fn, inputs, n_runs)
    print(f"[profile] Latency: mean={latency_stats['mean_ms']:.2f}ms  "
          f"std={latency_stats['std_ms']:.2f}ms  "
          f"p95={latency_stats['p95_ms']:.2f}ms")

    peak_ram = measure_peak_ram(inference_fn, inputs)
    print(f"[profile] Peak RAM delta: {peak_ram:.2f} MB")

    peak_vram = measure_peak_vram_mb()
    print(f"[profile] Peak VRAM: {peak_vram:.2f} MB")

    energy = estimate_energy(latency_stats["mean_ms"], tdp_watts)
    print(f"[profile] Energy estimate: {energy:.4f} mJ")

    load_time = measure_load_time(load_fn)
    print(f"[profile] Load time: {load_time:.2f} ms")

    results = {
        "tier":             tier,
        "model_name":       model_name,
        "inference_mode":   inference_mode,
        "n_samples":        n_samples,
        "mean_latency_ms":  round(latency_stats["mean_ms"], 4),
        "std_latency_ms":   round(latency_stats["std_ms"],  4),
        "p95_latency_ms":   round(latency_stats["p95_ms"],  4),
        "peak_ram_mb":      round(peak_ram,   4),
        "peak_vram_mb":     round(peak_vram,  4),
        "energy_mj":        round(energy,     6),
        "load_time_ms":     round(load_time,  4),
        "accuracy":         round(accuracy,   6),
        "f1_macro":         round(f1_macro,   6),
        "notes":            notes,
    }

    save_results_to_csv(results, results_csv_path)
    print(f"[profile] Results saved to {results_csv_path}")

    return results
