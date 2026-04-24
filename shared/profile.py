# shared/profile.py
# │
# ├── measure_latency(inference_fn: callable, inputs,
# │                   n_runs: int) -> dict
# │     Runs inference_fn on inputs n_runs times and records wall-clock
# │     time per run. Returns mean, std, and 95th-percentile latency.
# │     Returns: {'mean_ms': float, 'std_ms': float, 'p95_ms': float}
# │
# ├── measure_peak_ram(inference_fn: callable, inputs) -> float
# │     Uses memory_profiler to capture peak RAM consumption during a
# │     single inference call.
# │     Returns: peak RAM in MB
# │
# ├── estimate_energy(latency_ms: float,
# │                   cpu_tdp_watts: float) -> float
# │     Estimates energy as latency * TDP. This is a deliberate
# │     simplification: it assumes the CPU runs at full TDP during
# │     inference, which gives a conservative upper bound.
# │     Returns: estimated energy in mJ
# │
# └── measure_load_time(load_fn: callable) -> float
#       Times a single call to load_fn, which should be a zero-argument
#       callable that loads the model from disk.
#       Returns: load time in ms
