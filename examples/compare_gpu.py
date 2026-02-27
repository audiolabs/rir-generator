#!/usr/bin/env python3
"""Compare the GPU/vectorised implementation against the C++ reference backend.

Produces
--------
* Numerical comparison  — max absolute error, relative error, per-mic correlation
* Execution-time comparison — median wall-clock time and real-time factor (RTF)
  over several runs, with and without first-call overhead
* Visual comparison     — 4-panel matplotlib figure saved to rir_comparison.png

Usage
-----
    python examples/compare_gpu.py
"""

import time

import matplotlib
import numpy as np

matplotlib.use("Agg")  # works in headless / CI environments
import matplotlib.pyplot as plt

import rir_generator
from rir_generator import gpu as rir_gpu

# ---------------------------------------------------------------------------
# Room / signal parameters
# ---------------------------------------------------------------------------
C = 340.0
FS = 16000
R = [[2.0, 1.5, 2.0], [2.0, 1.5, 3.0]]  # two microphones
S = [2.0, 3.5, 2.0]
L = [5.0, 4.0, 6.0]
RT60 = 0.4
NSAMPLE = 4096
BENCH_RUNS = 10  # repeated runs for stable median timing

KWARGS = dict(c=C, fs=FS, r=R, s=S, L=L, reverberation_time=RT60, nsample=NSAMPLE)

print("=" * 62)
print("Room Impulse Response — implementation comparison")
print("=" * 62)
print(f"  Room       : {L} m")
print(f"  Source     : {S}")
print(f"  Receivers  : {R}")
print(f"  RT60       : {RT60} s  |  fs = {FS} Hz  |  N = {NSAMPLE}")
print()

# ---------------------------------------------------------------------------
# C++ reference backend — single run
# ---------------------------------------------------------------------------
t0 = time.perf_counter()
h_cpp = rir_generator.generate(**KWARGS)
t_cpp_first = time.perf_counter() - t0

# ---------------------------------------------------------------------------
# GPU/PyTorch backend (CPU device so results are reproducible everywhere) — single run
# ---------------------------------------------------------------------------
t0 = time.perf_counter()
h_gpu = rir_gpu.generate(**KWARGS, device="cpu")
t_gpu_first = time.perf_counter() - t0

# ---------------------------------------------------------------------------
# Numerical comparison
# ---------------------------------------------------------------------------
print("Numerical comparison")
print(f"  Shape (C++)   : {h_cpp.shape}")
print(f"  Shape (GPU)   : {h_gpu.shape}")

max_abs_err = float(np.max(np.abs(h_cpp - h_gpu)))
peak = float(np.max(np.abs(h_cpp)))
rel_err = max_abs_err / (peak + 1e-12)

corr = [
    float(np.corrcoef(h_cpp[:, m], h_gpu[:, m])[0, 1])
    for m in range(h_cpp.shape[1])
]

print(f"  Max |Δ|             : {max_abs_err:.3e}")
print(f"  Relative Δ          : {rel_err:.3e}")
for m, c_val in enumerate(corr):
    print(f"  Correlation mic {m + 1}  : {c_val:.10f}")
print()

# ---------------------------------------------------------------------------
# Execution-time comparison
# Warm-up first so we measure steady-state throughput, then time BENCH_RUNS.
# The first-call time is also reported separately since it captures PyTorch's
# lazy-initialisation overhead.
# ---------------------------------------------------------------------------

def _median_ms(fn, n=BENCH_RUNS):
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1000.0


# Warm-up
rir_generator.generate(**KWARGS)
rir_gpu.generate(**KWARGS, device="cpu")

t_cpp_med = _median_ms(lambda: rir_generator.generate(**KWARGS))
t_gpu_med = _median_ms(lambda: rir_gpu.generate(**KWARGS, device="cpu"))

audio_ms = NSAMPLE / FS * 1000.0  # ms of audio produced

def rtf(t_ms):
    return audio_ms / t_ms


print(f"Execution time  ({BENCH_RUNS}-run median after warm-up)")
print(f"  C++ backend      : {t_cpp_med:8.2f} ms   RTF {rtf(t_cpp_med):5.1f}×")
print(f"  GPU/CPU (PyTorch): {t_gpu_med:8.2f} ms   RTF {rtf(t_gpu_med):5.1f}×")
print(f"  Speedup          : {t_cpp_med / t_gpu_med:.2f}×  (CPU-only; see note below)")
print()
print(f"First-call time  (includes PyTorch initialisation overhead)")
print(f"  C++ backend      : {t_cpp_first * 1000:8.2f} ms")
print(f"  GPU/CPU (PyTorch): {t_gpu_first * 1000:8.2f} ms")
print()
print(
    "Note: on a CUDA device the PyTorch backend processes all ~60 000 image\n"
    "      sources in parallel; CPU throughput is limited by the irregular\n"
    "      reflection-accumulation pattern (scatter_add_) competing with the\n"
    "      cache-friendly sequential writes of the C++ loop.  Expect 10–100×\n"
    "      speedup over the C++ reference when running on GPU."
)
print()

# ---------------------------------------------------------------------------
# Visual comparison — 4-panel figure
# Top row    : overlaid impulse responses (C++ vs GPU)
# Bottom row : difference signal
# ---------------------------------------------------------------------------
t_ms = np.arange(NSAMPLE) / FS * 1000.0
n_mics = h_cpp.shape[1]

fig, axes = plt.subplots(2, n_mics, figsize=(6 * n_mics, 7), tight_layout=True)
if n_mics == 1:
    axes = axes[:, np.newaxis]

for m in range(n_mics):
    # ---- overlaid impulse responses ----
    ax = axes[0, m]
    ax.plot(t_ms, h_cpp[:, m], label="C++ reference", alpha=0.85, lw=0.8, color="steelblue")
    ax.plot(
        t_ms,
        h_gpu[:, m],
        label="GPU/PyTorch",
        alpha=0.85,
        lw=0.8,
        linestyle="--",
        color="tomato",
    )
    ax.set_title(f"Microphone {m + 1} — Room Impulse Response")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.legend(fontsize=9)

    # ---- difference signal ----
    diff = h_cpp[:, m] - h_gpu[:, m]
    ax = axes[1, m]
    ax.plot(t_ms, diff, color="dimgray", lw=0.6)
    ax.axhline(0, color="black", lw=0.4, ls="--")
    ax.set_title(
        f"Microphone {m + 1} — Difference  (max |Δ| = {np.max(np.abs(diff)):.2e})"
    )
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Δ Amplitude")

speedup_str = f"{t_cpp_med / t_gpu_med:.2f}×"
fig.suptitle(
    f"RIR: C++ vs GPU/PyTorch (CPU)  |  CPU speedup {speedup_str}  |  max |Δ| {max_abs_err:.1e}",
    fontsize=12,
    y=1.01,
)

out_path = "rir_comparison.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Figure saved → {out_path}")
