#!/usr/bin/env python3
"""
diag_gpu_raw_bench.py
------------------------
Benchmark puro di calcolo (nessun modello, nessun transformers) per capire
se la fetta MIG assegnata ha capacita' di calcolo (SM) ridotta.

Confronta il risultato con questa referenza approssimativa per H100 (bf16,
Tensor Core, matmul grande):
  - H100 intera (SXM/NVL)  : ~400-700+ TFLOPS bf16 (con Tensor Core, teorici)
  - Fetta MIG piccola (es. 1g.Xgb) : una frazione proporzionale ai suoi SM

Se questo benchmark mostra performance molto basse (es. <20-30 TFLOPS
sostenuti), la fetta MIG e' probabilmente il collo di bottiglia strutturale,
indipendentemente dal codice del modello.
"""

import time
import torch


def bench_matmul(n: int, dtype: torch.dtype, iters: int = 20) -> float:
    a = torch.randn(n, n, dtype=dtype, device="cuda")
    b = torch.randn(n, n, dtype=dtype, device="cuda")

    # warm-up
    for _ in range(3):
        c = a @ b
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        c = a @ b
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    flops_per_matmul = 2 * (n ** 3)  # moltiplicazioni + somme
    total_flops = flops_per_matmul * iters
    tflops = total_flops / elapsed / 1e12
    return tflops, elapsed / iters


def main():
    print(f"CUDA disponibile: {torch.cuda.is_available()}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Capability: {torch.cuda.get_device_capability(0)}")
    print(f"VRAM totale visibile: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    print("\n--- Benchmark matmul bf16 (Tensor Core) ---")
    for n in [2048, 4096, 8192]:
        tflops, t_per_iter = bench_matmul(n, torch.bfloat16)
        print(f"  n={n:5d}  {tflops:7.2f} TFLOPS  ({t_per_iter*1000:.2f} ms/iter)")

    print("\n--- Benchmark matmul fp32 (per confronto) ---")
    tflops, t_per_iter = bench_matmul(4096, torch.float32)
    print(f"  n=4096  {tflops:7.2f} TFLOPS  ({t_per_iter*1000:.2f} ms/iter)")

    print("\nRiferimento indicativo (bf16, Tensor Core):")
    print("  H100 intera        : ~400-700+ TFLOPS teorici")
    print("  Fetta MIG ridotta  : una frazione proporzionale ai suoi SM")


if __name__ == "__main__":
    main()