# Out-of-Stock Retail Detection: Evaluating VLM Core Capabilities (Cosmos-Reason2)

This repository is dedicated to evaluating and highlighting the intrinsic limitations of a frozen, non-fine-tuned Vision-Language Reasoning Model (Cosmos-Reason2) when applied to a highly domain-specific task: **Out-of-Stock (OOS) retail shelf detection**.

## 🎯 Project Core Objective

The primary objective of this project is to demonstrate that a state-of-the-art multimodal reasoning model, when used in a completely zero-shot or prompt-engineered configuration (frozen parameters), struggles to guarantee robust reliability on specialized tasks like empty supermarket shelf detection — even when given the ability to reason explicitly before answering.

While the model shows solid generic grounding and visual capacities, it suffers from specific structural vulnerabilities — such as missing targets due to poor lighting, misclassifying the uniform-dark shadow signature of a genuinely empty gap as a solid object, degenerate repetition loops under greedy decoding, and excessive hedging toward negative answers when reasoning is left unconstrained. 

This research establishes the critical necessity of domain-specific fine-tuning pipelines for industrial retail applications, and motivates the adoption of a dedicated fine-tuned architecture (SigLIP) as the project's final solution.

## 🤖 Target Evaluated Model

All experimental multi-modal pipelines within this repository are built around:
* **Cosmos-Reason2-8B (NVIDIA)**, served locally via vLLM (OpenAI-compatible HTTP API).

Unlike the companion Qwen2.5-VL evaluation track, no parameter-scale sweep is performed here — Cosmos-Reason2-8B is evaluated in a single scale across four evolutionary configurations, each targeting a specific structural weakness uncovered in the configuration before it.

## 📊 Dataset & Privacy Restrictions

> ⚠️ **Important Notice regarding Data Availability:** The evaluation dataset consists of private, high-resolution surveillance streams capturing retail supermarket shelf gaps (1679 test images, 5390 ground-truth bounding boxes), utilized as part of the research conducted at the **MIVIA Lab (University of Salerno)**. Due to strict corporate non-disclosure agreements (NDAs) and privacy regulations, the raw images and ground truth bounding box label annotations cannot be uploaded or distributed within this repository.

However, the complete programmatic evaluation matrices, structural JSON evaluation outputs, and rendered comparison tables are maintained inside the localized project `results/` directory footprint.

## 📑 Experimental Framework & Test Specifications

The codebase evaluates the synergy between an initial object detector blueprint (YOLO, aug/no_aug variants) and sequential validation passes handled by the VLM (Cosmos-Reason2-8B) through four evolutionary software stages. All pipelines serve the model through a persistent vLLM server and include fail-fast error handling (the run aborts with an explicit error after 5 consecutive failed HTTP requests, rather than silently degrading into meaningless results).

### Test 1: Single-Image Baseline
* **Script:** `evaluate_yolo_cosmos.py` (supported by `vllm_inference_module.py`).
* **Objective:** Establish an initial assessment of Cosmos-Reason2's zero-shot behavior on the OOS task, using the simplest possible visual input.
* **Input Structure:** The VLM receives the full reference image, resized to a 1280px cap, with the candidate box drawn as a green rectangle. No close-up crop is provided.
* **Message / Prompt Protocol:** Box coordinates are passed as native `[Region 0]<box>x1,y1,x2,y2</box>` grounding tokens, normalized to Cosmos' 0–1024 coordinate scale. The model is instructed to answer in exactly one word.
* **Output / Decision Mechanism:** Direct binary text parsing ("yes" / "no"), no logits or reasoning involved.
* **Main Result:** TPR = 0.748, FPR = 0.018 — Cosmos lowers YOLO's false-positive rate but at the cost of ~1000 additional false negatives versus the YOLO-only baseline.

### Test 2: Dual-Image, 20% Crop Padding
* **Script:** `evaluate_yolo_cosmos_v2_FN.py` (supported by `vllm_inference_module_v2.py`).
* **Objective:** Recover the sensitivity lost in Test 1 by giving the model the same level of local detail a human reviewer would get by physically moving closer to the shelf.
* **Input Structure:** Dual multi-modal visual stream. The model simultaneously receives the full annotated image (global context: shelf depth, lighting, surrounding products) and a tight crop of the candidate box with 20% spatial padding (upscaled to a 28px floor on the shorter side to preserve model patch resolution).
* **Message / Prompt Protocol:** Two prompt variants are compared — a minimal `no_context` prompt and a structured `context` prompt with an explicit two-step false-positive checklist (shelf structure, fridge glass, motion blur, poor lighting, distant products).
* **Output / Decision Mechanism:** Direct binary text parsing, same as Test 1.
* **Main Result:** **Best configuration found in this evaluation track.** TPR = 0.878, FPR = 0.024 — a ~3× reduction in Cosmos-caused false negatives versus Test 1 (+328 vs +998), while still holding a lower FPR than YOLO alone.
* **Additional output:** per-ground-truth false-negative decomposition (`fn_missed_by_yolo` vs `fn_rejected_by_cosmos`), isolating how many misses originate from the detector versus from the VLM filter.

### Test 3: Dual-Image, 40% Crop Padding
* **Script:** `evaluate_yolo_cosmos_v3.py` (supported by `vllm_inference_module_v3.py`).
* **Objective:** Test whether a wider local context window around the candidate box further improves the VLM's judgment.
* **Input Structure:** Identical dual-stream setup to Test 2, with crop padding increased from 20% to 40%.
* **Output / Decision Mechanism:** Same as Test 2.
* **Main Result:** No measurable improvement. TPR = 0.875, FPR = 0.025 — statistically indistinguishable from Test 2, confirming that Test 2 had already reached a performance plateau and that additional local context does not help.

### Test 4: Explicit Chain-of-Thought Reasoning
* **Script:** `evaluate_yolo_cosmos_v4.py` (supported by `vllm_inference_module_v4.py`).
* **Objective:** Exploit Cosmos-Reason2's native reasoning capability — previously suppressed by the single-word answer constraint in Tests 1–3 — by allowing the model to reason explicitly before answering.
* **Input Structure:** Same dual-stream setup as Test 2 (20% padding).
* **Message / Prompt Protocol:** The system prompt instructs the model to reason inside `<think></think>` tags and commit to a final answer inside `<answer>yes</answer>` / `<answer>no</answer>` tags. `max_tokens` raised from 300 to 700 to accommodate genuine reasoning chains.
* **Output / Decision Mechanism:** Regex extraction of the `<answer>` tag content, with a configurable fallback (`ambiguous_default`) for the rare cases where no tag is produced.
* **Main Result:** Severe performance collapse. TPR dropped to the 0.02–0.16 range depending on prompt variant, with FPR near zero — the model rejects almost every candidate box. Root-cause analysis identified two concurrent failure modes:
  1. **Degenerate repetition loops** under greedy decoding (`temperature=0`) on visually ambiguous crops, consuming the entire token budget without ever reaching the answer tag (mitigated, not eliminated, with `repetition_penalty=1.15`).
  2. **Systematic misclassification of shadow as solid object** — the model consistently interpreted the uniform dark rectangular appearance of a genuinely empty shelf gap (a shadow where no light reaches the back wall) as a physical product or shelf-structure element, even after targeted prompt revisions explicitly teaching the shadow/object distinction and an explicit "resolve uncertainty toward YES" decision rule.
* **Conclusion:** Enabling explicit reasoning did not recover the plateau reached in Test 2 — instead, it introduced instability and excessive conservatism that repeated prompt-engineering interventions could not fully resolve.

## 📈 Cross-Test Summary

| Test | TPR | FPR | FN caused by Cosmos (Δ vs YOLO-only) |
|---|---|---|---|
| **YOLO-only baseline** | 0.944 | 0.032 | — |
| **Test 1 — Single-image** | 0.748 | 0.018 | +998 |
| **Test 2 — Dual-image, 20% padding** | 0.878 | 0.024 | +328 |
| **Test 3 — Dual-image, 40% padding** | 0.875 | 0.025 | +350 |
| **Test 4 — Explicit reasoning** | 0.019–0.162 | ≈0.000–0.010 | — (collapse) |

*(All figures: detector aug, conf=0.25, IoU@0.0, #GT=5390)*

In no configuration does Cosmos-Reason2 match the YOLO-only baseline's recall without a substantial false-negative cost, and the attempt to leverage the model's native reasoning capability (Test 4) proved actively counterproductive. This progression — plateau, then collapse under reasoning — is the core experimental evidence supporting the decision to pursue a dedicated fine-tuned architecture (SigLIP) as the project's final solution, rather than a frozen, prompt-engineered VLM.

## 📁 Repository Directory Structure

```text
Cosmos Pipeline/
└── src/
    ├── models/
    │   ├── vllm_inference_module.py       # Test 1 — single-image interface
    │   ├── vllm_inference_module_v2.py    # Test 2 — dual-image, 20% padding (best config)
    │   ├── vllm_inference_module_v3.py    # Test 3 — dual-image, 40% padding
    │   └── vllm_inference_module_v4.py    # Test 4 — dual-image + explicit <think>/<answer> reasoning
    └── evaluation/
        ├── evaluate_yolo_cosmos.py        # Test 1 evaluation routine
        ├── evaluate_yolo_cosmos_v2_FN.py  # Test 2 evaluation routine (+ per-GT FN decomposition)
        ├── evaluate_yolo_cosmos_v3.py     # Test 3 evaluation routine
        ├── evaluate_yolo_cosmos_v4.py     # Test 4 evaluation routine
        ├── visualize_detector_only.py     # Baseline YOLO plot/table rendering suite
        └── visualize_yolo_cosmos_v2_FN.py # Cross-configuration charting suite (TPR/FPR/FNR tables, FN decomposition)
