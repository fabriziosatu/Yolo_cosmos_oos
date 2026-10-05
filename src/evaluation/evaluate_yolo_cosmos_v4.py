"""
evaluate_yolo_cosmos_v4.py  —  v4
--------------------------------------
Evaluates 4 pipelines: 2 detectors (noaug / aug) × 2 Cosmos prompts (no_context / context)

NEW IN v4:
  - Uses cosmos_inference_module.CosmosInferenceModule
    (updated prompts for single image input with green BB +
     normalized coordinates in text; textual yes/no response)
  - Cosmos responds with text ("yes"/"no") instead of numerical probabilities
  - cosmos_thr is removed: Cosmos decides directly with its response
  - Input to Cosmos: full image with green BB + normalized coordinates in the prompt
  - New metrics: Cosmos yes/no count vs YOLO predictions
    · cosmos_yes   = how many YOLO BBs Cosmos confirmed as empty
    · cosmos_no    = how many YOLO BBs Cosmos rejected
    · yolo_total = total BBs predicted by YOLO
  - TP/FP/FN calculation strategy:
    · BB with "yes" response → kept → multi-match vs GT → TP or FP
    · BB with "no" response  → discarded → if it was a real TP → FN

Usage (Bash):
    cd /user/fsaturnino/Progetto_tesi_Cosmos_user
    
    nohup env PYTHONPATH=/user/fsaturnino/Progetto_tesi_Cosmos_user \
    PYTHONUNBUFFERED=1 python3 -u src/evaluation/evaluate_yolo_cosmos.py \
        > /cache/fsaturnino/Progetto_tesi_Cosmos/results/baseline_cosmos_v1/log.txt 2>&1 &
        
    echo "PID: $!"; tail -f /cache/fsaturnino/Progetto_tesi_Cosmos/results/baseline_cosmos_v1/log.txt
"""

import argparse, json, time
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ══════════════════════════════════════════════════════════════
#  CONFIG — hardcoded paths (only modify these)
#  Updated for the second cluster (ainode, home directory,
#  vLLM server instead of local transformers)
# ══════════════════════════════════════════════════════════════
import os
_PROJECT_ROOT = os.path.expanduser("~/Progetto_tesi_Cosmos")

WEIGHTS_NOAUG   = f"{_PROJECT_ROOT}/weights/phase1_baseline_comune/no_aug/best.pt"
WEIGHTS_AUG     = f"{_PROJECT_ROOT}/weights/phase1_baseline_comune/aug/best.pt"
TEST_IMAGES     = f"{_PROJECT_ROOT}/dataset_test_ripulito/images"
TEST_LABELS     = f"{_PROJECT_ROOT}/dataset_test_ripulito/labels"
OUT_DIR_DEFAULT = f"{_PROJECT_ROOT}/results/baseline_cosmos_v4_reasoning"
VLLM_URL_DEFAULT = "http://localhost:8000"
# ══════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
CONF_THRESHOLDS = [0.25, 0.5]
IOU_THRESHOLDS  = [0.0, 0.25, 0.5, 0.75]
PROMPT_NAMES    = ["no_context", "context"]
# ─────────────────────────────────────────────────────────────────────────────


# ── Utility ──────────────────────────────────────────────────────────────────

def load_test_pairs(img_dir: Path, lbl_dir: Path) -> list[dict]:
    """
    Loads image and label pairs from the given directories.
    Converts YOLO format (center x, center y, width, height) normalized 
    coordinates to absolute pixel coordinates [x1, y1, x2, y2].
    Returns a list of dictionaries containing the image path, image object, and Ground Truth boxes.
    """
    pairs = []
    for img_path in sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")):
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        img  = Image.open(img_path).convert("RGB")
        W, H = img.size
        gt_boxes = []
        for line in lbl_path.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            _, cx, cy, bw, bh = map(float, parts[:5])
            x1 = (cx - bw / 2) * W
            y1 = (cy - bh / 2) * H
            x2 = (cx + bw / 2) * W
            y2 = (cy + bh / 2) * H
            gt_boxes.append([x1, y1, x2, y2])
        pairs.append({"path": img_path, "image": img, "gt": gt_boxes, "W": W, "H": H})
    return pairs


def iou(box_a: list, box_b: list) -> float:
    """
    Computes the Intersection over Union (IoU) between two bounding boxes.
    Boxes must be in [x1, y1, x2, y2] format.
    """
    x1 = max(box_a[0], box_b[0]); y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2]); y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def multi_match(pred_boxes: list, gt_boxes: list,
                iou_thr: float) -> tuple[list[bool], list[bool]]:
    """
    Multi-matching strategy based on professor directives:
      is_tp_pred[i] = True if prediction i matches at least one GT.
      is_fn_gt[j]   = True if GT j is not matched by any prediction.
    Returns these two boolean lists.
    """
    mat_pred = [[] for _ in range(len(pred_boxes))]
    mat_gt   = [[] for _ in range(len(gt_boxes))]
    for pi, pb in enumerate(pred_boxes):
        for gi, gb in enumerate(gt_boxes):
            v = iou(pb, gb)
            if (iou_thr > 0 and v >= iou_thr) or (iou_thr == 0 and v > 0):
                mat_pred[pi].append(gi)
                mat_gt[gi].append(pi)
    is_tp_pred = [len(lst) > 0 for lst in mat_pred]
    is_fn_gt   = [len(lst) == 0 for lst in mat_gt]
    return is_tp_pred, is_fn_gt


def compute_metrics(all_results: list[dict], iou_thr: float,
                    detector: str, prompt: str) -> dict:
    """
    Computes metrics for the YOLO+Cosmos pipeline with text response.

    Per-image flow:
      1. BBs with Cosmos response "yes" → kept
      2. BBs with Cosmos response "no"  → discarded
      3. Multi-match kept BBs vs GT:
         - TP = Kept BB that matches a GT
         - FP = Kept BB that does not match any GT
         - FN = GT not matched by any kept BB

    Additional Cosmos vs YOLO metrics:
      - cosmos_yes_total  : total BBs confirmed by Cosmos
      - cosmos_no_total   : total BBs rejected by Cosmos
      - yolo_total        : total BBs predicted by YOLO
      - cosmos_yes_were_tp: how many Cosmos "yes" were real TPs (YOLO was right)
      - cosmos_yes_were_fp: how many Cosmos "yes" were real FPs (YOLO was wrong)
      - cosmos_no_were_tp : how many Cosmos "no" were real TPs (Cosmos missed a real gap)
      - cosmos_no_were_fp : how many Cosmos "no" were real FPs (Cosmos corrected YOLO)
    """
    total_gt = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    # Breakdown of FN causes, computed correctly PER GT
    # (not per box — see note below on why this is important):
    fn_missed_by_yolo     = 0  # GT never covered by ANY YOLO box (even before Cosmos)
    fn_rejected_by_cosmos = 0  # GT covered by at least one YOLO box, but Cosmos rejected them all

    yolo_total       = 0
    cosmos_yes_total   = 0
    cosmos_no_total    = 0
    cosmos_yes_were_tp = 0
    cosmos_yes_were_fp = 0
    cosmos_no_were_tp  = 0
    cosmos_no_were_fp  = 0

    for r in all_results:
        gt_boxes   = r["gt"]
        pred_boxes = r["preds"][detector]
        responses  = r["responses"][detector][prompt]  # list of "yes"/"no"

        n_gt = len(gt_boxes)
        total_gt   += n_gt
        yolo_total += len(pred_boxes)

        # Real TP/FP classification on ALL YOLO predictions (ground truth match)
        is_tp_real, _ = multi_match(pred_boxes, gt_boxes, iou_thr)

        # Cosmos yes/no count vs reality
        for resp, is_real_tp in zip(responses, is_tp_real):
            if resp == "yes":
                cosmos_yes_total += 1
                if is_real_tp:
                    cosmos_yes_were_tp += 1
                else:
                    cosmos_yes_were_fp += 1
            else:  # "no"
                cosmos_no_total += 1
                if is_real_tp:
                    cosmos_no_were_tp += 1
                else:
                    cosmos_no_were_fp += 1

        # Filter: keep only BBs with a "yes" response
        kept_boxes = [b for b, r in zip(pred_boxes, responses) if r == "yes"]

        # ── Breakdown of FN cause, calculated PER GT ─────────────────────────
        # Match 1: the GT is covered by AT LEAST ONE YOLO box, regardless of
        #          Cosmos's response (i.e. whether YOLO proposed it or not)
        # Match 2: the GT is covered by at least one KEPT box ("yes" response)
        #
        # If Match1 = False → YOLO never found it (Cosmos has nothing to do with it)
        # If Match1 = True  and Match2 = False → YOLO had found it, but
        #                                        Cosmos rejected ALL the
        #                                        boxes covering it
        # If Match2 = True  → it's a TP, not an FN
        #
        # By construction: fn_missed_by_yolo + fn_rejected_by_cosmos == total_fn
        if pred_boxes:
            _, is_fn_gt_all_yolo = multi_match(pred_boxes, gt_boxes, iou_thr)
        else:
            is_fn_gt_all_yolo = [True] * n_gt  # no YOLO box -> all GTs "never proposed"

        # Multi-match on kept BBs
        if kept_boxes:
            is_tp_kept, is_fn_gt_kept = multi_match(kept_boxes, gt_boxes, iou_thr)
            total_tp += sum(is_tp_kept)
            total_fp += sum(not tp for tp in is_tp_kept)
            total_fn += sum(is_fn_gt_kept)
        else:
            is_fn_gt_kept = [True] * n_gt
            total_fn += n_gt

        for missed_all, missed_kept in zip(is_fn_gt_all_yolo, is_fn_gt_kept):
            if not missed_kept:
                continue  # it is a TP, not an FN, we don't care here
            if missed_all:
                fn_missed_by_yolo += 1       # YOLO had never found it
            else:
                fn_rejected_by_cosmos += 1   # YOLO found it, Cosmos discarded it

    fpr = total_fp / total_gt if total_gt > 0 else 0.0
    fnr = total_fn / total_gt if total_gt > 0 else 0.0
    tpr = total_tp / total_gt if total_gt > 0 else 0.0

    return {
        # Main pipeline metrics
        "FPR": fpr, "FNR": fnr, "TPR": tpr,
        "TP": total_tp, "FP": total_fp, "FN": total_fn, "GT": total_gt,
        # Breakdown of FN cause (per GT, exclusive and complete by construction)
        "fn_missed_by_yolo":     fn_missed_by_yolo,
        "fn_rejected_by_cosmos": fn_rejected_by_cosmos,
        # Cosmos vs YOLO metrics
        "yolo_total":       yolo_total,
        "cosmos_yes_total":   cosmos_yes_total,
        "cosmos_no_total":    cosmos_no_total,
        "cosmos_yes_were_tp": cosmos_yes_were_tp,   # Cosmos YES + was real TP ✓
        "cosmos_yes_were_fp": cosmos_yes_were_fp,   # Cosmos YES + was real FP ✗
        "cosmos_no_were_tp":  cosmos_no_were_tp,    # Cosmos NO  + was real TP ✗ (Cosmos wrong)
        "cosmos_no_were_fp":  cosmos_no_were_fp,    # Cosmos NO  + was real FP ✓ (Cosmos corrects)
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    """
    Main execution pipeline: parses CLI arguments, loads the two YOLO
    detectors and Cosmos via vLLM, runs YOLO inference and then Cosmos scoring 
    for every detector×prompt combination, computes the conf×IoU metric grids, 
    and writes JSON + text summaries.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--noaug_weights", default=WEIGHTS_NOAUG)
    parser.add_argument("--aug_weights",   default=WEIGHTS_AUG)
    parser.add_argument("--vllm_url",      default=VLLM_URL_DEFAULT,
                        help="URL of the already running vLLM server (vllm serve ...)")
    parser.add_argument("--test_images",   default=TEST_IMAGES)
    parser.add_argument("--test_labels",   default=TEST_LABELS)
    parser.add_argument("--out_dir",       default=OUT_DIR_DEFAULT)
    parser.add_argument("--conf",     nargs="+", type=float, default=CONF_THRESHOLDS)
    parser.add_argument("--iou_thr",  nargs="+", type=float, default=IOU_THRESHOLDS)
    parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load detectors ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Evaluation pipeline YOLO + Cosmos v4.1 (textual response, v2 module)")
    print("=" * 70)
    print(f"[1/3] Loading noaug detector ← '{args.noaug_weights}'")
    yolo_noaug = YOLO(args.noaug_weights)
    print(f"[2/3] Loading aug detector   ← '{args.aug_weights}'")
    yolo_aug   = YOLO(args.aug_weights)

    # ── Load Cosmos (via vLLM server, no longer local transformers) ───────────
    print(f"[3/3] Connecting to vLLM server ← '{args.vllm_url}'")
    from src.models.vllm_inference_module_v4 import CosmosInferenceModule
    cosmos = CosmosInferenceModule(
        base_url = args.vllm_url,
    )

    # ── Load test set ──────────────────────────────────────────────────────
    pairs = load_test_pairs(Path(args.test_images), Path(args.test_labels))
    if args.max_images:
        pairs = pairs[:args.max_images]
    print(f"\n  Test images : {len(pairs)}")
    print(f"  CONF        : {args.conf}")
    print(f"  IOU         : {args.iou_thr}")

    # ── Phase 1: YOLO inference ───────────────────────────────────────────────
    print("\nPhase 1/2: YOLO inference...")
    detectors   = {"noaug": yolo_noaug, "aug": yolo_aug}
    all_results: dict[float, list[dict]] = {}

    for conf in args.conf:
        results_conf = []
        for sample in pairs:
            entry = {
                "path":      sample["path"],
                "gt":        sample["gt"],
                "image":     sample["image"],
                "preds":     {},
                "responses": {},
            }
            for det_name, yolo in detectors.items():
                res = yolo(sample["path"], conf=conf, verbose=False)
                boxes = res[0].boxes.xyxy.cpu().numpy().tolist() \
                        if res and res[0].boxes is not None else []
                entry["preds"][det_name]     = boxes
                entry["responses"][det_name] = {}
            results_conf.append(entry)
        all_results[conf] = results_conf
        print(f"  conf={conf}: YOLO completed on {len(results_conf)} images")

    # ── Phase 2: Cosmos inference ───────────────────────────────────────────────
    print("\nPhase 2/2: Cosmos inference (annotated image + coordinates in prompt)...")

    from src.models.vllm_inference_module_v4 import VLLMServerDownError

    PROGRESS_EVERY = 50  # prints progress every N processed images

    try:
        for conf in args.conf:
            for det_name in detectors:
                for prompt_name in PROMPT_NAMES:
                    print(f"  det={det_name}  conf={conf}  prompt='{prompt_name}'")
                    t0 = time.time()
                    yes_count   = 0
                    no_count    = 0
                    boxes_done  = 0
                    n_images    = len(all_results[conf])

                    for img_idx, entry in enumerate(all_results[conf], start=1):
                        boxes = entry["preds"][det_name]
                        if boxes:
                            resps = cosmos.score_boxes(
                                image       = entry["image"],
                                boxes       = boxes,
                                prompt_name = prompt_name,
                            )
                        else:
                            resps = []
                        entry["responses"][det_name][prompt_name] = resps
                        yes_count  += resps.count("yes")
                        no_count   += resps.count("no")
                        boxes_done += len(boxes)

                        if img_idx % PROGRESS_EVERY == 0 or img_idx == n_images:
                            elapsed   = time.time() - t0
                            rate_box  = elapsed / max(boxes_done, 1)
                            rate_img  = elapsed / img_idx
                            eta_s     = rate_img * (n_images - img_idx)
                            print(
                                f"    [{img_idx}/{n_images}] "
                                f"total_box={boxes_done}  "
                                f"{rate_box:.2f}s/box  "
                                f"{elapsed/60:.1f} min elapsed  "
                                f"ETA {eta_s/60:.1f} min  "
                                f"YES={yes_count} NO={no_count}",
                                flush=True,
                            )

                    elapsed = time.time() - t0
                    n_boxes = sum(len(e["preds"][det_name]) for e in all_results[conf])
                    print(f"    → {n_boxes} boxes in {elapsed:.1f}s "
                          f"({elapsed/max(n_boxes,1):.2f}s/box) | "
                          f"Cosmos: YES={yes_count}  NO={no_count}")

    except VLLMServerDownError as e:
        print(f"\n{'!'*70}")
        print(f"  FATAL ERROR: {e}")
        print(f"{'!'*70}\n")
        print("Interrupting execution. No metrics file will be saved "
              "for this partial run — restart the server and relaunch from scratch.")
        import sys
        sys.exit(1)

    # ── Phase 3: Metrics ─────────────────────────────────────────────────────
    print("\nCalculating metrics...")
    output = {}

    for conf in args.conf:
        for iou_t in args.iou_thr:
            for det_name in detectors:
                for prompt_name in PROMPT_NAMES:
                    key = f"conf={conf}_iou={iou_t}_{det_name}_{prompt_name}"
                    m   = compute_metrics(
                        all_results[conf], iou_t, det_name, prompt_name
                    )
                    m.update({"conf": conf, "iou": iou_t,
                              "detector": det_name, "prompt": prompt_name})
                    output[key] = m
                    print(
                        f"  {det_name:5s} {prompt_name:12s} "
                        f"conf={conf} iou={iou_t} | "
                        f"TPR={m['TPR']:.3f} FPR={m['FPR']:.3f} FNR={m['FNR']:.3f} | "
                        f"YOLO={m['yolo_total']} "
                        f"Cosmos_YES={m['cosmos_yes_total']} "
                        f"Cosmos_NO={m['cosmos_no_total']} | "
                        f"YES_corr={m['cosmos_yes_were_tp']} YES_err={m['cosmos_yes_were_fp']} "
                        f"NO_corr={m['cosmos_no_were_fp']} NO_err={m['cosmos_no_were_tp']} | "
                        f"Total FN={m['FN']} "
                        f"(missed by YOLO={m['fn_missed_by_yolo']}, "
                        f"rejected by Cosmos={m['fn_rejected_by_cosmos']})"
                    )

    # ── Save JSON ───────────────────────────────────────────────────────────
    out_json = out_dir / "metrics_v4_1.json"
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved in: {out_json}")

    # ── Save human-readable summary ─────────────────────────────────────────────
    out_txt = out_dir / "metrics_v4_1_summary.txt"
    with open(out_txt, "w") as f:
        f.write(
            f"{'detector':8s} | {'prompt':12s} | {'conf':4s} | {'iou':4s} | "
            f"{'TPR':5s} | {'FPR':5s} | {'FNR':5s} | "
            f"{'YOLO_tot':8s} | {'C_YES':5s} | {'C_NO':5s} | "
            f"{'YES_TP':6s} | {'YES_FP':6s} | {'NO_FP':5s} | {'NO_TP':5s}\n"
        )
        f.write("-" * 130 + "\n")
        for key, m in output.items():
            f.write(
                f"{m['detector']:8s} | {m['prompt']:12s} | {m['conf']:.2f} | "
                f"{m['iou']:.2f} | "
                f"{m['TPR']:.3f} | {m['FPR']:.3f} | {m['FNR']:.3f} | "
                f"{m['yolo_total']:8d} | {m['cosmos_yes_total']:5d} | {m['cosmos_no_total']:5d} | "
                f"{m['cosmos_yes_were_tp']:6d} | {m['cosmos_yes_were_fp']:6d} | "
                f"{m['cosmos_no_were_fp']:5d} | {m['cosmos_no_were_tp']:5d}\n"
            )
    print(f"Text summary: {out_txt}")


if __name__ == "__main__":
    main()