"""
diag_finish_reason.py
------------------------
Verifica se Cosmos-Reason2 (via vLLM) tronca mai le risposte prima di
arrivare a "yes"/"no", controllando il campo finish_reason di ogni
richiesta HTTP.

Gira su UNA sola combinazione (detector=aug, prompt=context, conf=0.25)
sull'intero test set — non l'intera griglia da 8 combinazioni, perche'
per questa diagnostica basta un campione rappresentativo e vogliamo un
risultato rapido.

Usage:
    python3 diag_finish_reason.py [--max_images N]
"""

import argparse
import json
import os
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

_PROJECT_ROOT = os.path.expanduser("~/Progetto_tesi_Cosmos")

WEIGHTS_AUG   = f"{_PROJECT_ROOT}/weights/phase1_baseline_comune/aug/best.pt"
TEST_IMAGES   = f"{_PROJECT_ROOT}/dataset_test_ripulito/images"
TEST_LABELS   = f"{_PROJECT_ROOT}/dataset_test_ripulito/labels"
OUT_DIR       = f"{_PROJECT_ROOT}/results/diag_finish_reason"
VLLM_URL      = "http://localhost:8000"

CONF   = 0.25
PROMPT = "context"


def load_test_pairs(img_dir: Path, lbl_dir: Path) -> list[dict]:
    pairs = []
    for img_path in sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")):
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        img = Image.open(img_path).convert("RGB")
        pairs.append({"path": img_path, "image": img})
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/2] Carico detector aug ← '{WEIGHTS_AUG}'")
    yolo = YOLO(WEIGHTS_AUG)

    print(f"[2/2] Connessione al server vLLM ← '{VLLM_URL}'")
    from src.models.vllm_inference_module_diag import CosmosInferenceModule
    cosmos = CosmosInferenceModule(base_url=VLLM_URL)

    pairs = load_test_pairs(Path(TEST_IMAGES), Path(TEST_LABELS))
    if args.max_images:
        pairs = pairs[:args.max_images]
    print(f"\n  Immagini: {len(pairs)}  |  conf={CONF}  |  prompt='{PROMPT}'\n")

    total_boxes = 0
    for i, sample in enumerate(pairs, start=1):
        res = yolo(sample["path"], conf=CONF, verbose=False)
        boxes = res[0].boxes.xyxy.cpu().numpy().tolist() \
                if res and res[0].boxes is not None else []
        if boxes:
            cosmos.score_boxes(image=sample["image"], boxes=boxes, prompt_name=PROMPT)
            total_boxes += len(boxes)

        if i % 100 == 0 or i == len(pairs):
            print(f"  [{i}/{len(pairs)}] box processate finora: {total_boxes}")

    cosmos.print_diagnostic_summary()

    # Salva anche su file per ispezione successiva
    out_json = out_dir / "finish_reason_report.json"
    with open(out_json, "w") as f:
        json.dump({
            "finish_reason_counts": cosmos.finish_reason_counts,
            "truncated_boxes": cosmos.truncated_boxes,
            "total_boxes": total_boxes,
            "config": {"detector": "aug", "conf": CONF, "prompt": PROMPT},
        }, f, indent=2)
    print(f"\nReport salvato in: {out_json}")


if __name__ == "__main__":
    main()