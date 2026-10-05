"""
diag_v6_inspect_reasoning.py
--------------------------------
Gira su un campione di immagini, trova box che sono VERI TP reali (matchano
una GT), e stampa il <think> completo del modello per quei casi — per
capire perche' Cosmos (con reasoning attivo) dice "no" cosi' spesso anche
su scaffali che sono davvero vuoti.

Uso:
    python3 diag_v6_inspect_reasoning.py --max_images 30 --n_samples 8
"""

import argparse
import os
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

_PROJECT_ROOT = os.path.expanduser("~/Progetto_tesi_Cosmos")
WEIGHTS_AUG   = f"{_PROJECT_ROOT}/weights/phase1_baseline_comune/aug/best.pt"
TEST_IMAGES   = f"{_PROJECT_ROOT}/dataset_test_ripulito/images"
TEST_LABELS   = f"{_PROJECT_ROOT}/dataset_test_ripulito/labels"
VLLM_URL      = "http://localhost:8000"


def load_test_pairs(img_dir, lbl_dir):
    pairs = []
    for img_path in sorted(Path(img_dir).glob("*.jpg")) + sorted(Path(img_dir).glob("*.png")):
        lbl_path = Path(lbl_dir) / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        gt_boxes = []
        for line in lbl_path.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            _, cx, cy, bw, bh = map(float, parts[:5])
            x1, y1 = (cx-bw/2)*W, (cy-bh/2)*H
            x2, y2 = (cx+bw/2)*W, (cy+bh/2)*H
            gt_boxes.append([x1, y1, x2, y2])
        pairs.append({"path": img_path, "image": img, "gt": gt_boxes})
    return pairs


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    if inter == 0:
        return 0.0
    area_a = (a[2]-a[0])*(a[3]-a[1])
    area_b = (b[2]-b[0])*(b[3]-b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def is_real_tp(box, gt_boxes, iou_thr=0.0):
    for g in gt_boxes:
        v = iou(box, g)
        if (iou_thr > 0 and v >= iou_thr) or (iou_thr == 0 and v > 0):
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_images", type=int, default=30)
    parser.add_argument("--n_samples", type=int, default=8,
                        help="Quanti casi TP-reale-ma-risposta-no mostrare")
    parser.add_argument("--prompt", default="context", choices=["context", "no_context"])
    args = parser.parse_args()

    print("Carico YOLO...")
    yolo = YOLO(WEIGHTS_AUG)

    print("Connessione a vLLM...")
    from src.models.vllm_inference_module_v4 import CosmosInferenceModule
    cosmos = CosmosInferenceModule(base_url=VLLM_URL)

    pairs = load_test_pairs(TEST_IMAGES, TEST_LABELS)[:args.max_images]
    print(f"Analizzo {len(pairs)} immagini, prompt='{args.prompt}'...\n")

    shown = 0
    for sample in pairs:
        res = yolo(sample["path"], conf=0.25, verbose=False)
        boxes = res[0].boxes.xyxy.cpu().numpy().tolist() if res and res[0].boxes is not None else []
        if not boxes:
            continue

        for box in boxes:
            real_tp = is_real_tp(box, sample["gt"])
            if not real_tp:
                continue  # ci interessano solo i veri TP (vera GT)

            resp = cosmos.score_boxes(image=sample["image"], boxes=[box], prompt_name=args.prompt)[0]

            if resp == "no":  # QUESTO e' il caso interessante: vero TP, ma risposta "no"
                shown += 1
                print("=" * 70)
                print(f"CASO {shown}: {sample['path'].name}  box={[round(b,1) for b in box]}")
                print(f"  GT confermata: TP reale (matcha una ground truth)")
                print(f"  Risposta Cosmos: '{resp}'  (SBAGLIATA — doveva essere 'yes')")
                print("-" * 70)
                print("REASONING (<think>):")
                print(cosmos.last_reasoning if cosmos.last_reasoning else "(vuoto — vedi last_full_text)")
                print("-" * 70)
                print("TESTO COMPLETO:")
                print(cosmos.last_full_text[:1500])
                print("=" * 70 + "\n")

                if shown >= args.n_samples:
                    return

    print(f"\nTrovati solo {shown} casi su {args.max_images} immagini (potrebbe servire aumentare --max_images).")


if __name__ == "__main__":
    main()