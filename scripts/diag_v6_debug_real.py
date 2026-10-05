"""
diag_v6_debug_real.py
------------------------
Gira sulle prime N immagini del dataset REALE (dual-image, come in
produzione) e stampa il JSON completo della richiesta HTTP alla PRIMA
box per cui l'estrazione di <answer> fallisce — per capire cosa succede
davvero nei casi che il log mostrava come [AMBIGUOUS].

Uso:
    python3 diag_v6_debug_real.py --max_images 50
"""

import argparse
import base64
import io
import json
import os
import re
from pathlib import Path

import requests
from PIL import Image, ImageDraw
from ultralytics import YOLO

_PROJECT_ROOT = os.path.expanduser("~/Progetto_tesi_Cosmos")
WEIGHTS_AUG   = f"{_PROJECT_ROOT}/weights/phase1_baseline_comune/aug/best.pt"
TEST_IMAGES   = f"{_PROJECT_ROOT}/dataset_test_ripulito/images"
VLLM_URL      = "http://localhost:8000"

SYSTEM_PROMPT = (
    "You are an expert visual inspector for a retail Out-Of-Stock (OOS) "
    "detection system. You analyze surveillance camera images from "
    "supermarket aisles to determine whether a specific shelf section is "
    "EMPTY of products (an OOS gap).\n\n"
    "A TRUE empty shelf shows: bare horizontal shelf surface (metal, wood, "
    "or plastic), visible shelf backing or pegboard, and clear depth/"
    "perspective into the empty space. There are NO products, packages, "
    "boxes, bottles, or any merchandise visible.\n\n"
    "You will see two images of the SAME area: (1) the full shelf photo "
    "with a green box marking the region to inspect, and (2) a zoomed-in "
    "crop of that exact region. Use the crop for fine detail and the full "
    "image for context (shelf depth, lighting, surrounding products) to "
    "make your decision.\n\n"
    "First, think through the evidence step by step inside <think></think> "
    "tags — describe what you actually see in the crop and in the full "
    "image, and weigh it against the criteria given in the user prompt. "
    "Then give your final answer inside <answer></answer> tags, containing "
    "ONLY the single word 'yes' or 'no' — nothing else inside the answer "
    "tags."
)

USER_PROMPT = (
    "The first image shows the full shelf view with a green bounding box "
    "highlighting the area of interest — this is [Region 0]"
    "<box>{x1},{y1},{x2},{y2}</box> (coordinates normalized 0-1024). "
    "The second image is a close-up crop of that same area. "
    "Is the highlighted shelf area [Region 0] empty (an OOS gap, no "
    "products)? Think it through, then answer inside <answer></answer> tags."
)


def resize_if_needed(img, max_side=1280):
    W, H = img.size
    if max(W, H) <= max_side:
        return img
    scale = max_side / max(W, H)
    return img.resize((int(W*scale), int(H*scale)), Image.LANCZOS)


def draw_box(img, box):
    img_r = resize_if_needed(img)
    W_o, H_o = img.size
    W_r, H_r = img_r.size
    sx, sy = W_r/W_o, H_r/H_o
    out = img_r.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    x1, y1 = max(0, int(box[0]*sx)), max(0, int(box[1]*sy))
    x2, y2 = min(W_r-1, int(box[2]*sx)), min(H_r-1, int(box[3]*sy))
    if x2 > x1 and y2 > y1:
        draw.rectangle([x1, y1, x2, y2], outline=(0,255,0), width=3)
    return out


def extract_crop(img, box, padding=0.20, min_size=28):
    W, H = img.size
    bw, bh = box[2]-box[0], box[3]-box[1]
    cx1 = max(0, int(box[0]-bw*padding)); cy1 = max(0, int(box[1]-bh*padding))
    cx2 = min(W-1, int(box[2]+bw*padding)); cy2 = min(H-1, int(box[3]+bh*padding))
    crop = img.crop((cx1, cy1, cx2, cy2)).convert("RGB")
    cw, ch = crop.size
    if cw < min_size or ch < min_size:
        scale = max(min_size/max(cw,1), min_size/max(ch,1))
        crop = crop.resize((max(int(cw*scale),min_size), max(int(ch*scale),min_size)), Image.LANCZOS)
    return crop


def cosmos_coords(box, W, H):
    return {
        "x1": str(int((box[0]/W)*1024)), "y1": str(int((box[1]/H)*1024)),
        "x2": str(int((box[2]/W)*1024)), "y2": str(int((box[3]/H)*1024)),
    }


def to_data_uri(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=100)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_images", type=int, default=50)
    args = parser.parse_args()

    print("Carico YOLO...")
    yolo = YOLO(WEIGHTS_AUG)

    img_paths = (sorted(Path(TEST_IMAGES).glob("*.jpg")) +
                 sorted(Path(TEST_IMAGES).glob("*.png")))[:args.max_images]

    print(f"Cerco il primo caso di fallimento su {len(img_paths)} immagini...\n")

    checked = 0
    for img_path in img_paths:
        image = Image.open(img_path).convert("RGB")
        res = yolo(img_path, conf=0.25, verbose=False)
        boxes = res[0].boxes.xyxy.cpu().numpy().tolist() if res and res[0].boxes is not None else []
        W, H = image.size

        for box in boxes:
            annotated = draw_box(image, box)
            crop = extract_crop(image, box)
            coords = cosmos_coords(box, W, H)
            prompt = USER_PROMPT.format(**coords)

            payload = {
                "model": "cosmos-reason2-8b",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": to_data_uri(annotated)}},
                        {"type": "image_url", "image_url": {"url": to_data_uri(crop)}},
                        {"type": "text", "text": prompt},
                    ]},
                ],
                "max_tokens": 700, "temperature": 0, "top_p": 1.0,
            }

            resp = requests.post(f"{VLLM_URL}/v1/chat/completions", json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            checked += 1

            msg = data["choices"][0]["message"]
            content = (msg.get("content") or "").strip()
            reasoning = (msg.get("reasoning") or msg.get("reasoning_content") or "").strip()
            full_text = content if content else reasoning

            match = re.search(r"<answer>(.*?)</answer>", full_text, re.IGNORECASE | re.DOTALL)

            if not match:
                print("!" * 70)
                print(f"CASO FALLITO TROVATO — immagine: {img_path.name}, box: {box}")
                print("!" * 70)
                print("\nJSON COMPLETO:")
                print(json.dumps(data, indent=2))
                print(f"\ncontent (repr):   {repr(msg.get('content'))}")
                print(f"\nreasoning (repr): {repr(msg.get('reasoning'))}")
                print(f"\nfinish_reason: {data['choices'][0].get('finish_reason')}")
                print(f"completion_tokens: {data.get('usage', {}).get('completion_tokens')}")
                print(f"\n(Controllate {checked} box prima di trovare questo fallimento)")
                return

    print(f"Nessun fallimento trovato su {checked} box controllate.")


if __name__ == "__main__":
    main()