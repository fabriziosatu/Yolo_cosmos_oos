"""
diag_vllm_resolution_sweep.py
--------------------------------
Misura il tempo di risposta del server vLLM (Cosmos-Reason2-8B) al variare
della risoluzione massima dell'immagine intera (MAX_FULL_SIZE), per capire
se alzarla (es. 1280 -> 1536/1600) avrebbe un impatto reale sui tempi con
vLLM (a differenza della curva gia' misurata su transformers, che NON e'
detto si applichi identica qui).

Usa immagini sintetiche (rumore + box verde), stesso principio dello sweep
gia' fatto su transformers — il tempo dipende dalla risoluzione finale in
pixel, non dal contenuto specifico dell'immagine.

Per ogni risoluzione: manda N richieste dual-image (full+crop) identiche
a quelle della pipeline reale, misura il tempo medio.

Uso:
    python3 diag_vllm_resolution_sweep.py --url http://localhost:8000 --n 10
"""

import argparse
import base64
import io
import time

import requests
from PIL import Image, ImageDraw

SYSTEM_PROMPT = (
    "You are an expert visual inspector for a retail Out-Of-Stock (OOS) "
    "detection system. Respond with exactly one word: 'yes' or 'no'."
)

USER_PROMPT = (
    "The first image shows the full shelf view with a green bounding box "
    "highlighting the area of interest — this is [Region 0]"
    "<box>320,320,560,640</box> (coordinates normalized 0-1024). "
    "The second image is a close-up crop of that same area. "
    "Is the highlighted shelf area [Region 0] empty? Answer yes or no."
)


def make_image(side: int, seed: int) -> Image.Image:
    """Crea un'immagine sintetica con rumore + box verde, a una data risoluzione."""
    import numpy as np
    h = int(side * 0.75)
    rng = np.random.RandomState(seed)
    arr = rng.randint(40, 160, (h, side, 3)).astype("uint8")
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    bw, bh = int(side * 0.23), int(h * 0.31)
    cx, cy = side // 2, h // 2
    draw.rectangle([cx - bw, cy - bh, cx + bw, cy + bh], outline=(0, 255, 0), width=3)
    return img


def make_crop(full_img: Image.Image) -> Image.Image:
    """Estrae un crop centrale (simula il crop reale con padding 20%)."""
    W, H = full_img.size
    bw, bh = int(W * 0.23 * 1.4), int(H * 0.31 * 1.4)  # box + ~20% padding
    cx, cy = W // 2, H // 2
    return full_img.crop((cx - bw, cy - bh, cx + bw, cy + bh))


def image_to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def send_request(base_url: str, model: str, full_img: Image.Image, crop_img: Image.Image) -> tuple[float, int]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_to_data_uri(full_img)}},
                    {"type": "image_url", "image_url": {"url": image_to_data_uri(crop_img)}},
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
        ],
        "max_tokens": 300,
        "temperature": 0,
        "top_p": 1.0,
    }

    t0 = time.time()
    resp = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=120)
    elapsed = time.time() - t0
    resp.raise_for_status()
    data = resp.json()
    prompt_tokens = data.get("usage", {}).get("prompt_tokens", -1)
    return elapsed, prompt_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="cosmos-reason2-8b")
    parser.add_argument("--n", type=int, default=10, help="Richieste per ogni risoluzione")
    args = parser.parse_args()

    resolutions = [896, 1024, 1280, 1536, 1600]

    print(f"\n--- Sweep risoluzione su vLLM ({args.n} richieste per valore) ---\n")
    results = []

    for side in resolutions:
        times = []
        prompt_tok = None
        for i in range(args.n):
            full_img = make_image(side, seed=i)
            crop_img = make_crop(full_img)
            elapsed, ptok = send_request(args.url, args.model, full_img, crop_img)
            times.append(elapsed)
            prompt_tok = ptok

        avg = sum(times) / len(times)
        results.append((side, avg, prompt_tok))
        print(f"  MAX_FULL_SIZE={side:4d}px  prompt_tokens={prompt_tok:5d}  "
              f"tempo medio={avg:.3f}s  (min={min(times):.3f}s max={max(times):.3f}s)")

    print("\n--- Riepilogo e stima impatto sul grid completo ---")
    baseline_time = results[2][1]  # 1280px = baseline attuale
    total_boxes_est = 15671  # stima da run precedenti (8 combinazioni, dataset ripulito)

    for side, avg, ptok in results:
        delta_pct = 100 * (avg - baseline_time) / baseline_time
        hours_est = (total_boxes_est * avg) / 3600
        marker = "  <- attuale" if side == 1280 else ""
        print(f"  {side:4d}px: {avg:.3f}s/box ({delta_pct:+.1f}% vs 1280px)  "
              f"-> stima grid completo: {hours_est:.2f} ore{marker}")


if __name__ == "__main__":
    main()