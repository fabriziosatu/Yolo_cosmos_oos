#!/usr/bin/env python3
"""
test_vllm_speed.py
---------------------
Client di test per il server vLLM (Cosmos-Reason2-8B).
Misura il tempo di risposta per una richiesta con immagine (full-size,
box verde disegnata), usando l'endpoint OpenAI-compatible /v1/chat/completions.

Uso:
    python3 test_vllm_speed.py --url http://localhost:8000 --n 5
"""

import argparse
import base64
import io
import time

import requests
from PIL import Image, ImageDraw


def make_test_image(side: int = 1280, seed: int = 0) -> str:
    """Crea un'immagine di test con box verde e rumore casuale (seed diverso
    ad ogni chiamata), per evitare hit di prefix-cache artificiali in vLLM.
    Ritorna base64 data URI."""
    import random
    rng = random.Random(seed)

    h = int(side * 0.75)
    # Colore di sfondo leggermente diverso ad ogni chiamata + rumore pixel
    base_color = (
        80 + rng.randint(-20, 20),
        80 + rng.randint(-20, 20),
        80 + rng.randint(-20, 20),
    )
    img = Image.new("RGB", (side, h), color=base_color)

    # Aggiunge rumore casuale per rendere ogni immagine byte-diversa
    # (evita hit di prefix cache che falserebbero il benchmark)
    import numpy as np
    arr = np.array(img).astype(np.int16)
    noise = np.random.RandomState(seed).randint(-15, 15, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    draw = ImageDraw.Draw(img)
    bw, bh = int(side * 0.23), int(h * 0.31)
    cx, cy = side // 2, h // 2
    draw.rectangle([cx - bw, cy - bh, cx + bw, cy + bh], outline=(0, 255, 0), width=3)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


SYSTEM_PROMPT = (
    "You are an expert visual inspector for a retail Out-Of-Stock (OOS) "
    "detection system. Respond with exactly one word: 'yes' or 'no'."
)

USER_PROMPT = (
    "The image shows a supermarket shelf with a green bounding box "
    "highlighting the area of interest — this is [Region 0]"
    "<box>320,320,560,640</box> (coordinates normalized 0-1024). "
    "Is the highlighted shelf area [Region 0] empty? Answer yes or no."
)


def send_request(base_url: str, model: str, image_data_uri: str, max_tokens: int, debug: bool = False) -> tuple[str, float]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_uri}},
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
        ],
        "max_tokens": max_tokens,
        "temperature": 0,   # decodifica deterministica, greedy
        "top_p": 1.0,
    }

    t0 = time.time()
    resp = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=120)
    elapsed = time.time() - t0

    resp.raise_for_status()
    data = resp.json()

    if debug:
        import json as _json
        print("  [DEBUG] risposta grezza:", _json.dumps(data, indent=2)[:2000])

    msg = data["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    reasoning = (msg.get("reasoning") or msg.get("reasoning_content") or "").strip()

    # Con reasoning-parser attivo, la risposta finale puo' comparire in
    # "content" (dopo aver chiuso il tag <think>) oppure, per risposte
    # brevi come yes/no senza reasoning esplicito, direttamente nel campo
    # "reasoning". Controlliamo entrambi.
    finish_reason = data["choices"][0].get("finish_reason", "?")
    text = content if content else (reasoning if reasoning else f"[VUOTO, finish_reason={finish_reason}]")

    return text, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="cosmos-reason2-8b")
    parser.add_argument("--n", type=int, default=5, help="Numero di richieste sequenziali")
    parser.add_argument("--max_tokens", type=int, default=300,
                        help="Token massimi per lasciare completare il ragionamento")
    args = parser.parse_args()

    print(f"\n--- {args.n} richieste SEQUENZIALI (max_tokens={args.max_tokens}) ---")
    print("(ogni richiesta usa un'immagine leggermente diversa, per evitare hit di prefix-cache artificiali)")
    times = []
    for i in range(args.n):
        image_data_uri = make_test_image(1280, seed=i)
        text, elapsed = send_request(
            args.url, args.model, image_data_uri,
            max_tokens=args.max_tokens, debug=(i == 0),
        )
        times.append(elapsed)
        print(f"  [{i+1}] {elapsed:.2f}s  risposta='{text}'")

    avg = sum(times) / len(times)
    print(f"\nMedia sequenziale: {avg:.2f}s/richiesta")

    # Stima per il grid completo (8 combinazioni, ~15671 box totali stimati
    # da precedenti speedtest su transformers)
    total_boxes_est = 15671
    hours_est = (total_boxes_est * avg) / 3600
    print(f"Stima grid completo (8 combo, sequenziale): {hours_est:.1f} ore")


if __name__ == "__main__":
    main()