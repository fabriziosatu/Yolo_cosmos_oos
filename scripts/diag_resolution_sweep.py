#!/usr/bin/env python3
"""
diag_resolution_sweep.py
---------------------------
Testa il tempo di generate() di Cosmos-Reason2 su piu' risoluzioni immagine
(384, 512, 640, 896, 1280 px di lato), per trovare il MAX_FULL_SIZE che
bilancia velocita' e dettaglio visivo per il task OOS.

Uso:
    python3 diag_resolution_sweep.py --model /path/a/Cosmos-Reason2-2B
"""

import argparse
import time

import torch
from PIL import Image, ImageDraw
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    print("Carico processor...")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    print("Carico modello SENZA device_map (diretto su cuda:0)...")
    t0 = time.time()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).to("cuda:0")
    model.eval()
    print(f"Caricamento: {time.time()-t0:.1f}s")

    # ── Verifica se qualche parte del modello e' su CPU ─────────────────────
    if hasattr(model, "hf_device_map"):
        devices = set(model.hf_device_map.values())
        print(f"\nDevice map — device unici usati: {devices}")
        if any(str(d) == "cpu" for d in devices):
            print("  ⚠️  ATTENZIONE: parte del modello e' su CPU! "
                  "Questo spiegherebbe la lentezza.")
            for name, dev in model.hf_device_map.items():
                if str(dev) == "cpu":
                    print(f"    -> {name}: {dev}")
        else:
            print("  OK: tutti i layer sono su GPU.")
    else:
        print("\n(Nessun hf_device_map trovato — modello non partizionato)")

    print(f"\nVRAM allocata: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"VRAM riservata: {torch.cuda.memory_reserved()/1e9:.2f} GB")

    print("\n--- Sweep risoluzioni: trova il compromesso velocità/dettaglio ---")
    system_prompt_oos = (
        "You are an expert visual inspector for a retail Out-Of-Stock (OOS) "
        "detection system. Respond with exactly one word: 'yes' or 'no'."
    )
    user_prompt_oos = (
        "The image shows a supermarket shelf with a green bounding box "
        "highlighting the area of interest — this is [Region 0]"
        "<box>320,320,560,640</box> (coordinates normalized 0-1024). "
        "Is the highlighted shelf area [Region 0] empty? Answer yes or no."
    )

    resolutions = [384, 512, 640, 896, 1280]
    results = []

    for side in resolutions:
        h = int(side * 0.75)  # mantiene proporzioni ~4:3 come le foto reali
        img = Image.new("RGB", (side, h), color=(80, 80, 80))
        draw = ImageDraw.Draw(img)
        bw, bh = int(side * 0.23), int(h * 0.31)
        cx, cy = side // 2, h // 2
        draw.rectangle([cx - bw, cy - bh, cx + bw, cy + bh],
                       outline=(0, 255, 0), width=3)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt_oos}]},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": user_prompt_oos},
            ]},
        ]

        t0 = time.time()
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        n_tok = inputs["input_ids"].shape[1]

        results.append((side, n_tok, elapsed))
        print(f"  {side:4d}px  input_tokens={n_tok:5d}  generate={elapsed:6.2f}s")

    print("\n--- Riepilogo (per scegliere il MAX_FULL_SIZE) ---")
    for side, n_tok, elapsed in results:
        boxes_per_combo_est = 2300  # stima da speedtest_v1 scalato a 1679 immagini
        h_totali = (boxes_per_combo_est * elapsed * 8) / 3600  # 8 combinazioni
        print(f"  MAX_FULL_SIZE={side:4d} -> {elapsed:6.2f}s/box -> "
              f"stima grid completo (8 combo): {h_totali:.1f} ore")


if __name__ == "__main__":
    main()