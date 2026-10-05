#!/usr/bin/env python3
"""
diag_cosmos_speed.py
----------------------
Diagnostica isolata: carica SOLO Cosmos-Reason2 (niente YOLO, niente test
set) e misura il tempo di 3 chiamate consecutive, stampando anche la mappa
dei device per verificare se qualche layer e' finito sulla CPU.

Uso:
    python3 diag_cosmos_speed.py --model /path/a/Cosmos-Reason2-8B
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

    # ── Immagine di test fittizia (nera, con box verde) ─────────────────────
    img = Image.new("RGB", (1280, 960), color=(80, 80, 80))
    draw = ImageDraw.Draw(img)
    draw.rectangle([400, 300, 700, 600], outline=(0, 255, 0), width=3)

    system_prompt = (
        "You are an expert visual inspector for a retail Out-Of-Stock (OOS) "
        "detection system. Respond with exactly one word: 'yes' or 'no'."
    )
    user_prompt = (
        "The image shows a supermarket shelf with a green bounding box "
        "highlighting the area of interest — this is [Region 0]"
        "<box>320,320,560,640</box> (coordinates normalized 0-1024). "
        "Is the highlighted shelf area [Region 0] empty? Answer yes or no."
    )

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": user_prompt},
        ]},
    ]

    print("\n--- Eseguo 3 chiamate consecutive isolate ---")
    for i in range(3):
        t0 = time.time()
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        t_prep = time.time() - t0

        n_vision_tokens = None
        if "image_grid_thw" in inputs:
            thw = inputs["image_grid_thw"][0]
            n_vision_tokens = int(thw[0] * thw[1] * thw[2])

        t1 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        torch.cuda.synchronize()
        t_gen = time.time() - t1

        input_len = inputs["input_ids"].shape[1]
        text = processor.tokenizer.decode(
            out[0, input_len:], skip_special_tokens=True
        ).strip()

        print(f"  [{i+1}] prep={t_prep:.2f}s  generate={t_gen:.2f}s  "
              f"input_tokens_tot={input_len}  vision_tokens={n_vision_tokens}  "
              f"risposta='{text}'")


if __name__ == "__main__":
    main()