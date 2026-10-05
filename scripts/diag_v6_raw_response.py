"""
diag_v6_raw_response.py
--------------------------
Manda UNA richiesta al server vLLM con il prompt think/answer della v6,
e stampa il JSON grezzo completo (senza nessun parsing), per capire
esattamente cosa succede al tag <answer> nella risposta.

Uso:
    python3 diag_v6_raw_response.py
"""

import base64
import io
import json

import requests
from PIL import Image, ImageDraw

SYSTEM_PROMPT = (
    "You are an expert visual inspector for a retail Out-Of-Stock (OOS) "
    "detection system. You analyze surveillance camera images from "
    "supermarket aisles to determine whether a specific shelf section is "
    "EMPTY of products (an OOS gap).\n\n"
    "First, think through the evidence step by step inside <think></think> "
    "tags — describe what you actually see. Then give your final answer "
    "inside <answer></answer> tags, containing ONLY the single word "
    "'yes' or 'no' — nothing else inside the answer tags."
)

USER_PROMPT = (
    "The image shows a supermarket shelf with a green bounding box "
    "highlighting the area of interest. Is the highlighted shelf area "
    "empty? Think it through, then answer inside <answer></answer> tags."
)


def make_image() -> Image.Image:
    img = Image.new("RGB", (960, 720), color=(90, 90, 90))
    draw = ImageDraw.Draw(img)
    draw.rectangle([300, 250, 600, 500], outline=(0, 255, 0), width=3)
    return img


def image_to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=100)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def main():
    img = make_image()
    payload = {
        "model": "cosmos-reason2-8b",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_to_data_uri(img)}},
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
        ],
        "max_tokens": 700,
        "temperature": 0,
        "top_p": 1.0,
    }

    resp = requests.post("http://localhost:8000/v1/chat/completions", json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    print("=" * 70)
    print("JSON COMPLETO DELLA RISPOSTA:")
    print("=" * 70)
    print(json.dumps(data, indent=2))
    print("=" * 70)

    msg = data["choices"][0]["message"]
    print(f"\ncontent (repr):   {repr(msg.get('content'))}")
    print(f"\nreasoning (repr): {repr(msg.get('reasoning'))}")
    print(f"\nfinish_reason:    {data['choices'][0].get('finish_reason')}")
    print(f"\ncompletion_tokens: {data.get('usage', {}).get('completion_tokens')}")


if __name__ == "__main__":
    main()