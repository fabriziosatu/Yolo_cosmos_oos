"""
vllm_inference_module.py  —  v1.0
-------------------------------------
Cosmos-Reason2-8B via vLLM server (HTTP), no longer local transformers.

Why this module: the implementation of Cosmos-Reason2/Qwen3-VL in
transformers proved to be ~250x slower than vLLM for the exact same
task (measured: ~36-40s/box with transformers vs ~0.15s/box with vLLM, on
the same 1280px image, same H100 hardware). Cause: DeepStack (multi-level
injection of visual tokens) and Interleaved-MRoPE managed inefficiently
by the "research" implementation of transformers.

The vLLM server must already be running (see the vllm serve command in
the project documentation) before using this module.

IDENTICAL interface to CosmosInferenceModule (cosmos_inference_module.py):
same score_boxes() method, same return format (list[str] of
"yes"/"no"), same OOS prompts with [Region 0]<box>x1,y1,x2,y2</box> coordinates
in 0-1024 scale — ONLY how the model is queried changes (HTTP instead
of local forward pass).

DIFFERENCES compared to cosmos_inference_module.py:
  - No local model loading: the constructor connects to an already
    running vLLM server (base_url)
  - temperature=0, top_p=1.0 explicit in the request, to get
    deterministic greedy decoding (the default generation_config.json
    of the model would use temperature=0.7, random sampling)
  - max_tokens=300 (not 5!): the server has --reasoning-parser qwen3 active,
    so the final answer of the model can appear in the "reasoning" field
    of the JSON response instead of in "content" — enough tokens are needed
    for the model to complete the short thought before answering.
    Verified: with direct yes/no prompts the model usually responds in
    2 tokens inside "reasoning", without long serious chains of thought.
"""

from __future__ import annotations
import base64
import io

import requests
from PIL import Image, ImageDraw


# ─────────────────────────────────────────────────────────────────────────────
#  SYSTEM PROMPT  (identical to cosmos_inference_module.py v2.2)
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert visual inspector for a retail Out-Of-Stock (OOS) "
    "detection system. You analyze surveillance camera images from "
    "supermarket aisles to determine whether a specific shelf section is "
    "EMPTY of products (an OOS gap).\n\n"
    "A TRUE empty shelf shows: bare horizontal shelf surface (metal, wood, "
    "or plastic), visible shelf backing or pegboard, and clear depth/"
    "perspective into the empty space. There are NO products, packages, "
    "boxes, bottles, or any merchandise visible.\n\n"
    "You will be shown the full shelf photo with a green box marking the "
    "region to inspect.\n"
    "Respond with exactly one word: 'yes' (the shelf is empty / OOS) or "
    "'no' (it is not). Do not add any other words."
)

# ─────────────────────────────────────────────────────────────────────────────
#  PROMPT TEMPLATES  (identical to cosmos_inference_module.py v2.2)
#  {x1} {y1} {x2} {y2} are replaced with coordinates in 0-1024 scale
# ─────────────────────────────────────────────────────────────────────────────
PROMPT_TEMPLATES: dict[str, str] = {

    "no_context": (
        "The image shows a supermarket shelf with a green bounding box "
        "highlighting the area of interest — this is [Region 0]"
        "<box>{x1},{y1},{x2},{y2}</box> (coordinates normalized 0-1024). "
        "Is the highlighted shelf area [Region 0] empty (an OOS gap, no "
        "products)? Answer yes or no."
    ),

    "context": (
        "The image shows a supermarket shelf with a green bounding box "
        "marking the area of interest — this is [Region 0]"
        "<box>{x1},{y1},{x2},{y2}</box> (coordinates normalized 0-1024).\n\n"
        "Your goal is to detect Out-Of-Stock (OOS) shelf gaps — areas "
        "where products are completely missing from the shelf.\n\n"
        "STEP 1 — Look at [Region 0]. Can you see a flat, bare shelf "
        "surface with visible depth (the back wall, pegboard, or shadow "
        "of the empty space)? If yes, this supports OOS/EMPTY.\n\n"
        "STEP 2 — Check [Region 0] does NOT match any of these common "
        "false-positive categories:\n"
        "  - Shelf STRUCTURE only: metallic uprights, edge profile, "
        "frame borders, or dividers seen edge-on (this is shelf "
        "hardware, not empty space)\n"
        "  - REFRIGERATOR/FREEZER glass door or frozen food cabinet "
        "surface, possibly with reflections\n"
        "  - MOTION BLUR: the region is blurry/smeared from camera "
        "movement, details cannot be confirmed\n"
        "  - DISTANT/DARK shelf that LOOKS empty due to poor lighting, "
        "but products are likely still there (check the surrounding "
        "rows in the rest of the image)\n"
        "  - PRODUCTS present but small, dark-colored, or partially "
        "out of frame\n\n"
        "Answer YES only if step 1 is satisfied AND none of the step 2 "
        "false-positive cases apply.\n"
        "Answer NO if [Region 0] matches any step 2 case, or if you "
        "cannot clearly confirm bare shelf surface.\n"
        "Answer with exactly one word: yes or no."
    ),
}
# ─────────────────────────────────────────────────────────────────────────────

COLOR_TARGET  = (0, 255, 0)
BB_WIDTH      = 3
MAX_FULL_SIZE = 1280
COSMOS_COORD_SCALE = 1024


def _resize_if_needed(img: Image.Image, max_side: int = MAX_FULL_SIZE) -> Image.Image:
    """Proportionally resizes the image if its longest side exceeds max_side."""
    W, H = img.size
    if max(W, H) <= max_side:
        return img
    scale = max_side / max(W, H)
    return img.resize((int(W * scale), int(H * scale)), Image.LANCZOS)


def _draw_target_box(img: Image.Image, box: list[float]) -> Image.Image:
    """
    Resizes the image to MAX_FULL_SIZE and draws the green bounding box.
    Box coordinates are in absolute pixels on the original image.
    """
    img_r = _resize_if_needed(img)
    W_o, H_o = img.size
    W_r, H_r = img_r.size
    sx, sy = W_r / W_o, H_r / H_o

    out  = img_r.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    x1 = max(0,        int(box[0] * sx))
    y1 = max(0,        int(box[1] * sy))
    x2 = min(W_r - 1,  int(box[2] * sx))
    y2 = min(H_r - 1,  int(box[3] * sy))
    if x2 > x1 and y2 > y1:
        draw.rectangle([x1, y1, x2, y2], outline=COLOR_TARGET, width=BB_WIDTH)
    return out


def _cosmos_coords(box: list[float], W: int, H: int) -> dict[str, str]:
    """Converts absolute pixel coordinates to a 0-1024 scale (Cosmos-Reason2 format)."""
    return {
        "x1": str(int((box[0] / W) * COSMOS_COORD_SCALE)),
        "y1": str(int((box[1] / H) * COSMOS_COORD_SCALE)),
        "x2": str(int((box[2] / W) * COSMOS_COORD_SCALE)),
        "y2": str(int((box[3] / H) * COSMOS_COORD_SCALE)),
    }


def _image_to_data_uri(img: Image.Image, quality: int = 90) -> str:
    """Encodes a PIL image into a base64 JPEG data URI for the OpenAI-compatible API."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _parse_response(raw: str, box: list[float]) -> str:
    """
    Parses the text response from Cosmos.
    Returns "yes" or "no". Ambiguous cases -> "no" (conservative approach).
    """
    cleaned = raw.strip().lower()

    if cleaned in ("yes", "yes.", "yes!", "sì", "si"):
        return "yes"
    if cleaned in ("no", "no.", "no!"):
        return "no"
    if cleaned.startswith("yes"):
        return "yes"
    if cleaned.startswith("no"):
        return "no"

    print(f"  [AMBIGUOUS] Uninterpretable response: '{raw}' "
          f"for box {[round(b, 1) for b in box]} → treated as 'no'")
    return "no"


class CosmosInferenceModule:
    """
    Queries an already running vLLM server (Cosmos-Reason2-8B), instead of
    loading the model locally. Same public interface as
    cosmos_inference_module.CosmosInferenceModule.

    For each bounding box, the model receives:
      - Full image (max 1280px) with green BB drawn on it
      - Box coordinates in the prompt (format [Region 0]<box>, 0-1024 scale)

    Response: "yes" (empty area / OOS) or "no" (not empty area).
    Ambiguous cases: printed and treated as "no".
    Network/timeout errors: printed and treated as "no" (no automatic retry
    — if the server is unreachable it is better to know it immediately
    than hide the problem with silent retries).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model_name: str = "cosmos-reason2-8b",
        max_tokens: int = 300,
        timeout: int = 120,
    ):
        self.base_url    = base_url.rstrip("/")
        self.model_name  = model_name
        self.max_tokens  = max_tokens
        self.timeout     = timeout

        print(f"  [vLLM] Verifying connection to server {self.base_url} ...")
        try:
            resp = requests.get(f"{self.base_url}/v1/models", timeout=10)
            resp.raise_for_status()
            models = [m["id"] for m in resp.json().get("data", [])]
            print(f"  [vLLM] Connected. Available models: {models}")
            if self.model_name not in models:
                print(f"  [vLLM] WARNING: '{self.model_name}' not among the models "
                      f"served — verify the exact name used to start vllm serve")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Cannot connect to the vLLM server at {self.base_url}. "
                f"Verify that it is active (vllm serve ...) and reachable "
                f"(same node, or correct hostname). Error: {e}"
            )

    def _forward_request(
        self,
        image: Image.Image,
        prompt_text: str,
        system_prompt: str,
    ) -> str:
        """Sends a request to the vLLM server and returns the raw text response."""
        image_data_uri = _image_to_data_uri(image)

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_data_uri}},
                        {"type": "text", "text": prompt_text},
                    ],
                },
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0,    # deterministic decoding (greedy)
            "top_p": 1.0,
        }

        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        msg = data["choices"][0]["message"]
        content   = (msg.get("content") or "").strip()
        reasoning = (msg.get("reasoning") or msg.get("reasoning_content") or "").strip()

        # With --reasoning-parser qwen3 active on the server, the final response
        # can appear in "content" (after closing <think>) or,
        # for direct responses like yes/no, directly in the
        # "reasoning" field. We check both.
        return content if content else reasoning

    def score_boxes(
        self,
        image: Image.Image,
        boxes: list[list[float]],
        prompt_name: str = "no_context",
    ) -> list[str]:
        """
        Returns the text response from Cosmos ("yes"/"no") for each BB.

          "yes" → Cosmos confirms the area is an OOS gap (empty)
          "no"  → Cosmos rejects YOLO's prediction
          Ambiguous → printed and treated as "no"

        HTTP/timeout errors: printed and treated as "no".
        """
        if not boxes:
            return []

        W, H      = image.size
        template  = PROMPT_TEMPLATES[prompt_name]
        responses = []
        err_count = 0

        for box in boxes:
            annotated = _draw_target_box(image, box)
            coords    = _cosmos_coords(box, W, H)
            prompt    = template.format(**coords)

            try:
                raw = self._forward_request(
                    image         = annotated,
                    prompt_text   = prompt,
                    system_prompt = SYSTEM_PROMPT,
                )
            except requests.exceptions.RequestException as e:
                err_count += 1
                print(f"  [HTTP ERROR] box {[round(b,1) for b in box]}: {e} → response forced 'no'")
                raw = "no"

            responses.append(_parse_response(raw, box))

        if err_count > 0:
            print(f"  [HTTP ERROR] Total network/timeout errors: {err_count}")

        return responses