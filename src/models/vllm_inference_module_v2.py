"""
vllm_inference_module_v2.py  —  v2.0
-------------------------------------
Cosmos-Reason2-8B via vLLM server (HTTP).

NEW IN v2.0 (compared to v1.0):
  - DUAL IMAGE: for each box, in addition to the full image with green BB,
    a close-up CROP (20% padding) of the same area is also sent.
    Goal: to reduce the FNR on cases where the box is small compared
    to the 1280px image and the detail is lost in the full photo — the crop
    gives the model the same level of detail that a human would have
    "getting closer" to look at that spot.
  - FAIL-FAST: if the vLLM server stops responding for N consecutive
    requests (default 5), the script stops immediately with an explicit
    error instead of continuing for tens of minutes forcing "no" on
    everything (happened on 13/07: a server crash halfway through evaluation
    produced 40 minutes of unusable results before we noticed).

IDENTICAL interface to before: same score_boxes() method, same return
format (list[str] of "yes"/"no").
"""

from __future__ import annotations
import base64
import io

import requests
from PIL import Image, ImageDraw


# ─────────────────────────────────────────────────────────────────────────────
#  SYSTEM PROMPT  (updated for dual-image)
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
    "You will see two images of the SAME area: (1) the full shelf photo "
    "with a green box marking the region to inspect, and (2) a zoomed-in "
    "crop of that exact region. Use the crop for fine detail and the full "
    "image for context (shelf depth, lighting, surrounding products) to "
    "make your decision.\n"
    "Respond with exactly one word: 'yes' (the shelf is empty / OOS) or "
    "'no' (it is not). Do not add any other words."
)

SYSTEM_PROMPT_CROP_ONLY = (
    "You are an expert visual inspector for a retail Out-Of-Stock (OOS) "
    "detection system. You analyze a cropped photo of a supermarket shelf "
    "section to determine whether it is EMPTY of products (an OOS gap).\n\n"
    "A TRUE empty shelf shows: bare horizontal shelf surface, visible "
    "shelf backing or pegboard, and no products, packages, boxes, bottles, "
    "or merchandise of any kind.\n"
    "Respond with exactly one word: 'yes' (empty / OOS) or 'no' (it is "
    "not). Do not add any other words."
)

# ─────────────────────────────────────────────────────────────────────────────
#  PROMPT TEMPLATES  (dual-image: they refer to both images)
#  {x1} {y1} {x2} {y2} are replaced with coordinates in 0-1024 scale
# ─────────────────────────────────────────────────────────────────────────────
PROMPT_TEMPLATES: dict[str, str] = {

    "no_context": (
        "The first image shows the full shelf view with a green bounding "
        "box highlighting the area of interest — this is [Region 0]"
        "<box>{x1},{y1},{x2},{y2}</box> (coordinates normalized 0-1024). "
        "The second image is a close-up crop of that same area. "
        "Is the highlighted shelf area [Region 0] empty (an OOS gap, no "
        "products)? Answer yes or no."
    ),

    "context": (
        "The first image shows the full shelf view with a green bounding "
        "box marking the area of interest — this is [Region 0]"
        "<box>{x1},{y1},{x2},{y2}</box> (coordinates normalized 0-1024). "
        "The second image is a zoomed-in crop of that exact region.\n\n"
        "Your goal is to detect Out-Of-Stock (OOS) shelf gaps — areas "
        "where products are completely missing from the shelf.\n\n"
        "STEP 1 — Look at the crop of [Region 0]. Can you see a flat, "
        "bare shelf surface with visible depth (the back wall, pegboard, "
        "or shadow of the empty space)? If yes, this supports OOS/EMPTY.\n\n"
        "STEP 2 — Check the crop does NOT match any of these common "
        "false-positive categories:\n"
        "  - Shelf STRUCTURE only: metallic uprights, edge profile, "
        "frame borders, or dividers seen edge-on (this is shelf "
        "hardware, not empty space)\n"
        "  - REFRIGERATOR/FREEZER glass door or frozen food cabinet "
        "surface, possibly with reflections\n"
        "  - MOTION BLUR: the region is blurry/smeared from camera "
        "movement, details cannot be confirmed\n"
        "  - DISTANT/DARK shelf that LOOKS empty due to poor lighting, "
        "but the full image shows products are likely still there "
        "(check the surrounding rows)\n"
        "  - PRODUCTS present but small, dark-colored, or partially "
        "out of frame\n\n"
        "Answer YES only if step 1 is satisfied AND none of the step 2 "
        "false-positive cases apply.\n"
        "Answer NO if the crop matches any step 2 case, or if you "
        "cannot clearly confirm bare shelf surface.\n"
        "Answer with exactly one word: yes or no."
    ),

    # ── Fallback crop-only (in case a crop error needs to be handled in the future) ───
    "no_context_crop": (
        "This is a close-up crop of a supermarket shelf area. "
        "Is this shelf area empty (an OOS gap, no products)? "
        "Answer yes or no."
    ),

    "context_crop": (
        "This is a close-up crop of a supermarket shelf area.\n\n"
        "Your goal is to detect Out-Of-Stock (OOS) shelf gaps — areas "
        "where products are completely missing from the shelf.\n\n"
        "STEP 1 — Can you see a flat, bare shelf surface with visible "
        "depth (back wall, pegboard, or shadow)? If yes, this supports "
        "OOS/EMPTY.\n\n"
        "STEP 2 — Check the crop does NOT match any of these common "
        "false-positive categories:\n"
        "  - Shelf STRUCTURE only: metallic uprights, edge profile, "
        "frame borders, or dividers\n"
        "  - REFRIGERATOR/FREEZER glass door or frozen food cabinet "
        "surface\n"
        "  - MOTION BLUR: the crop is blurry/smeared\n"
        "  - DARK shelf that looks empty due to poor lighting only\n"
        "  - PRODUCTS present but small, dark-colored, or partially "
        "visible\n\n"
        "Answer YES only if step 1 is satisfied AND none of the step 2 "
        "false-positive cases apply.\n"
        "Answer NO otherwise.\n"
        "Answer with exactly one word: yes or no."
    ),
}
# ─────────────────────────────────────────────────────────────────────────────

COLOR_TARGET  = (0, 255, 0)
BB_WIDTH      = 3
MAX_FULL_SIZE = 1280
COSMOS_COORD_SCALE = 1024

#CROP_PADDING = 0.20   # 20% padding around the box, as requested
CROP_PADDING = 0.40
CROP_MIN_SIZE = 28     # minimum dimension per side (model patch size)

# Fail-fast: after how many CONSECUTIVE failed HTTP requests to stop
MAX_CONSECUTIVE_ERRORS = 5


class VLLMServerDownError(RuntimeError):
    """Raised when the vLLM server does not respond for too many consecutive requests."""
    pass


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


def _extract_crop(img: Image.Image, box: list[float]) -> Image.Image:
    """Extracts a crop of the bounding box with 20% padding, minimum CROP_MIN_SIZE px per side."""
    W, H = img.size
    bw   = box[2] - box[0]
    bh   = box[3] - box[1]
    cx1  = max(0,     int(box[0] - bw * CROP_PADDING))
    cy1  = max(0,     int(box[1] - bh * CROP_PADDING))
    cx2  = min(W - 1, int(box[2] + bw * CROP_PADDING))
    cy2  = min(H - 1, int(box[3] + bh * CROP_PADDING))
    crop = img.crop((cx1, cy1, cx2, cy2)).convert("RGB")
    cw, ch = crop.size
    if cw < CROP_MIN_SIZE or ch < CROP_MIN_SIZE:
        scale = max(CROP_MIN_SIZE / max(cw, 1), CROP_MIN_SIZE / max(ch, 1))
        crop  = crop.resize(
            (max(int(cw * scale), CROP_MIN_SIZE),
             max(int(ch * scale), CROP_MIN_SIZE)),
            Image.LANCZOS,
        )
    return crop


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
    Queries an already running vLLM server (Cosmos-Reason2-8B).

    For each bounding box, the model receives TWO images:
      1. Full image (max 1280px) with green BB drawn on it
      2. Close-up crop of the BB (20% padding)
    accompanied by the box coordinates in the prompt (format
    [Region 0]<box>x1,y1,x2,y2</box>, 0-1024 scale).

    Response: "yes" (empty area / OOS) or "no" (not empty area).
    Ambiguous cases: printed and treated as "no".

    FAIL-FAST: if the server does not respond for MAX_CONSECUTIVE_ERRORS
    consecutive requests, a VLLMServerDownError is raised and execution
    stops — it does not keep forcing "no" endlessly.
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
        self._consecutive_errors = 0

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
        images: list[Image.Image],
        prompt_text: str,
        system_prompt: str,
    ) -> str:
        """Sends a request to the vLLM server with 1 or 2 images, returns the raw text response."""
        content = [
            {"type": "image_url", "image_url": {"url": _image_to_data_uri(img)}}
            for img in images
        ]
        content.append({"type": "text", "text": prompt_text})

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
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
        content_out = (msg.get("content") or "").strip()
        reasoning   = (msg.get("reasoning") or msg.get("reasoning_content") or "").strip()

        # With --reasoning-parser qwen3 active on the server, the final response
        # can appear in "content" or, for direct responses like
        # yes/no, directly in the "reasoning" field. We check both.
        return content_out if content_out else reasoning

    def score_boxes(
        self,
        image: Image.Image,
        boxes: list[list[float]],
        prompt_name: str = "no_context",
    ) -> list[str]:
        """
        Returns the text response from Cosmos ("yes"/"no") for each BB,
        using TWO images (annotated full + crop) per box.

        Raises VLLMServerDownError if the server does not respond for
        MAX_CONSECUTIVE_ERRORS consecutive requests — stopping
        the entire evaluation immediately instead of forcing "no".
        """
        if not boxes:
            return []

        W, H      = image.size
        template  = PROMPT_TEMPLATES[prompt_name]
        responses = []

        for box in boxes:
            annotated = _draw_target_box(image, box)
            crop      = _extract_crop(image, box)
            coords    = _cosmos_coords(box, W, H)
            prompt    = template.format(**coords)

            try:
                raw = self._forward_request(
                    images        = [annotated, crop],
                    prompt_text   = prompt,
                    system_prompt = SYSTEM_PROMPT,
                )
                self._consecutive_errors = 0  # request succeeded, reset counter

            except requests.exceptions.RequestException as e:
                self._consecutive_errors += 1
                print(f"  [HTTP ERROR {self._consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}] "
                      f"box {[round(b,1) for b in box]}: {e}")

                if self._consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    raise VLLMServerDownError(
                        f"The vLLM server ({self.base_url}) has not responded for "
                        f"{MAX_CONSECUTIVE_ERRORS} consecutive requests — "
                        f"it has probably crashed. Stopping execution "
                        f"instead of continuing to produce unusable results. "
                        f"Verify the server (curl {self.base_url}/v1/models) and "
                        f"re-run the evaluation."
                    ) from e

                raw = "no"  # isolated error, not yet at the limit: force 'no' and continue

            responses.append(_parse_response(raw, box))

        return responses