"""
vllm_inference_module_v2_uncertain_yes.py  —  v2.1
------------------------------------------------------
Cosmos-Reason2-8B via server vLLM (HTTP), dual-image, padding 20%.

NOVITA' (rispetto a vllm_inference_module_v2_ambiguous_yes.py):
  - Il SYSTEM_PROMPT ora istruisce esplicitamente il modello a rispondere
    'yes' quando e' incerto nel merito ("Be decisive: if the area looks
    empty or you are uncertain, answer YES") — non solo un fallback lato
    codice per risposte non interpretabili (quello resta comunque attivo
    come rete di sicurezza tramite ambiguous_default="yes").
  - Stessa modifica applicata al template "context" (checklist STEP1/STEP2).

Le due modifiche (prompt + ambiguous_default) sono complementari:
  - Il prompt cambia COME il modello decide internamente nei casi dubbi
  - ambiguous_default resta la rete di sicurezza per i (rari) casi in cui
    il testo grezzo non e' comunque interpretabile come yes/no
"""

from __future__ import annotations
import base64
import io

import requests
from PIL import Image, ImageDraw


# ─────────────────────────────────────────────────────────────────────────────
#  SYSTEM PROMPT  (aggiornato: decisivo verso YES in caso di dubbio)
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
    "make your decision.\n\n"
    "Be decisive: if the area looks empty, or if you are genuinely "
    "uncertain, answer YES. Only answer NO if you can clearly and "
    "confidently see a real product or a specific reason the area is not "
    "empty.\n"
    "Respond with exactly one word: 'yes' (the shelf is empty / OOS) or "
    "'no' (it is not)."
)

SYSTEM_PROMPT_CROP_ONLY = (
    "You are an expert visual inspector for a retail Out-Of-Stock (OOS) "
    "detection system. You analyze a cropped photo of a supermarket shelf "
    "section to determine whether it is EMPTY of products (an OOS gap).\n\n"
    "A TRUE empty shelf shows: bare horizontal shelf surface, visible "
    "shelf backing or pegboard, and no products, packages, boxes, bottles, "
    "or merchandise of any kind.\n"
    "Respond with exactly one word: 'yes' (empty / OOS) or 'no' (it is "
    "not)."
)

# ─────────────────────────────────────────────────────────────────────────────
#  PROMPT TEMPLATES  (dual-image: fanno riferimento a entrambe le immagini)
#  {x1} {y1} {x2} {y2} vengono sostituiti con le coordinate in scala 0-1024
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
        "Answer NO only if the crop clearly matches a step 2 case, or if "
        "you can clearly and confidently see it is NOT empty. If you are "
        "genuinely uncertain, answer YES.\n"
        "Answer with exactly one word: yes or no."
    ),

    # ── Fallback crop-only (se in futuro serve gestire un errore sul crop) ───
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

CROP_PADDING = 0.20   # 20% di padding attorno alla box, come richiesto
#CROP_PADDING = 0.40
CROP_MIN_SIZE = 28     # dimensione minima per lato (patch size del modello)

# Fail-fast: dopo quante richieste HTTP fallite CONSECUTIVE ci si ferma
MAX_CONSECUTIVE_ERRORS = 5


class VLLMServerDownError(RuntimeError):
    """Sollevata quando il server vLLM non risponde per troppe richieste di fila."""
    pass


def _resize_if_needed(img: Image.Image, max_side: int = MAX_FULL_SIZE) -> Image.Image:
    """Ridimensiona proporzionalmente se il lato più lungo supera max_side."""
    W, H = img.size
    if max(W, H) <= max_side:
        return img
    scale = max_side / max(W, H)
    return img.resize((int(W * scale), int(H * scale)), Image.LANCZOS)


def _draw_target_box(img: Image.Image, box: list[float]) -> Image.Image:
    """
    Ridimensiona l'immagine a MAX_FULL_SIZE e disegna la BB verde.
    Le coordinate box sono in pixel assoluti sull'immagine originale.
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
    """Estrae il crop della BB con padding 20%, minimo CROP_MIN_SIZE px per lato."""
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
    """Converte coordinate pixel assolute in scala 0-1024 (formato Cosmos-Reason2)."""
    return {
        "x1": str(int((box[0] / W) * COSMOS_COORD_SCALE)),
        "y1": str(int((box[1] / H) * COSMOS_COORD_SCALE)),
        "x2": str(int((box[2] / W) * COSMOS_COORD_SCALE)),
        "y2": str(int((box[3] / H) * COSMOS_COORD_SCALE)),
    }


def _image_to_data_uri(img: Image.Image, quality: int = 90) -> str:
    """Codifica un'immagine PIL in data URI base64 JPEG per l'API OpenAI-compatible."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _parse_response(raw: str, box: list[float], ambiguous_default: str = "no") -> str:
    """
    Interpreta la risposta testuale di Cosmos.
    Ritorna "yes" o "no". Casi ambigui -> ambiguous_default (default "no",
    approccio conservativo — configurabile per testare l'altra direzione).
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

    print(f"  [AMBIGUOUS] Risposta non interpretabile: '{raw}' "
          f"per box {[round(b, 1) for b in box]} → trattata come '{ambiguous_default}'")
    return ambiguous_default


class CosmosInferenceModule:
    """
    Interroga un server vLLM (Cosmos-Reason2-8B) gia' avviato.

    Per ogni BB il modello riceve DUE immagini:
      1. Immagine intera (max 1280px) con BB verde disegnata sopra
      2. Crop ravvicinato della BB (padding 20%)
    accompagnate dalle coordinate della box nel prompt (formato
    [Region 0]<box>x1,y1,x2,y2</box>, scala 0-1024).

    Risposta: "yes" (area vuota / OOS) o "no" (area non vuota).
    Casi ambigui: stampati e trattati secondo ambiguous_default ("no" di
    default, conservativo — passare "yes" per il comportamento permissivo).

    FAIL-FAST: se il server non risponde per MAX_CONSECUTIVE_ERRORS
    richieste di fila, viene sollevata VLLMServerDownError e l'esecuzione
    si interrompe — non continua a forzare "no" all'infinito.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model_name: str = "cosmos-reason2-8b",
        max_tokens: int = 300,
        timeout: int = 120,
        ambiguous_default: str = "yes",  # coerente con il prompt "decisive->yes"
    ):
        self.base_url    = base_url.rstrip("/")
        self.model_name  = model_name
        self.max_tokens  = max_tokens
        self.timeout     = timeout
        self.ambiguous_default = ambiguous_default
        self._consecutive_errors = 0

        print(f"  [vLLM] Verifico connessione al server {self.base_url} ...")
        try:
            resp = requests.get(f"{self.base_url}/v1/models", timeout=10)
            resp.raise_for_status()
            models = [m["id"] for m in resp.json().get("data", [])]
            print(f"  [vLLM] Connesso. Modelli disponibili: {models}")
            if self.model_name not in models:
                print(f"  [vLLM] ATTENZIONE: '{self.model_name}' non tra i modelli "
                      f"serviti — verifica il nome esatto con cui hai avviato vllm serve")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Impossibile connettersi al server vLLM su {self.base_url}. "
                f"Verifica che sia attivo (vllm serve ...) e raggiungibile "
                f"(stesso nodo, o hostname corretto). Errore: {e}"
            )

    def _forward_request(
        self,
        images: list[Image.Image],
        prompt_text: str,
        system_prompt: str,
    ) -> str:
        """Manda una richiesta al server vLLM con 1 o 2 immagini, ritorna la risposta grezza."""
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
            "temperature": 0,    # decodifica deterministica (greedy)
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

        # Con --reasoning-parser qwen3 attivo sul server, la risposta finale
        # puo' comparire in "content" oppure, per risposte dirette come
        # yes/no, direttamente nel campo "reasoning". Controlliamo entrambi.
        return content_out if content_out else reasoning

    def score_boxes(
        self,
        image: Image.Image,
        boxes: list[list[float]],
        prompt_name: str = "no_context",
    ) -> list[str]:
        """
        Ritorna la risposta testuale di Cosmos ("yes"/"no") per ogni BB,
        usando DUE immagini (full annotata + crop) per box.

        Solleva VLLMServerDownError se il server non risponde per
        MAX_CONSECUTIVE_ERRORS richieste consecutive — interrompe subito
        l'intera valutazione invece di continuare a forzare "no".
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
                self._consecutive_errors = 0  # richiesta riuscita, azzera il contatore

            except requests.exceptions.RequestException as e:
                self._consecutive_errors += 1
                print(f"  [ERRORE HTTP {self._consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}] "
                      f"box {[round(b,1) for b in box]}: {e}")

                if self._consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    raise VLLMServerDownError(
                        f"Il server vLLM ({self.base_url}) non risponde da "
                        f"{MAX_CONSECUTIVE_ERRORS} richieste consecutive — "
                        f"probabilmente e' crashato. Interrompo l'esecuzione "
                        f"invece di continuare a produrre risultati inutilizzabili. "
                        f"Verifica il server (curl {self.base_url}/v1/models) e "
                        f"rilancia la valutazione."
                    ) from e

                raw = "no"  # errore isolato, non ancora al limite: forza 'no' e continua

            responses.append(_parse_response(raw, box, self.ambiguous_default))

        return responses