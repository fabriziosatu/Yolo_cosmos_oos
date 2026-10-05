"""
vllm_inference_module_diag.py  —  v2.0-diag
-----------------------------------------------
Versione DIAGNOSTICA di vllm_inference_module.py (v2.0, dual-image,
padding 20% — la configurazione migliore trovata finora).

UNICA DIFFERENZA rispetto a vllm_inference_module.py:
  - Traccia il campo "finish_reason" di ogni risposta del server vLLM.
  - Se finish_reason == "length", significa che il modello NON ha
    completato la generazione entro max_tokens=300 — la risposta e' stata
    troncata a meta' frase. In quel caso la "no" che finisce nel conteggio
    finale potrebbe non essere una vera decisione del modello, ma un
    troncamento silenzioso.
  - CosmosInferenceModule.finish_reason_counts (dict) accumula i conteggi
    per l'intera sessione, consultabile a fine esecuzione.
  - CosmosInferenceModule.truncated_boxes (list) salva le coordinate delle
    box troncate, per ispezione mirata.

Interfaccia IDENTICA a vllm_inference_module.py per il resto (stesso
score_boxes(), stesso formato di ritorno).
"""

from __future__ import annotations
import base64
import io

import requests
from PIL import Image, ImageDraw


# ─────────────────────────────────────────────────────────────────────────────
#  SYSTEM PROMPT  (aggiornato per dual-image)
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
        "Answer NO if the crop matches any step 2 case, or if you "
        "cannot clearly confirm bare shelf surface.\n"
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

CROP_PADDING  = 0.20   # 20% di padding attorno alla box, come richiesto
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


def _parse_response(raw: str, box: list[float]) -> str:
    """
    Interpreta la risposta testuale di Cosmos.
    Ritorna "yes" o "no". Casi ambigui -> "no" (approccio conservativo).
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
          f"per box {[round(b, 1) for b in box]} → trattata come 'no'")
    return "no"


class CosmosInferenceModule:
    """
    Interroga un server vLLM (Cosmos-Reason2-8B) gia' avviato.

    Per ogni BB il modello riceve DUE immagini:
      1. Immagine intera (max 1280px) con BB verde disegnata sopra
      2. Crop ravvicinato della BB (padding 20%)
    accompagnate dalle coordinate della box nel prompt (formato
    [Region 0]<box>x1,y1,x2,y2</box>, scala 0-1024).

    Risposta: "yes" (area vuota / OOS) o "no" (area non vuota).
    Casi ambigui: stampati e trattati come "no".

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
    ):
        self.base_url    = base_url.rstrip("/")
        self.model_name  = model_name
        self.max_tokens  = max_tokens
        self.timeout     = timeout
        self._consecutive_errors = 0

        # ── Diagnostica finish_reason ────────────────────────────────────
        self.finish_reason_counts: dict[str, int] = {}
        self.truncated_boxes: list[dict] = []  # box + risposta parziale, se troncate

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
    ) -> tuple[str, str]:
        """Manda una richiesta al server vLLM con 1 o 2 immagini.
        Ritorna (risposta_grezza, finish_reason)."""
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

        choice       = data["choices"][0]
        msg          = choice["message"]
        finish_reason = choice.get("finish_reason", "unknown")
        content_out  = (msg.get("content") or "").strip()
        reasoning    = (msg.get("reasoning") or msg.get("reasoning_content") or "").strip()

        # Con --reasoning-parser qwen3 attivo sul server, la risposta finale
        # puo' comparire in "content" oppure, per risposte dirette come
        # yes/no, direttamente nel campo "reasoning". Controlliamo entrambi.
        text = content_out if content_out else reasoning
        return text, finish_reason

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
                raw, finish_reason = self._forward_request(
                    images        = [annotated, crop],
                    prompt_text   = prompt,
                    system_prompt = SYSTEM_PROMPT,
                )
                self._consecutive_errors = 0  # richiesta riuscita, azzera il contatore

                # ── Diagnostica finish_reason ─────────────────────────────
                self.finish_reason_counts[finish_reason] = (
                    self.finish_reason_counts.get(finish_reason, 0) + 1
                )
                if finish_reason == "length":
                    print(f"  [TRONCATO] box {[round(b,1) for b in box]}: "
                          f"finish_reason='length', risposta parziale='{raw[:80]}'")
                    self.truncated_boxes.append({
                        "box": [round(b, 1) for b in box],
                        "raw_partial": raw,
                    })

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

            responses.append(_parse_response(raw, box))

        return responses

    def print_diagnostic_summary(self):
        """Stampa il riepilogo di finish_reason accumulato durante la sessione."""
        total = sum(self.finish_reason_counts.values())
        print("\n" + "=" * 60)
        print("  DIAGNOSTICA finish_reason")
        print("=" * 60)
        if total == 0:
            print("  Nessuna richiesta registrata.")
            return
        for reason, count in sorted(self.finish_reason_counts.items(),
                                     key=lambda x: -x[1]):
            pct = 100 * count / total
            marker = "  ⚠️  TRONCAMENTI!" if reason == "length" and count > 0 else ""
            print(f"  {reason:15s}: {count:6d} ({pct:5.1f}%){marker}")
        print(f"  {'TOTALE':15s}: {total:6d}")
        n_trunc = self.finish_reason_counts.get("length", 0)
        if n_trunc > 0:
            print(f"\n  {n_trunc} risposte troncate su {total} "
                  f"({100*n_trunc/total:.2f}%) — vedi self.truncated_boxes "
                  f"per le coordinate esatte.")
        else:
            print("\n  Nessun troncamento — max_tokens=300 e' sempre sufficiente.")
        print("=" * 60)