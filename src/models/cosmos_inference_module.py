"""
cosmos_inference_module.py  —  v2.2
-------------------------------------
Cosmos-Reason2-8B frozen, solo inferenza.

INPUT PER OGNI BOX: UNA sola immagine — quella intera (max 1280px) con la
box verde disegnata sopra (predizione YOLO) — accompagnata dalle coordinate
della stessa box nel prompt testuale, in formato nativo Cosmos-Reason2/
Qwen3-VL. NESSUN crop (rimosso su richiesta esplicita rispetto alla v2.1
dual-image).

Architettura:
  - STRUTTURA da scripts/inference_sample.py ufficiale (repo NVIDIA):
    un'unica chiamata apply_chat_template(tokenize=True, return_dict=True)
    con l'immagine incorporata direttamente nel messaggio, piu' i limiti
    sui token visivi (min/max_vision_tokens) per contenere il consumo VRAM.
  - PROMPT: checklist STEP1/STEP2 (adattato per singola immagine) +
    terminologia esplicita "Out-Of-Stock (OOS)" e lista di falsi positivi.
  - COORDINATE: riferimento testuale [Region 0]<box>x1,y1,x2,y2</box> in
    scala 0-1024 — formato di grounding NATIVO di Cosmos-Reason2/Qwen3-VL
    (diverso da Qwen2.5-VL, che usa <box>(x1,y1),(x2,y2)</box> in scala
    0-1000). Cosmos-Reason2 e' post-trained su Qwen3-VL-8B-Instruct e ne
    eredita questa convenzione.

Risposta: testo "yes"/"no" via generate() (max_new_tokens=5), compatibile
con l'interfaccia list[str] attesa da evaluate_yolo_cosmos.py.

OOM: senza crop di fallback, in caso di OOM la risposta viene forzata a
"no" (nessuna seconda immagine più leggera da tentare).

Casi ambigui: trattati come "no" (approccio conservativo).
"""

from __future__ import annotations
import torch
from PIL import Image, ImageDraw
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


# ─────────────────────────────────────────────────────────────────────────────
#  SYSTEM PROMPT
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
#  PROMPT TEMPLATES
#  {x1} {y1} {x2} {y2} vengono sostituiti con le coordinate in scala 0-1024
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

# Scala di normalizzazione per il grounding di Cosmos-Reason2/Qwen3-VL.
# IMPORTANTE: diversa da Qwen2.5-VL (0-1000) — Cosmos-Reason2 usa 0-1024,
# formato <box>x1,y1,x2,y2</box> (senza parentesi interne), spesso preceduto
# da un tag [Region N] quando si fa riferimento alla box nel testo.
COSMOS_COORD_SCALE = 1024


def _cosmos_coords(box: list[float], W: int, H: int) -> dict[str, str]:
    """
    Converte coordinate pixel assolute (sull'immagine originale, non
    ridimensionata) nella scala 0-1024 nativa di Cosmos-Reason2/Qwen3-VL.
    """
    return {
        "x1": str(int((box[0] / W) * COSMOS_COORD_SCALE)),
        "y1": str(int((box[1] / H) * COSMOS_COORD_SCALE)),
        "x2": str(int((box[2] / W) * COSMOS_COORD_SCALE)),
        "y2": str(int((box[3] / H) * COSMOS_COORD_SCALE)),
    }


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
    Carica Cosmos-Reason2-8B frozen e lo usa per classificare BB di scaffali
    (rilevamento gap Out-Of-Stock).

    Per ogni BB il modello riceve UNA sola immagine:
      - Immagine intera (max 1280px) con BB verde disegnata sopra
    accompagnata dalle coordinate della stessa box nel prompt (formato
    [Region 0]<box>x1,y1,x2,y2</box>, scala 0-1024).

    Risposta: "yes" (area vuota / OOS) o "no" (area non vuota).
    Casi ambigui: stampati e trattati come "no".
    OOM: nessun fallback su crop — risposta forzata a "no".
    """

    PIXELS_PER_TOKEN = 32 ** 2

    def __init__(
        self,
        model_path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str | None = None,
        min_vision_tokens: int = 256,
        max_vision_tokens: int = 2048,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype  = torch_dtype

        print(f"  [Cosmos] Carico processor da {model_path}")
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )

        # Limita i token visivi (come da inference_sample.py ufficiale),
        # per contenere il consumo VRAM sull'immagine full-size.
        vision_size = {
            "shortest_edge": min_vision_tokens * self.PIXELS_PER_TOKEN,
            "longest_edge":  max_vision_tokens * self.PIXELS_PER_TOKEN,
        }
        self.processor.image_processor.size = vision_size

        print(f"  [Cosmos] Carico modello ({torch_dtype}) ...")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch_dtype,
            device_map="auto",
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        self.model.eval()
        print(f"  [Cosmos] Modello pronto su {self.device}")

    def _forward_text(
        self,
        image: Image.Image,
        prompt_text: str,
        system_prompt: str,
    ) -> str:
        """
        Esegue un forward pass con generazione testuale su una singola
        immagine. Ritorna la risposta grezza del modello (stringa).
        """
        # IMPORTANT: il media va PRIMA del testo nel content, per
        # rispecchiare il formato di training (come indicato in
        # inference_sample.py ufficiale).
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": prompt_text},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[0, input_len:]
        return self.processor.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def score_boxes(
        self,
        image: Image.Image,
        boxes: list[list[float]],
        prompt_name: str = "no_context",
    ) -> list[str]:
        """
        Ritorna la risposta testuale di Cosmos ("yes"/"no") per ogni BB.

          "yes" → Cosmos conferma che l'area è un gap OOS (vuota)
          "no"  → Cosmos rigetta la predizione di YOLO
          Ambigui → stampati e trattati come "no"

        Ogni box viene valutata con UNA sola immagine (full + BB verde) e
        le coordinate della box nel prompt. In caso di OOM, risposta
        forzata a "no" (nessun crop di fallback disponibile).
        """
        if not boxes:
            return []

        W, H      = image.size
        template  = PROMPT_TEMPLATES[prompt_name]
        responses = []
        oom_count = 0

        for box in boxes:
            annotated = _draw_target_box(image, box)

            # Coordinate nel formato nativo Cosmos-Reason2 (scala 0-1024),
            # calcolate sulle dimensioni ORIGINALI dell'immagine (il
            # rapporto normalizzato non cambia con il resize per il disegno)
            coords = _cosmos_coords(box, W, H)
            prompt = template.format(**coords)

            try:
                raw = self._forward_text(
                    image         = annotated,
                    prompt_text   = prompt,
                    system_prompt = SYSTEM_PROMPT,
                )
            except torch.cuda.OutOfMemoryError:
                oom_count += 1
                torch.cuda.empty_cache()
                print(f"  [OOM] Nessun fallback su crop per box "
                      f"{[round(b,1) for b in box]} → risposta forzata 'no' "
                      f"(OOM totali: {oom_count})")
                raw = "no"

            responses.append(_parse_response(raw, box))
            torch.cuda.empty_cache()

        if oom_count > 0:
            print(f"  [OOM] Totale episodi OOM: {oom_count}")

        return responses