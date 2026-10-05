"""
visualize_yolo_cosmos_v2_FN.py  —  v2.1
------------------------------------
Summary plot and table generator dedicated to the new hybrid YOLO + Cosmos pipelines 
(v4.1/v5.1 versions based on direct text responses, without numerical thresholds).

NEW IN v2.0 (compared to v1.0):
  - Compatibility with text logs: the code has been updated to read the new structured JSON.
  - Cosmos vs YOLO comparison table: introduces a new table that displays exactly how the two 
    models collaborated. It highlights how many real gaps were confirmed by both and, 
    most importantly, how many YOLO errors were successfully corrected by the Cosmos model (NO_FP).

NEW IN v2.1:
  - Hardcoded defaults for the Cosmos project (no mandatory command-line
    arguments needed for daily use: simply run the script as is
    to regenerate the visualization of the last run).
"""

import argparse, json, os
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")

# ══════════════════════════════════════════════════════════════
#  CONFIG — default paths/parameters (only modify these)
# ══════════════════════════════════════════════════════════════
_PROJECT_ROOT = os.path.expanduser("~/Progetto_tesi_Cosmos")

JSON_PATH_DEFAULT = f"{_PROJECT_ROOT}/results/baseline_cosmos_vllm_v1/metrics_v4_1.json"
OUT_DIR_DEFAULT   = f"{_PROJECT_ROOT}/results/baseline_cosmos_vllm_v1/vis"
MODE_DEFAULT      = "cosmos"
PROMPT_DEFAULT    = "context"
CONF_DEFAULT      = [0.25, 0.5]
IOU_DEFAULT       = [0.0, 0.25, 0.5, 0.75]
BAR_CONF_DEFAULT  = 0.25
# ══════════════════════════════════════════════════════════════
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# ─────────────────────────────────────────────────────────────────────────────
#  FONT
# ─────────────────────────────────────────────────────────────────────────────
FONT_BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
FONT_REG  = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

def _font(path, size):
    """
    Attempts to load a TrueType font from the given path.
    Falls back to the default PIL font if the specified font is not found.
    """
    try:    return ImageFont.truetype(path, size)
    except: return ImageFont.load_default()

def _fmt(v, d=3):
    """
    Formats a numeric value to a specific number of decimal places as a string.
    Returns an em dash ('—') if the value is None.
    """
    return f"{v:.{d}f}" if v is not None else "—"

def _text_w(text, font):
    """
    Calculates the exact pixel width of a given text string when rendered 
    with the specified font. Uses a heuristic fallback if getbbox fails.
    """
    try:
        bb = font.getbbox(str(text))
        return bb[2] - bb[0]
    except:
        return len(str(text)) * 7


# ─────────────────────────────────────────────────────────────────────────────
#  THEME COLORS (inspired by attached tables)
# ─────────────────────────────────────────────────────────────────────────────
C_BG       = (255, 252, 248)
C_HDR_GRP  = (180, 140, 100)
C_HDR_SUB  = (210, 180, 150)
C_GRP_BG   = (235, 220, 200)
C_ROW1     = (255, 252, 248)
C_ROW2     = (248, 242, 235)
C_SECTION  = (220, 205, 185)
C_TEXT     = (40,  40,  40)
C_BEST     = (0,  140,   0)
C_LINE     = (200, 185, 170)
C_SEP      = (150, 110,  70)
C_WHITE    = (255, 255, 255)
C_TITLE    = (80,  50,  20)


# ─────────────────────────────────────────────────────────────────────────────
#  IoU UTILITY
# ─────────────────────────────────────────────────────────────────────────────
def _iou(a, b):
    """
    Computes the Intersection over Union (IoU) between two bounding boxes.
    Boxes must be in [x1, y1, x2, y2] format.
    """
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0: return 0.0
    aa = (a[2]-a[0])*(a[3]-a[1]); ab = (b[2]-b[0])*(b[3]-b[1])
    return inter / (aa + ab - inter + 1e-6)


# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_pairs(img_dir: Path, lbl_dir: Path):
    """
    Loads image and label pairs from the given directories.
    Converts YOLO format (center x, center y, width, height) normalized 
    coordinates to absolute pixel coordinates [x1, y1, x2, y2].
    Returns a list of dictionaries containing the image path and Ground Truth boxes.
    """
    pairs = []
    for img_path in sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")):
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists(): continue
        from PIL import Image as _I
        img = _I.open(img_path)
        W, H = img.size
        gt = []
        for line in lbl_path.read_text().strip().splitlines():
            p = line.split()
            if len(p) < 5: continue
            _, cx, cy, bw, bh = map(float, p[:5])
            gt.append([(cx-bw/2)*W, (cy-bh/2)*H, (cx+bw/2)*W, (cy+bh/2)*H])
        pairs.append({"path": img_path, "gt": gt})
    return pairs


def _multi_match(preds, gts, iou_thr):
    """
    Multi-matching identical to evaluate_yolo_cosmos.py (professor's directives).
    A prediction is a True Positive (TP) if it matches at least one GT with sufficient IoU.
    A GT is a False Negative (FN) if it is not matched by any prediction.
    Returns a tuple containing a boolean list (is_tp_pred) and a set of matched GT indices.
    """
    n_preds = len(preds)
    n_gts   = len(gts)
    mat_pred = [[] for _ in range(n_preds)]
    mat_gt   = [[] for _ in range(n_gts)]

    for pi, pb in enumerate(preds):
        for gi, gb in enumerate(gts):
            v = _iou(pb, gb)
            if (iou_thr > 0 and v >= iou_thr) or (iou_thr == 0 and v > 0):
                mat_pred[pi].append(gi)
                mat_gt[gi].append(pi)

    is_tp_pred = [len(lst) > 0 for lst in mat_pred]
    matched    = {gi for gi, lst in enumerate(mat_gt) if len(lst) > 0}
    return is_tp_pred, matched


def _classify_fp(box):
    """
    Classifies a False Positive (FP) bounding box into a specific error category 
    based on its dimensions (width, height) and area.
    """
    x1,y1,x2,y2 = box; bw=x2-x1; bh=y2-y1; area=bw*bh
    if bw<40 or bh<20 or area<800:   return "dark_region"
    if area<1600:                     return "occupied_shelf"
    return "other_fp"

def _classify_fn(box):
    """
    Classifies a False Negative (FN) ground truth box into a specific error category 
    based on its dimensions (width, height) and area.
    """
    x1,y1,x2,y2 = box; bw=x2-x1; bh=y2-y1; area=bw*bh
    if bw<32 or bh<32 or area<1024:  return "small_box"
    return "other_fn"


def evaluate_yolo_run(yolo, pairs, conf_vals, iou_vals):
    """
    Evaluates a YOLO detector using the same formulas as evaluate_yolo_cosmos.py.
    Runs inference on the dataset for given confidence thresholds and calculates 
    TP, FP, FN, TPR, FPR, FNR and categorizes errors for each IoU threshold.
    """
    from ultralytics import YOLO as _Y
    results = {}
    for conf in conf_vals:
        samples = []
        for s in pairs:
            res   = yolo(s["path"], conf=conf, verbose=False)
            boxes = res[0].boxes.xyxy.cpu().numpy().tolist() \
                    if res and res[0].boxes is not None else []
            samples.append({"gt": s["gt"], "preds": boxes})

        for iou_t in iou_vals:
            total_gt=0; total_tp=0; total_fp=0; total_fn=0
            fp_cats=defaultdict(int); fn_cats=defaultdict(int)
            for s in samples:
                gt=s["gt"]; preds=s["preds"]
                is_tp, matched = _multi_match(preds, gt, iou_t)
                total_gt += len(gt)
                for pb, tp in zip(preds, is_tp):
                    if tp: total_tp+=1
                    else:  total_fp+=1; fp_cats[_classify_fp(pb)]+=1
                for j,gb in enumerate(gt):
                    if j not in matched:
                        total_fn+=1; fn_cats[_classify_fn(gb)]+=1

            results[(conf,iou_t)] = {
                "GT":total_gt,"TP":total_tp,"FP":total_fp,"FN":total_fn,
                "TPR": total_tp/total_gt if total_gt>0 else 0.0,
                "FPR": total_fp/total_gt if total_gt>0 else 0.0,
                "FNR": total_fn/total_gt if total_gt>0 else 0.0,
                "fp_cats": dict(fp_cats),
                "fn_cats": dict(fn_cats),
            }

    # Ultralytics for Table 1
    try:
        uv = yolo.val(verbose=False)
        rd = uv.results_dict
        p  = float(rd.get("metrics/precision(B)",0))
        r  = float(rd.get("metrics/recall(B)",   0))
        results["ultralytics"] = {
            "precision": p, "recall": r,
            "f1":        2*p*r/(p+r) if (p+r)>0 else 0.0,
            "map50":     float(rd.get("metrics/mAP50(B)",    0)),
            "map50_95":  float(rd.get("metrics/mAP50-95(B)", 0)),
        }
    except Exception as e:
        print(f"  [WARN] Ultralytics val: {e}")
        results["ultralytics"] = {}

    return results


def load_cosmos_results(json_path, conf_vals, iou_vals, cosmos_thrs, prompt):
    """
    Loads results from a v2 JSON (evaluate_yolo_cosmos.py).
    v2 keys format: conf={conf}_iou={iou}_{detector}_{prompt} (without cosmos_thr).
    'cosmos_thrs' is ignored (kept for backward compatibility) — returns {None: all_results}.
    Extracts metrics regarding Ground Truths, predictions, and Cosmos vs YOLO performance.
    """
    with open(json_path) as f:
        data = json.load(f)
    run_names = sorted({v["detector"] for v in data.values()})
    all_results = {}
    for det in run_names:
        res = {}
        for conf in conf_vals:
            for iou_t in iou_vals:
                k = f"conf={conf}_iou={iou_t}_{det}_{prompt}"
                if k not in data:
                    continue
                m = data[k]
                res[(conf, iou_t)] = {
                    "GT":  m["GT"],  "TP": m["TP"],
                    "FP":  m["FP"],  "FN": m["FN"],
                    "TPR": m["TPR"],
                    "FPR": m["FP"] / m["GT"] if m["GT"] > 0 else 0.0,
                    "FNR": m["FNR"],
                    "fp_cats": m.get("fp_cats", {}),
                    "fn_cats": m.get("fn_cats", {}),
                    # Cosmos vs YOLO metrics (new in v2)
                    "yolo_total":       m.get("yolo_total", 0),
                    "cosmos_yes_total":   m.get("cosmos_yes_total", 0),
                    "cosmos_no_total":    m.get("cosmos_no_total", 0),
                    "cosmos_yes_were_tp": m.get("cosmos_yes_were_tp", 0),
                    "cosmos_yes_were_fp": m.get("cosmos_yes_were_fp", 0),
                    "cosmos_no_were_tp":  m.get("cosmos_no_were_tp", 0),
                    "cosmos_no_were_fp":  m.get("cosmos_no_were_fp", 0),
                    # FN cause breakdown, per GT (v4)
                    "fn_missed_by_yolo":     m.get("fn_missed_by_yolo", 0),
                    "fn_rejected_by_cosmos": m.get("fn_rejected_by_cosmos", 0),
                }
        res["ultralytics"] = {}
        all_results[det] = res
    return run_names, {None: all_results}


# ─────────────────────────────────────────────────────────────────────────────
#  1. BAR CHART
# ─────────────────────────────────────────────────────────────────────────────

def plot_bars(all_results, run_names, conf_val, iou_vals, title, out_path):
    """
    Generates a Matplotlib bar chart comparing TPR, FPR, and FNR across different 
    detectors (run_names) for each specified IoU threshold.
    Saves the output figure to 'out_path'.
    """
    fig, axes = plt.subplots(1, len(iou_vals), figsize=(6*len(iou_vals), 6),
                             sharey=False)
    if len(iou_vals) == 1: axes = [axes]

    bar_colors = {"TPR": "#2ecc71", "FPR": "#e74c3c", "FNR": "#f39c12"}
    w = 0.22

    fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, iou_t in zip(axes, iou_vals):
        x = np.arange(len(run_names))
        for mi, (metric, mc) in enumerate(bar_colors.items()):
            vals = [all_results[rn].get((conf_val, iou_t), {}).get(metric, 0)
                    for rn in run_names]
            offset = (mi - 1) * w
            rects  = ax.bar(x + offset, vals, w, color=mc, alpha=0.88,
                            edgecolor="white", linewidth=0.5, label=metric)
            for r, v in zip(rects, vals):
                if v > 0.02:
                    ax.text(r.get_x()+r.get_width()/2, r.get_height()+0.01,
                            f"{v:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_title(f"IoU={iou_t}", fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace("_"," ").title() for n in run_names], fontsize=9)
        ax.set_ylabel("Metric value"); ax.set_ylim(0, 1.18)
        ax.grid(axis="y", alpha=0.3)
        if iou_t == iou_vals[0]:
            ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Bar chart → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  2. PNG TABLE — engine
# ─────────────────────────────────────────────────────────────────────────────

def _make_png_table(title_text, cols, col_widths, all_rows, get_val_fn,
                    best_cols, lower_is_better_cols, int_cols,
                    footer_text, out_path, conf_thresholds):
    """
    Generates a PNG table using PIL. Features a two-level header for grouped columns
    (columns with '/c' in their name are grouped together by confidence threshold).
    Highlights the 'best' values dynamically based on provided criteria.
    """
    ROW_H  = 30; GRP_H = 26
    HDR_R1 = 20; HDR_R2 = 20; HDR_H = HDR_R1 + HDR_R2
    PAD    = 14; TITLE_H = 26

    n_data = sum(1 for r in all_rows if r[0] != "__group__")
    n_grp  = sum(1 for r in all_rows if r[0] == "__group__")
    total_h = TITLE_H + HDR_H + n_data*ROW_H + n_grp*GRP_H + 44
    total_w = sum(col_widths) + PAD*2 + 4

    F_T   = _font(FONT_BOLD, 13); F_H1 = _font(FONT_BOLD, 11)
    F_H2  = _font(FONT_REG,  10); F_C  = _font(FONT_REG,  11)
    F_M   = _font(FONT_BOLD, 11)

    img  = Image.new("RGB", (total_w, total_h), C_BG)
    draw = ImageDraw.Draw(img)

    def x_col(ci): return PAD + sum(col_widths[:ci])
    def is_grp(col): return "/c" in col

    def draw_centered(txt, x, y, w, h, font, color):
        tw = _text_w(txt, font)
        draw.text((x+(w-tw)//2, y+(h-13)//2), str(txt), fill=color, font=font)

    def hline(y, width=1, color=None):
        draw.line([(PAD,y),(total_w-PAD,y)], fill=color or C_LINE, width=width)

    def vline(x, y1, y2, width=1, color=None):
        draw.line([(x,y1),(x,y2)], fill=color or C_SEP, width=width)

    # Title
    draw.text((PAD, 4), title_text, fill=C_TITLE, font=F_T)

    # Header
    y_hdr = TITLE_H
    draw.rectangle([PAD, y_hdr, total_w-PAD, y_hdr+HDR_H], fill=C_HDR_SUB)
    n_conf = len(conf_thresholds)
    grp_ci = 0
    for ci, (col, w) in enumerate(zip(cols, col_widths)):
        x = x_col(ci)
        if not is_grp(col):
            draw.rectangle([x, y_hdr, x+w, y_hdr+HDR_H], fill=C_HDR_GRP)
            ty = y_hdr+(HDR_H-13)//2
            tx = x+6 if ci<=1 else x+(w-_text_w(col,F_H1))//2
            draw.text((tx,ty), col, fill=C_WHITE, font=F_H1)
        else:
            ci_in = grp_ci % n_conf
            if ci_in == 0:
                gw = sum(col_widths[ci:ci+n_conf])
                draw.rectangle([x,y_hdr,x+gw,y_hdr+HDR_R1], fill=C_HDR_GRP)
                lbl = col.split("/c")[0].replace("_"," ")
                tw = _text_w(lbl, F_H1)
                draw.text((x+(gw-tw)//2, y_hdr+(HDR_R1-13)//2), lbl, fill=C_WHITE, font=F_H1)
                vline(x, y_hdr, y_hdr+HDR_H)
            sub_lbl = "c"+col.split("/c")[1]
            draw.rectangle([x, y_hdr+HDR_R1, x+w, y_hdr+HDR_H], fill=C_HDR_SUB)
            draw.text((x+(w-_text_w(sub_lbl,F_H2))//2, y_hdr+HDR_R1+(HDR_R2-11)//2),
                      sub_lbl, fill=C_WHITE, font=F_H2)
            draw.line([(x,y_hdr+HDR_R1),(x+w,y_hdr+HDR_R1)], fill=C_SEP, width=1)
            grp_ci += 1

    hline(y_hdr+HDR_H, width=2)
    y = y_hdr+HDR_H

    # Calculate best (data rows only)
    data_rows = [(l,s) for l,s in all_rows if l != "__group__"]
    def best_val(col):
        vals = [get_val_fn(l,s,col) for l,s in data_rows]
        vals = [v for v in vals if v is not None]
        if not vals: return None
        return min(vals) if col in lower_is_better_cols else max(vals)
    bests = {col: best_val(col) for col in best_cols}

    # Rows
    di = 0
    for label, split in all_rows:
        if label == "__group__":
            draw.rectangle([PAD,y,total_w-PAD,y+GRP_H], fill=C_GRP_BG)
            draw.text((PAD+8, y+(GRP_H-13)//2), split, fill=C_TITLE, font=F_M)
            hline(y+GRP_H, width=2); y+=GRP_H; continue

        bg = C_ROW1 if di%2==0 else C_ROW2
        draw.rectangle([PAD,y,total_w-PAD,y+ROW_H], fill=bg)
        draw.text((x_col(0)+6, y+(ROW_H-13)//2), label, fill=C_TEXT, font=F_M)
        draw.text((x_col(1)+6, y+(ROW_H-13)//2), split, fill=C_TEXT, font=F_C)

        grp_c2 = 0
        for ci, col in enumerate(cols[2:], start=2):
            val = get_val_fn(label, split, col)
            if col in int_cols and val is not None:
                txt = str(int(val))
            else:
                txt = _fmt(val) if val is not None else "—"
            best  = bests.get(col)
            isbest = (val is not None and best is not None and abs(val-best)<1e-6)
            draw_centered(txt, x_col(ci), y, col_widths[ci], ROW_H,
                          F_C, C_BEST if isbest else C_TEXT)
            if is_grp(col):
                if grp_c2 % n_conf == 0:
                    vline(x_col(ci), y, y+ROW_H, color=C_LINE)
                grp_c2 += 1

        hline(y+ROW_H); y+=ROW_H; di+=1

    draw.rectangle([PAD,TITLE_H,total_w-PAD,y+4], outline=C_LINE, width=1)
    draw.text((PAD, y+8), footer_text, fill=(120,100,80), font=F_C)
    img.save(out_path)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
#  4. TABLES 2a (FPR) and 2c (TPR)
# ─────────────────────────────────────────────────────────────────────────────

def make_table2(all_results, run_names, conf_vals, iou_vals,
                split_label, out_dir, conf_thresholds):
    """
    Creates and saves three summary PNG tables:
      - 2a: False Positive Rate (FPR)
      - 2b: False Negative Rate (FNR)
      - 2c: True Positive Rate (TPR)
    Groups columns by IoU and confidence values for easy reading.
    """
    fpr_cols = [f"FPR_IoU@{iou}/c{conf}" for iou in iou_vals for conf in conf_vals]
    tpr_cols = [f"TPR_IoU@{iou}/c{conf}" for iou in iou_vals for conf in conf_vals]
    fp_cols  = [f"#FP_IoU@{iou}/c{conf}" for iou in iou_vals for conf in conf_vals]
    tp_cols  = [f"#TP_IoU@{iou}/c{conf}" for iou in iou_vals for conf in conf_vals]
    W = 78

    all_rows = [("__group__", split_label)]
    for rn in run_names:
        all_rows.append((rn, split_label))

    def get_val(label, split, col):
        for iou_t in iou_vals:
            for conf_t in conf_vals:
                m = all_results[label].get((conf_t, iou_t), {})
                if col == f"FPR_IoU@{iou_t}/c{conf_t}": return m.get("FPR")
                if col == f"TPR_IoU@{iou_t}/c{conf_t}": return m.get("TPR")
                if col == f"#FP_IoU@{iou_t}/c{conf_t}": return m.get("FP")
                if col == f"#TP_IoU@{iou_t}/c{conf_t}": return m.get("TP")
                if col == "#GT": return m.get("GT")
        return None

    # 2a FPR
    _make_png_table(
        title_text=f"Table 2a — FPR (#pred with empty list / #GT)\nGrouped by IoU  |  conf: {conf_vals}",
        cols=["Run","Split","#GT",*fpr_cols,*fp_cols],
        col_widths=[160,55,52,*[W]*len(fpr_cols),*[W]*len(fp_cols)],
        all_rows=all_rows, get_val_fn=get_val,
        best_cols=fpr_cols, lower_is_better_cols=fpr_cols,
        int_cols={*fp_cols,"#GT"},
        footer_text=f"Green=lowest value  |  FPR=FP/GT  |  conf: {conf_vals}  |  IoU: {iou_vals} (Multi-match)",
        out_path=str(out_dir/"tabella2a_fpr.png"),
        conf_thresholds=conf_thresholds,
    )
    print(f"  [OK] Table 2a → {out_dir/'tabella2a_fpr.png'}")


    # 2b FNR
    fnr_cols = [f"FNR_IoU@{iou}/c{conf}" for iou in iou_vals for conf in conf_vals]
    fn_cols  = [f"#FN_IoU@{iou}/c{conf}" for iou in iou_vals for conf in conf_vals]

    def get_val_fnr(label, split, col):
        for iou_t in iou_vals:
            for conf_t in conf_vals:
                m = all_results[label].get((conf_t, iou_t), {})
                if col == f"FNR_IoU@{iou_t}/c{conf_t}": return m.get("FNR")
                if col == f"#FN_IoU@{iou_t}/c{conf_t}": return m.get("FN")
                if col == "#GT": return m.get("GT")
        return None

    _make_png_table(
        title_text=f"Table 2b — FNR (#GT with empty list / #GT)\nGrouped by IoU  |  conf: {conf_vals}",
        cols=["Run","Split","#GT",*fnr_cols,*fn_cols],
        col_widths=[160,55,52,*[W]*len(fnr_cols),*[W]*len(fn_cols)],
        all_rows=all_rows, get_val_fn=get_val_fnr,
        best_cols=fnr_cols, lower_is_better_cols=fnr_cols,
        int_cols={*fn_cols,"#GT"},
        footer_text=f"Green=lowest value  |  FNR=FN/GT  |  conf: {conf_vals}  |  IoU: {iou_vals} (Multi-match)",
        out_path=str(out_dir/"tabella2b_fnr.png"),
        conf_thresholds=conf_thresholds,
    )
    print(f"  [OK] Table 2b → {out_dir/'tabella2b_fnr.png'}")

    # 2c TPR
    _make_png_table(
        title_text=f"Table 2c — TPR (#match pred↔GT / #GT)\nGrouped by IoU  |  conf: {conf_vals}",
        cols=["Run","Split","#GT",*tpr_cols,*tp_cols],
        col_widths=[160,55,52,*[W]*len(tpr_cols),*[W]*len(tp_cols)],
        all_rows=all_rows, get_val_fn=get_val, best_cols=tpr_cols, lower_is_better_cols=[],
        int_cols={*tp_cols,"#GT"},
        footer_text=f"Green=highest value  |  TPR=TP/GT  |  conf: {conf_vals}  |  IoU: {iou_vals} (Multi-match)",
        out_path=str(out_dir/"tabella2c_tpr.png"),
        conf_thresholds=conf_thresholds,
    )
    print(f"  [OK] Table 2c → {out_dir/'tabella2c_tpr.png'}")


# ─────────────────────────────────────────────────────────────────────────────
#  6. COSMOS vs YOLO TABLE — numerical YES/NO count
# ─────────────────────────────────────────────────────────────────────────────

def make_cosmos_vs_yolo_table(all_results, run_names, conf_vals, iou_vals,
                             split_label, out_path):
    """
    Generates a table specifically focusing on how Cosmos's responses 
    (YES/NO) aligned with or corrected YOLO's True Positives and False Positives.
    """
    cols = ["Run", "Split", "#GT",
            *[f"YOLO_tot/c{c}" for c in conf_vals],
            *[f"C_YES/c{c}"    for c in conf_vals],
            *[f"C_NO/c{c}"     for c in conf_vals],
            *[f"YES_TP/c{c}"   for c in conf_vals],
            *[f"YES_FP/c{c}"   for c in conf_vals],
            *[f"NO_FP/c{c}"    for c in conf_vals],
            *[f"NO_TP/c{c}"    for c in conf_vals]]
    W        = 72
    cw       = [160, 55, 52] + [W] * (len(cols) - 3)
    iou_ref  = iou_vals[0]
    all_rows = [("__group__", split_label)] + [(rn, split_label) for rn in run_names]

    def get_val(label, split, col):
        for c in conf_vals:
            m = all_results[label].get((c, iou_ref), {})
            if col == f"YOLO_tot/c{c}": return m.get("yolo_total")
            if col == f"C_YES/c{c}":    return m.get("cosmos_yes_total")
            if col == f"C_NO/c{c}":     return m.get("cosmos_no_total")
            if col == f"YES_TP/c{c}":   return m.get("cosmos_yes_were_tp")
            if col == f"YES_FP/c{c}":   return m.get("cosmos_yes_were_fp")
            if col == f"NO_FP/c{c}":    return m.get("cosmos_no_were_fp")
            if col == f"NO_TP/c{c}":    return m.get("cosmos_no_were_tp")
            if col == "#GT":
                return all_results[label].get((conf_vals[0], iou_ref), {}).get("GT")
        return None

    int_cols = (
        {f"YOLO_tot/c{c}" for c in conf_vals} |
        {f"C_YES/c{c}"    for c in conf_vals} |
        {f"C_NO/c{c}"     for c in conf_vals} |
        {f"YES_TP/c{c}"   for c in conf_vals} |
        {f"YES_FP/c{c}"   for c in conf_vals} |
        {f"NO_FP/c{c}"    for c in conf_vals} |
        {f"NO_TP/c{c}"    for c in conf_vals} | {"#GT"}
    )
    best_cols       = [f"NO_FP/c{c}" for c in conf_vals]
    lower_is_better = ([f"YES_FP/c{c}" for c in conf_vals] +
                       [f"NO_TP/c{c}"  for c in conf_vals])

    _make_png_table(
        title_text=(f"Cosmos vs YOLO Table — YES/NO by detector"
                    f"  (IoU={iou_ref}, conf: {conf_vals})"),
        cols=cols, col_widths=cw, all_rows=all_rows,
        get_val_fn=get_val, best_cols=best_cols,
        lower_is_better_cols=lower_is_better, int_cols=int_cols,
        footer_text=("Green=better value  |  YES_TP: confirmed real empty  |  "
                     "YES_FP: confirmed YOLO FP  |  NO_FP: corrected FP  |  "
                     "NO_TP: discarded real empty"),
        out_path=str(out_path), conf_thresholds=conf_vals,
    )
    print(f"  [OK] Cosmos vs YOLO Table -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  7. CASES TABLE — Cosmos vs YOLO decision flow
# ─────────────────────────────────────────────────────────────────────────────

def make_casistiche_table(all_results, run_names, conf_val, iou_val, out_path):
    """
    Generates a flowchart-style PNG table detailing the 4 operational cases 
    of Cosmos on YOLO predictions, grouped by "CORRECT OUTCOME" and "WRONG OUTCOME",
    along with a summary section analyzing False Negatives (FN).
    """
    C_HDR_BG   = (60,  60,  80)
    C_HDR_FG   = (255, 255, 255)
    C_SEC_OK   = (210, 240, 210)
    C_SEC_ERR  = (245, 215, 215)
    C_ROW1_OK  = (240, 255, 240)
    C_ROW1_ERR = (255, 242, 242)
    C_ROW2_OK  = (225, 245, 225)
    C_ROW2_ERR = (250, 230, 230)
    C_TEXT     = (40,  40,  40)
    C_GREEN    = (0,   130,  0)
    C_RED      = (180,  20, 20)
    C_LINE     = (200, 185, 170)
    C_BG       = (255, 252, 248)
    C_TITLE    = (80,  50,  20)
    C_ESITO_OK = (0,   110,  0)
    C_ESITO_ER = (160,  10, 10)

    FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    FONT_REG  = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    F_T = _font(FONT_BOLD, 13)
    F_H = _font(FONT_BOLD, 11)
    F_C = _font(FONT_REG,  11)
    F_M = _font(FONT_BOLD, 11)

    TITLE_H = 30; HDR_H = 40; ROW_H = 38; SEC_H = 26; FOOT_H = 28; PAD = 14
    SUMMARY_H = 34 * 3 + 30  # 3 summary rows + section title
    col_w_cas  = 170; col_w_desc = 270; col_w_run = 140; col_w_esit = 130
    n_runs  = len(run_names)
    total_w = PAD*2 + col_w_cas + col_w_desc + col_w_run * n_runs + col_w_esit
    total_h = TITLE_H + HDR_H + 3 * SEC_H + 3 * ROW_H + SUMMARY_H + FOOT_H + 10

    img  = Image.new("RGB", (total_w, total_h), C_BG)
    draw = ImageDraw.Draw(img)

    def hline(y, color=None, width=1):
        draw.line([(PAD, y), (total_w - PAD, y)], fill=color or C_LINE, width=width)

    def col_x(ci):
        xs = [PAD,
              PAD + col_w_cas,
              PAD + col_w_cas + col_w_desc]
        for _ in range(n_runs):
            xs.append(xs[-1] + col_w_run)
        xs.append(xs[-1] + col_w_esit)
        return xs[ci]

    def center_text(text, x, y, w, h, font, color):
        tw = _text_w(text, font)
        draw.text((x + (w - tw) // 2, y + (h - 13) // 2), text, fill=color, font=font)

    # Title
    draw.text((PAD, 6),
              f"Cosmos vs YOLO Cases  (conf={conf_val}, IoU={iou_val})",
              fill=C_TITLE, font=F_T)

    # Header
    y = TITLE_H
    draw.rectangle([PAD, y, total_w - PAD, y + HDR_H], fill=C_HDR_BG)
    draw.text((col_x(0) + 6, y + (HDR_H - 13) // 2), "Case",   fill=C_HDR_FG, font=F_H)
    draw.text((col_x(1) + 6, y + (HDR_H - 13) // 2), "Description", fill=C_HDR_FG, font=F_H)
    for ri, rn in enumerate(run_names):
        m        = all_results[rn].get((conf_val, iou_val), {})
        yolo_tot = m.get("yolo_total", 0)
        cx       = col_x(2 + ri)
        tw1 = _text_w(rn, F_H)
        draw.text((cx + (col_w_run - tw1) // 2, y + 5),          rn,                 fill=C_HDR_FG,       font=F_H)
        sub = f"YOLO tot={yolo_tot}"
        tw2 = _text_w(sub, F_C)
        draw.text((cx + (col_w_run - tw2) // 2, y + HDR_H - 17), sub,                fill=(210, 200, 190), font=F_C)
    center_text("Final outcome", col_x(2 + n_runs), y, col_w_esit, HDR_H, F_H, C_HDR_FG)
    hline(y + HDR_H, color=(100, 100, 120), width=2)
    y += HDR_H

    casistiche = [
        ("YOLO YES / was TP",  "Cosmos confirms a real empty space",            "cosmos_yes_were_tp", "\u2713  final TP",  True),
        ("YOLO YES / was FP",  "Cosmos confirms a YOLO false positive",       "cosmos_yes_were_fp", "\u2717  final FP",  False),
        ("YOLO NO  / was FP",  "Cosmos corrects a YOLO false positive",       "cosmos_no_were_fp",  "\u2713  reduces FP",  True),
    ]

    prev_ok = None
    di = 0
    for cas_label, desc, key, esito, is_ok in casistiche:
        if prev_ok != is_ok:
            sec_bg  = C_SEC_OK  if is_ok else C_SEC_ERR
            sec_lbl = "CASES WITH CORRECT OUTCOME" if is_ok else "CASES WITH WRONG OUTCOME"
            sec_col = C_ESITO_OK if is_ok else C_ESITO_ER
            draw.rectangle([PAD, y, total_w - PAD, y + SEC_H], fill=sec_bg)
            draw.text((PAD + 8, y + (SEC_H - 13) // 2), sec_lbl, fill=sec_col, font=F_M)
            hline(y + SEC_H)
            y += SEC_H
            prev_ok = is_ok

        row_bg = (C_ROW1_OK if is_ok else C_ROW1_ERR) if di % 2 == 0 else (C_ROW2_OK if is_ok else C_ROW2_ERR)
        draw.rectangle([PAD, y, total_w - PAD, y + ROW_H], fill=row_bg)
        draw.text((col_x(0) + 6, y + (ROW_H - 13) // 2), cas_label, fill=C_TEXT, font=F_M)
        draw.text((col_x(1) + 6, y + (ROW_H - 13) // 2), desc,      fill=C_TEXT, font=F_C)

        for ri, rn in enumerate(run_names):
            m        = all_results[rn].get((conf_val, iou_val), {})
            yolo_tot = m.get("yolo_total", 1) or 1
            n        = m.get(key, 0)
            pct      = n / yolo_tot * 100
            txt      = f"{n}  ({pct:.1f}%)"
            col      = C_GREEN if is_ok else C_RED
            cx       = col_x(2 + ri)
            tw       = _text_w(txt, F_M)
            draw.text((cx + (col_w_run - tw) // 2, y + (ROW_H - 13) // 2), txt, fill=col, font=F_M)

        esito_col = C_ESITO_OK if is_ok else C_ESITO_ER
        center_text(esito, col_x(2 + n_runs), y, col_w_esit, ROW_H, F_M, esito_col)
        hline(y + ROW_H)
        y  += ROW_H
        di += 1

    # ── Summary: Baseline FN (YOLO only) vs Pipeline FN (with Cosmos) ───────
    # Both calculated PER GT (same unit of measurement, safely subtractable
    # from one another — unlike the per-box counts above, there is no
    # risk of double counting from multi-match here).
    SUM_BG_HDR = (50, 70, 50)
    y += 6
    draw.rectangle([PAD, y, total_w - PAD, y + 26], fill=SUM_BG_HDR)
    draw.text((PAD + 8, y + 6), "FN — baseline (YOLO only) vs pipeline (with Cosmos) comparison, per GT",
               fill=(255, 255, 255), font=F_H)
    y += 26

    summary_rows = [
        ("Baseline FN (YOLO only, no filters)",   "fn_missed_by_yolo", None),
        ("Pipeline FN (after Cosmos filter)",             "FN",                None),
        ("Δ FN caused by Cosmos (pipeline − baseline)", "fn_rejected_by_cosmos", "delta"),
    ]
    for label, key, mark in summary_rows:
        row_bg = (245, 248, 240)
        draw.rectangle([PAD, y, total_w - PAD, y + 34], fill=row_bg)
        draw.text((col_x(0) + 6, y + 10), label, fill=C_TEXT, font=F_C)
        for ri, rn in enumerate(run_names):
            m   = all_results[rn].get((conf_val, iou_val), {})
            n   = m.get(key, 0)
            txt = f"+{n}" if mark == "delta" else str(n)
            col = C_RED if mark == "delta" else C_TEXT
            cx  = col_x(2 + ri)
            tw  = _text_w(txt, F_M)
            draw.text((cx + (col_w_run - tw) // 2, y + 10), txt, fill=col, font=F_M)
        hline(y + 34)
        y += 34

    draw.rectangle([PAD, TITLE_H, total_w - PAD, y + 4], outline=C_LINE, width=1)
    draw.text((PAD, y + 8),
              f"% of total YOLO BBs  |  conf={conf_val}  IoU={iou_val}  Green=correct  Red=wrong  |  "
              f"FN Summary: same unit (per GT), valid subtraction",
              fill=(120, 100, 80), font=F_C)
    img.save(str(out_path))
    print(f"  [OK] Cases table -> {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """
    Main execution script: parses arguments, loads YOLO model runs or JSON outputs 
    from Cosmos evaluations, triggers evaluations, and orchestrates the creation 
    of metric plots and tables.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",      default=MODE_DEFAULT, choices=["yolo","cosmos"])
    # YOLO
    parser.add_argument("--runs",      nargs="+",
                        help="name:weights_path  (only --mode yolo)")
    parser.add_argument("--images",    help="images folder (only --mode yolo)")
    parser.add_argument("--labels",    help="labels folder (only --mode yolo)")
    parser.add_argument("--split",     default="test")
    # Cosmos
    parser.add_argument("--json_path", default=JSON_PATH_DEFAULT,
                        help="evaluate_yolo_cosmos.py JSON (only --mode cosmos)")
    parser.add_argument("--cosmos_thr",  nargs="+", type=float, default=[None],
                        help="Ignored in cosmos v2 mode (no threshold)")
    parser.add_argument("--prompt",    default=PROMPT_DEFAULT,
                        choices=["context","no_context"])
    # Common
    parser.add_argument("--conf",      nargs="+", type=float, default=CONF_DEFAULT)
    parser.add_argument("--iou",       nargs="+", type=float, default=IOU_DEFAULT)
    parser.add_argument("--bar_conf",  type=float, default=BAR_CONF_DEFAULT)
    parser.add_argument("--out_dir",   default=OUT_DIR_DEFAULT)
    parser.add_argument("--max_images",type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ───────────────────────────────────────────────────────────
    if args.mode == "yolo":
        from ultralytics import YOLO as _Y
        run_names = []; run_paths = {}
        for r in args.runs:
            name, path = r.split(":",1)
            run_names.append(name); run_paths[name] = path

        pairs = load_pairs(Path(args.images), Path(args.labels))
        if args.max_images: pairs = pairs[:args.max_images]
        print(f"Images: {len(pairs)}  Total GT: {sum(len(s['gt']) for s in pairs)}")

        all_results = {}
        for name in run_names:
            print(f"\n[{name}] Evaluating...")
            yolo = _Y(run_paths[name])
            all_results[name] = evaluate_yolo_run(yolo, pairs, args.conf, args.iou)

        split_label = args.split.upper()
        bar_title   = (f"Full pipeline (YOLO) — conf={args.bar_conf}  split={args.split}\n"
                       f"{len(run_names)} runs × {len(args.iou)} IoU thresholds")

    else:  # cosmos v2
        run_names, results_by_thr = load_cosmos_results(
            args.json_path, args.conf, args.iou, args.cosmos_thr, args.prompt)
        split_label = "TEST"
        all_results = results_by_thr[None]
        bar_title   = (f"YOLO-only vs YOLO+Cosmos ({args.prompt}) — conf={args.bar_conf}  "
                       f"prompt={args.prompt}  split=test\n"
                       f"{len(run_names)} detectors x {len(args.iou)} IoU thresholds")
        print(f"\n{'='*50}\n  output: {out_dir}\n{'='*50}")
        print("  [1/4] Bar chart...")
        plot_bars(all_results, run_names, args.bar_conf, args.iou,
                  bar_title, out_dir/"bar_metrics.png")
        print("  [2/4] Tables 2a/2b/2c...")
        make_table2(all_results, run_names, args.conf, args.iou,
                    split_label, out_dir, args.conf)
        print("  [3/4] Cosmos vs YOLO Table (numeric)...")
        make_cosmos_vs_yolo_table(all_results, run_names, args.conf, args.iou,
                                split_label, out_dir/"tabella_cosmos_vs_yolo.png")
        print("  [4/4] Cases table (decision flow)...")
        make_casistiche_table(all_results, run_names, args.bar_conf, args.iou[0],
                              out_dir/"tabella_casistiche.png")
        print(f"\nEverything saved in: {out_dir}")
        return

    # ── YOLO-only Output ──────────────────────────────────────────────────────
    print(f"\n[1/2] Bar chart...")
    plot_bars(all_results, run_names, args.bar_conf, args.iou,
              bar_title, out_dir/"bar_metrics.png")
    print(f"\n[2/2] Tables 2a/2b/2c (FPR/FNR/TPR)...")
    make_table2(all_results, run_names, args.conf, args.iou,
                split_label, out_dir, args.conf)
    print(f"\nEverything saved in: {out_dir}")


if __name__ == "__main__":
    main()