#!/usr/bin/env python3
"""
plot_tikz_single_shot.py
------------------------
Generate standalone TikZ/pgfplots figures for the single-shot W-sweep:
**Logical error rate vs window size**, one figure per p-value, panels per decoder.

Reads any CSV in data/results/single_shot/ matching the two schemas:
    old (run_single_shot.py):  decoder,code,w,ler,errors,shots
    new (single_shot.py):      code,n,variant,decoder,W,T,p,noise,ler,errors,shots

Output: fig/single_shot/ler_per_round_p<P>_<noise>.tex (standalone, compilable).

Usage:
    python util/plot_tikz_single_shot.py              # auto-discover all CSVs
    python util/plot_tikz_single_shot.py --p 0.001    # filter by p-value
    python util/plot_tikz_single_shot.py --T 14       # override T for ler_per_round
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR  = PROJECT_ROOT / "data" / "results" / "single_shot"
FIG_DIR      = PROJECT_ROOT / "fig" / "single_shot"

# Mapped to the line styles in your target format.
# All will use colTess and mark=*
STYLES = {
    "tetrahedral":    dict(line_style="solid"),
    "ja25/rm":        dict(line_style="dashed"),
    "ja25/rm_opt":    dict(line_style="dash dot"),
    "ja25/tetra":     dict(line_style="dotted"),
    "ja25/tetra_opt": dict(line_style="dash dot dot"),
}
CODE_ORDER  = ["tetrahedral", "ja25/rm", "ja25/rm_opt", "ja25/tetra", "ja25/tetra_opt"]
DECODER_ORDER = ["tesseract"]


def _normalize_label(code, variant):
    if not variant or variant == "":
        return code
    return f"{code}/{variant}".replace("ja25_transversal_t", "ja25")


def _latex_escape(s):
    return s.replace("_", r"\_")


def load_csv(path):
    """Yield dicts with keys (decoder, code, w, ler, errors, shots, p, noise, T, tps)."""
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Infer schema by header presence.
            if "W" in row:  # new schema
                code = _normalize_label(row["code"], row.get("variant", ""))
                # Prefer new marginal decode time; fall back to old wall-clock column.
                tps_raw = row.get("decode_s_per_shot") or row.get("time_per_shot_s", "")
                yield dict(
                    decoder = row["decoder"],
                    code    = code,
                    w       = int(row["W"]),
                    ler     = float(row["ler"]),
                    errors  = int(row["errors"]),
                    shots   = int(row["shots"]),
                    p       = float(row["p"]),
                    noise   = row.get("noise", "phenomenological"),
                    T       = int(row["T"]),
                    tps     = float(tps_raw) if tps_raw not in ("", None) else None,
                )
            else:  # old schema — infer p, noise, T from filename
                m = re.search(r"T(\d+)_p([\d.e-]+)_s\d+_(phenom|dep)", path.name)
                T     = int(m.group(1)) if m else 14
                p     = float(m.group(2)) if m else 0.0
                noise = {"phenom": "phenomenological",
                         "dep":    "depolarizing"}.get(m.group(3), "unknown") if m else "unknown"
                yield dict(
                    decoder = row["decoder"],
                    code    = row["code"],
                    w       = int(row["w"]),
                    ler     = float(row["ler"]),
                    errors  = int(row["errors"]),
                    shots   = int(row["shots"]),
                    p       = p,
                    noise   = noise,
                    T       = T,
                    tps     = None,
                )


def collect(csv_paths):
    """Group rows by (p, noise, decoder, code) → sorted list[(w, ler, errors, shots, T, tps)]."""
    bucket = defaultdict(list)
    for path in csv_paths:
        for row in load_csv(path):
            key = (row["p"], row["noise"], row["decoder"], row["code"])
            bucket[key].append((row["w"], row["ler"], row["errors"],
                                row["shots"], row["T"], row["tps"]))
    for key in bucket:
        bucket[key].sort(key=lambda t: t[0])
    return bucket


def emit_tex(p, noise, bucket, T_override=None, out_path=None):
    """Write one standalone TikZ file for a given (p, noise).

    Single-row groupplot of LER/round, with a dashed reference tick at the
    largest T in the data so the reader can see where W = T sits.
    """
    keys = [(dec, code) for dec in DECODER_ORDER for code in CODE_ORDER
            if (p, noise, dec, code) in bucket]
    if not keys:
        return None
    decoders_present = sorted({dec for dec, _ in keys},
                              key=lambda d: DECODER_ORDER.index(d))
    n_panels = len(decoders_present)

    T_mark = max(T for dec, code in keys
                 for _, _, _, _, T, _ in bucket[(p, noise, dec, code)])

    lines = []
    lines.append(r"\documentclass[tikz,border=5pt]{standalone}")
    lines.append(r"\usepackage{pgfplots}")
    lines.append(r"\pgfplotsset{compat=1.18}")
    lines.append(r"\usepgfplotslibrary{groupplots}")
    lines.append(r"\begin{document}")
    lines.append(r"\definecolor{colTess}{RGB}{31,119,180}")
    lines.append(r"\begin{tikzpicture}")

    lines.append(r"\begin{groupplot}[")
    lines.append(f"  group style={{group size={n_panels} by 1, "
                 r"horizontal sep=1.6cm, vertical sep=1.2cm, "
                 r"ylabels at=edge left},")
    lines.append(r"  width=0.85\linewidth,")
    lines.append(r"  height=0.85\linewidth,")
    lines.append(r"  xlabel={Window size $w$},")
    lines.append(r"  ymode=log,")
    lines.append(r"  xmin=0.5,")
    lines.append(r"  grid=both,")
    lines.append(r"  major grid style={line width=.2pt,draw=gray!30},")
    lines.append(r"  minor grid style={line width=.1pt,draw=gray!10},")
    lines.append(r"  clip=false,")
    lines.append(r"  title style={font=\small\bfseries},")
    lines.append(r"  tick label style={font=\footnotesize},")
    lines.append(r"  label style={font=\footnotesize},")
    lines.append(r"]")
    lines.append("")

    for dec in decoders_present:
        lines.append(r"\nextgroupplot[ylabel={Logical error rate}, title={}]")
        lines.append("")
        lines.append(rf"\coordinate (Tx) at (axis cs:{T_mark}, 0);")
        lines.append(r"\draw[dashed, black!70, line width=0.5pt] "
                     r"(Tx |- current axis.south) -- "
                     r"(Tx |- current axis.north);")
        lines.append(r"\node[anchor=north west, font=\footnotesize, "
                     r"inner sep=2pt] at "
                     r"(Tx |- current axis.north) {$T$};")
        lines.append("")
        for code in CODE_ORDER:
            if (p, noise, dec, code) not in bucket:
                continue
            pts = bucket[(p, noise, dec, code)]
            style = STYLES.get(code, dict(line_style="solid"))
            opts = f"color=colTess, mark=*, {style['line_style']}, thick, mark size=1.8pt, forget plot"
            coords = " ".join(
                f"({w},{(ler / (T_override or T)):.6e})"
                for w, ler, _, _, T, _ in pts if ler > 0
            )
            if not coords:
                floor_pts = [(w, 1.0 / s / (T_override or T))
                             for w, _, _, s, T, _ in pts]
                coords = " ".join(f"({w},{v:.6e})" for w, v in floor_pts)
            lines.append(f"%% {code}")
            lines.append(rf"\addplot[{opts}] coordinates {{{coords}}};")
        lines.append("")

    lines.append(r"\end{groupplot}")
    lines.append(r"\end{tikzpicture}")
    lines.append(r"\end{document}")

    out_path = Path(out_path) if out_path else (
        FIG_DIR / f"ler_per_round_p{p:g}_{noise}.tex")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate TikZ LER-per-round plots for single-shot W-sweeps.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out-dir",     type=Path, default=FIG_DIR)
    parser.add_argument("--p",           type=float, default=None,
                        help="Filter to this p-value.")
    parser.add_argument("--noise",       type=str, default=None,
                        help="Filter to this noise model.")
    parser.add_argument("--T",           type=int, default=None,
                        help="Override T for LER-per-round normalization.")
    args = parser.parse_args()

    csv_paths = sorted(args.results_dir.glob("*.csv"))
    if not csv_paths:
        print(f"No CSVs found in {args.results_dir}")
        return
    bucket = collect(csv_paths)
    groups = sorted({(p, n) for (p, n, _, _) in bucket})
    if args.p is not None:
        groups = [(p, n) for (p, n) in groups if abs(p - args.p) < 1e-12]
    if args.noise:
        groups = [(p, n) for (p, n) in groups if n == args.noise]

    if not groups:
        print("No matching (p, noise) groups.")
        return

    for p, noise in groups:
        out = emit_tex(p, noise, bucket, T_override=args.T,
                       out_path=args.out_dir /
                                f"ler_per_round_p{p:g}_{noise}.tex")
        if out:
            print(f"wrote {out}")


if __name__ == "__main__":
    main()