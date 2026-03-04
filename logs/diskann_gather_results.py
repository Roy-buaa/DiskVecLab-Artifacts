#!/usr/bin/env python3
"""
Parse DiskANN search logs and aggregate the table metrics into a single CSV.

Expected table snippet inside each log (example):

     L   Beamwidth             QPS    Mean Latency    99.9 Latency        Mean IOs    Mean IO (us)         CPU (s)      Mean IOLat     IOLat Ratio        Mean DCs       Recall@10
====================================================================================================================================================================================================
    10           1        14208.26         8821.80           19601           18.80         7722.55         1052.30          410.73            0.88         3884.81           14.30

The script will:
- Walk an input directory (default: /root/paper/sift1b/logs) for *.log files
- Extract one or more table rows from each log (robust to multi‑row tables)
- Parse metadata from the filename when available: timestamp, ratio, quant, W, L, optional tag
  (filename pattern used by the provided batch script: {ts}_{ratio}_W{W}_L{L}_{quant}[_tag].log)
- Prefer W/L from the content table (Beamwidth/L columns) when present; fallback to filename
- Output a CSV with unified columns

Usage:
    python3 gather_logs_to_csv.py \
        --in_dir /root/paper/sift1b/logs \
        --out_csv /root/paper/sift1b/agg_metrics.csv

Optional filters:
    --glob "*.log"            glob pattern for files (default: *.log)

"""
from __future__ import annotations
import argparse
import csv
import re
from pathlib import Path
from typing import Iterable, List, Dict, Optional

DEFAULT_IN_DIR = "/root/paper/sift1b/logs"
DEFAULT_OUT_CSV = "/root/paper/sift1b/agg_metrics.csv"

# Split columns on 2+ spaces, so labels like "Mean IO (us)" remain intact
COL_SPLIT = re.compile(r"\s{2,}")
HEADER_HINT = re.compile(r"Recall@\d+|Recall@K", re.IGNORECASE)
SEP_LINE = re.compile(r"^=+")
NUM_ROW = re.compile(r"^\s*[-+]?\d")  # data row starts with a number (possibly spaces before)

# Filename patterns (try in order)
# PATTERNS = [
#     re.compile(r"^(?P<ts>\d{4}_\d{4})_(?P<ratio>[^_]+)_W(?P<W>\d+)_L(?P<L>\d+(?:-\d+)*)_(?P<quant>[A-Za-z0-9]+)(?:_(?P<tag>[^.]+))?\.log$"),
#     re.compile(r"^(?P<ts>\d{4}_\d{4})_W(?P<W>\d+)_L(?P<L>\d+(?:-\d+)*)_(?P<quant>[A-Za-z0-9]+)(?:_(?P<tag>[^.]+))?\.log$"),
#     re.compile(r"^(?P<ts>\d{4}_\d{4})_(?P<ratio>[^_]+)_W(?P<W>\d+?)_(?P<quant>[A-Za-z0-9]+)(?:_(?P<tag>[^.]+))?\.log$"),
# ]

PATTERNS = [
    # NEW: 1219_1610_quant128B_W4_L..._RABITQ(.log)  (tag固定在ts后面，且以quant开头)
    re.compile(
        r"^(?P<ts>\d{4}_\d{4})_(?P<ratio>quant[^_]+)_W(?P<W>\d+)_L(?P<L>.+)_(?P<quant>[A-Za-z0-9]+)(?:_(?P<extra>[^.]+))?\.log$"
    ),

    # old: ts_ratio_Wx_L..._quant(_tag).log   (L can contain underscores)
    re.compile(
        r"^(?P<ts>\d{4}_\d{4})_(?P<ratio>[^_]+)_W(?P<W>\d+)_L(?P<L>.+)_(?P<quant>[A-Za-z0-9]+)(?:_(?P<tag>[^.]+))?\.log$"
    ),

    # old: ts_Wx_L..._quant(_tag).log         (L can contain underscores)
    re.compile(
        r"^(?P<ts>\d{4}_\d{4})_W(?P<W>\d+)_L(?P<L>.+)_(?P<quant>[A-Za-z0-9]+)(?:_(?P<tag>[^.]+))?\.log$"
    ),

    # old: ts_ratio_Wx_quant(_tag).log        (no L)
    re.compile(
        r"^(?P<ts>\d{4}_\d{4})_(?P<ratio>[^_]+)_W(?P<W>\d+?)_(?P<quant>[A-Za-z0-9]+)(?:_(?P<tag>[^.]+))?\.log$"
    ),
]


def parse_filename_meta(path: Path) -> Dict[str, Optional[str]]:
    name = path.name
    out: Dict[str, Optional[str]] = {
        "filename": name,
        "ts": None,
        "ratio": None,
        "quant": None,
        "W_file": None,
        "L_file": None,
        "tag": None,
    }
    for pat in PATTERNS:
        m = pat.match(name)
        if m:
            d = m.groupdict()
            out["ts"] = d.get("ts")
            out["ratio"] = d.get("ratio")
            out["quant"] = d.get("quant")
            out["W_file"] = d.get("W")
            out["L_file"] = d.get("L")
            out["tag"] = d.get("tag")
            break
    return out


def iter_tables(lines: Iterable[str]):
    """Yield (header_cols, data_row_text) pairs for each table found."""
    lines = list(lines)
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].rstrip("\n")
        # find header line that contains Recall@X (heuristic anchor)
        if HEADER_HINT.search(line):
            header_cols = COL_SPLIT.split(line.strip())
            # next: a separator line of '=' and then one or more data rows
            j = i + 1
            # skip exactly one separator (or multiple)
            while j < n and SEP_LINE.match(lines[j]):
                j += 1
            # collect subsequent numeric rows until a blank or another header/sep
            while j < n and NUM_ROW.match(lines[j]):
                row = COL_SPLIT.split(lines[j].strip())
                yield header_cols, row
                j += 1
            i = j
            continue
        i += 1


def parse_log(path: Path) -> List[Dict[str, str]]:
    meta = parse_filename_meta(path)
    rows: List[Dict[str, str]] = []
    try:
        with path.open("r", errors="ignore") as f:
            content = f.readlines()
    except Exception as e:
        print(f"WARN: failed to read {path}: {e}")
        return rows

    for header, row in iter_tables(content):
        # Align header and row length (defensive)
        if len(row) < len(header):
            # try padding with empty strings (rare)
            row = row + [""] * (len(header) - len(row))
        elif len(row) > len(header):
            # truncate extras
            row = row[:len(header)]
        rec: Dict[str, str] = {}
        for k, v in zip(header, row):
            rec[k.strip()] = v.strip()

        # Attach filename meta
        rec["_file"] = str(path)
        rec["_filename"] = meta.get("filename") or path.name
        rec["_ts"] = meta.get("ts") or ""
        rec["_ratio"] = meta.get("ratio") or ""
        rec["_quant"] = meta.get("quant") or ""
        rec["_tag"] = meta.get("tag") or ""

        # Prefer content W/L (Beamwidth/L); fallback to filename meta
        if "Beamwidth" in rec and rec["Beamwidth"]:
            rec["_W"] = rec["Beamwidth"]
        else:
            rec["_W"] = meta.get("W_file") or ""
        if "L" in rec and rec["L"]:
            rec["_L"] = rec["L"]
        else:
            rec["_L"] = meta.get("L_file") or ""

        rows.append(rec)
    return rows


def gather(in_dir: Path, glob_pat: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    files = sorted(in_dir.glob(glob_pat))
    for p in files:
        if p.is_file():
            rows.extend(parse_log(p))
    return rows


def infer_all_columns(rows: List[Dict[str, str]]) -> List[str]:
    # Preserve a sensible order: metadata first, then known metrics (if present), then any others
    known = [
        "L", "Beamwidth", "QPS", "Mean Latency", "99.9 Latency", "Mean IOs", "Mean IO (us)",
        "CPU (s)", "Mean IOLat", "IOLat Ratio", "Mean DCs", "Recall@10"
    ]
    meta_cols = ["_filename", "_file", "_ts", "_ratio", "_quant", "_W", "_L", "_tag"]
    present = set()
    for r in rows:
        present.update(r.keys())
    # remove meta keys from present (we will place meta first explicitly)
    for m in meta_cols:
        present.discard(m)
    ordered = meta_cols + [k for k in known if k in present] + sorted(present - set(known))
    return ordered


def write_csv(rows: List[Dict[str, str]], out_csv: Path) -> None:
    if not rows:
        print("No rows parsed. Nothing to write.")
        return
    fieldnames = infer_all_columns(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {len(rows)} rows to {out_csv}")


def main():
    ap = argparse.ArgumentParser(description="Aggregate DiskANN log metrics to CSV")
    ap.add_argument("--in_dir", default="", help="Directory containing .log files")
    ap.add_argument("--glob", default="*.log", help="Glob pattern for log files")
    ap.add_argument("--out_csv", default="", help="Output CSV path")
    args = ap.parse_args()

    if args.in_dir == "":
        args.in_dir = input(f"Input directory (default: {DEFAULT_IN_DIR}): ") or DEFAULT_IN_DIR
    if args.out_csv == "":
        args.out_csv = input(f"Output CSV path (default: {DEFAULT_OUT_CSV}): ") or DEFAULT_OUT_CSV


    in_dir = Path(args.in_dir)
    out_csv = Path(args.out_csv)

    rows = gather(in_dir, args.glob)
    write_csv(rows, out_csv)


if __name__ == "__main__":
    main()
