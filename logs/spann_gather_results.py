#!/usr/bin/env python3
"""
Parse SPANN search logs (ssdserving) into a CSV for analysis.

- Walks a log root directory (e.g. /root/paper/sift1m/logs).
- Finds files named *search_mc*_ir*.log.
- Extracts metadata from path and filename.
- Parses QPS, recall, latency distributions, IO distributions, etc.
"""

import argparse
import csv
import glob
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Section name -> base key for its 7-number distribution
SECTION_KEYS = {
    "Ex Elements Count:": "ex_elements",
    "Head Latency Distribution:": "head_latency_ms",
    "Ex Latency Distribution:": "ex_latency_ms",
    "Total Latency Distribution:": "total_latency_ms",
    "Total Disk Page Access Distribution:": "disk_pages",
    "Total Disk IO Distribution:": "disk_ios",
}

DIST_SUFFIXES = ["avg", "p50", "p90", "p95", "p99", "p999", "max"]


def clean_prefix(line: str) -> str:
    """Strip leading '[1] ' / '[4] ' etc."""
    return re.sub(r"^\[\d+\]\s*", "", line)


def parse_path_meta(path: Path) -> Dict[str, Any]:
    """
    Extract dataset, exp_name, quant_size_B, quant_type,
    maxcheck, internal_result_num, ts from log path.
    """
    meta: Dict[str, Any] = {}
    meta["log_path"] = str(path)
    meta["log_filename"] = path.name

    parts = list(path.parts)
    dataset = ""
    exp_name = ""

    if "logs" in parts:
        idx_logs = parts.index("logs")
        if idx_logs >= 1:
            dataset = parts[idx_logs - 1]
        if idx_logs + 1 < len(parts):
            exp_name = parts[idx_logs + 1]

    meta["dataset"] = dataset or None
    meta["exp_name"] = exp_name or None

    # Thread count: infer from directory names like "..._16t" / "..._128t".
    # Example: <root>/spann/laion50m512dood_spann_16t/2026..._search_mc...
    meta["T"] = None
    t_pat = re.compile(r"(?:^|_)(\d+)t$")
    for part in reversed(parts):
        m_t = t_pat.search(part)
        if m_t:
            try:
                meta["T"] = int(m_t.group(1))
            except Exception:
                meta["T"] = None
            break

    # exp_name pattern: <dataset>_q<quant_size>B_<quant_type>
    quant_size_B: Optional[int] = None
    quant_type: Optional[str] = None
    if exp_name and "_q" in exp_name and "B_" in exp_name:
        try:
            _, rest = exp_name.split("_q", 1)
            size_str, quant_type = rest.split("B_", 1)
            quant_size_B = int(size_str)
        except Exception:
            pass

    meta["quant_size_B"] = quant_size_B
    meta["quant_type"] = quant_type

    # from filename: 20251115_223057_search_mc100_ir100.log
    name = path.name
    # ts prefix
    if "_search_" in name:
        meta["log_ts"] = name.split("_search_")[0]
    else:
        meta["log_ts"] = None

    # maxcheck / internal_result_num
    m = re.search(r"search_mc(\d+)_ir(\d+)", name)
    if m:
        meta["maxcheck"] = int(m.group(1))
        meta["internal_result_num"] = int(m.group(2))
    else:
        meta["maxcheck"] = None
        meta["internal_result_num"] = None

    return meta


def parse_log_file(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    clean_lines = [clean_prefix(l) for l in lines]

    result: Dict[str, Any] = {}
    result.update(parse_path_meta(path))

    # TIME header(s)
    # Current logs use TIME_START / TIME_END, but we keep backward-compat for older '# TIME:'
    m = re.search(r"# TIME_START:\s*([^\s]+)", text)
    result["log_time_start"] = m.group(1) if m else None

    m = re.search(r"# TIME_END:\s*([^\s]+)", text)
    result["log_time_end"] = m.group(1) if m else None

    m = re.search(r"# TIME:\s*([^\s]+)", text)
    result["log_time"] = m.group(1) if m else (result["log_time_start"] or None)

    # Optional: compute elapsed wall time if both timestamps exist
    result["elapsed_wall_s"] = None
    if result["log_time_start"] and result["log_time_end"]:
        try:
            t0 = datetime.fromisoformat(result["log_time_start"])
            t1 = datetime.fromisoformat(result["log_time_end"])
            result["elapsed_wall_s"] = (t1 - t0).total_seconds()
        except Exception:
            pass

    # QPS line
    m = re.search(
        r"Finish sending in ([0-9\.]+) seconds, actuallQPS is ([0-9\.]+), query count ([0-9]+)",
        text,
    )
    if m:
        result["total_time_s"] = float(m.group(1))
        result["qps"] = float(m.group(2))
        result["num_queries"] = int(m.group(3))
    else:
        result["total_time_s"] = None
        result["qps"] = None
        result["num_queries"] = None

    # Recall / MRR
    # Examples observed in logs:
    #   Recall@100: 0.471960 MRR@100: 1.000000
    #   Recall100@100: 0.471960 MRR@100: 1.000000
    #   Recall@10: ... / Recall10@10: ...
    result["recall"] = None
    result["mrr"] = None
    result["recall_k"] = None
    result["mrr_k"] = None

    recall_matches: List[Dict[str, Any]] = []
    recall_pat = re.compile(
        r"Recall(?:(?P<k_prefix>\d+)@(?P<k_same>\d+)|@(?P<k_at>\d+)):\s*(?P<recall>[0-9\.]+)\s+MRR@(?P<mrrk>\d+):\s*(?P<mrr>[0-9\.]+)"
    )
    for m in recall_pat.finditer(text):
        k_str = m.group("k_at") or m.group("k_same") or m.group("k_prefix")
        if not k_str:
            continue
        try:
            k = int(k_str)
            mrrk = int(m.group("mrrk"))
            if mrrk != k:
                continue
            recall_v = float(m.group("recall"))
            mrr_v = float(m.group("mrr"))
            recall_matches.append({"k": k, "recall": recall_v, "mrr": mrr_v})
        except Exception:
            continue

    # Populate per-K columns and also a generic (latest) recall/mrr
    for rm in recall_matches:
        k = rm["k"]
        result[f"recall_at{k}"] = rm["recall"]
        result[f"mrr_at{k}"] = rm["mrr"]

    if recall_matches:
        last = recall_matches[-1]
        result["recall"] = last["recall"]
        result["mrr"] = last["mrr"]
        result["recall_k"] = last["k"]
        result["mrr_k"] = last["k"]

    # Backward-compat aliases for older analysis code
    result.setdefault("recall_at10", result.get("recall_at10"))
    result.setdefault("mrr_at10", result.get("mrr_at10"))

    # Parse distribution blocks
    i = 0
    n = len(clean_lines)
    while i < n:
        line = clean_lines[i].strip()
        if line in SECTION_KEYS:
            base_key = SECTION_KEYS[line]

            # header line (Avg 50tiles ...)
            j = i + 1
            while j < n and not clean_lines[j].strip():
                j += 1
            if j >= n:
                break

            header_line = clean_lines[j].strip()
            # we don't strictly enforce header content, just assume 7 numbers next line

            # data line
            k = j + 1
            while k < n and not clean_lines[k].strip():
                k += 1
            if k >= n:
                break

            data_parts = clean_lines[k].split()
            if len(data_parts) >= 7:
                try:
                    values = [float(x) for x in data_parts[:7]]
                    for suffix, v in zip(DIST_SUFFIXES, values):
                        key = f"{base_key}_{suffix}"
                        result[key] = v
                except ValueError:
                    pass

            i = k + 1
        else:
            i += 1

    # Derived aliases for convenience
    # total latency aliases
    avg_total = result.get("total_latency_ms_avg")
    p999_total = result.get("total_latency_ms_p999")

    result["mean_latency_ms"] = avg_total
    result["latency_99p9_ms"] = p999_total

    # IO aliases
    mean_ios = result.get("disk_ios_avg")
    result["mean_ios"] = mean_ios

    if avg_total is not None and mean_ios and mean_ios != 0:
        result["mean_io_latency_us"] = avg_total * 1000.0 / mean_ios
    else:
        result["mean_io_latency_us"] = None

    return result


def collect_logs(log_root: Path) -> List[Path]:
    pattern = "*search_mc*_ir*.log"
    return list(log_root.rglob(pattern))


def process_single_log_root(log_root: Path, output_csv: Path) -> None:
    """Process a single log root directory and write results to CSV."""
    paths = collect_logs(log_root)
    if not paths:
        print(f"No search_mc*_ir*.log found under {log_root}")
        return

    print(f"Found {len(paths)} log files under {log_root}")

    rows: List[Dict[str, Any]] = []
    for p in sorted(paths):
        try:
            row = parse_log_file(p)
            rows.append(row)
        except Exception as e:
            print(f"[WARN] Failed to parse {p}: {e}")

    if not rows:
        print(f"No valid logs parsed from {log_root}")
        return

    # Collect all keys to form CSV header
    all_keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in all_keys:
                all_keys.append(k)

    # Make header stable & roughly ordered
    preferred_order = [
        "log_path",
        "log_filename",
        "log_ts",
        "log_time",
        "log_time_start",
        "log_time_end",
        "elapsed_wall_s",
        "dataset",
        "exp_name",
        "T",
        "quant_size_B",
        "quant_type",
        "maxcheck",
        "internal_result_num",
        "num_queries",
        "total_time_s",
        "qps",
        "recall_k",
        "recall",
        "mrr_k",
        "mrr",
        "recall_at10",
        "mrr_at10",
        "ex_elements_avg",
        "ex_elements_p50",
        "ex_elements_p90",
        "ex_elements_p95",
        "ex_elements_p99",
        "ex_elements_p999",
        "ex_elements_max",
        "head_latency_ms_avg",
        "head_latency_ms_p50",
        "head_latency_ms_p90",
        "head_latency_ms_p95",
        "head_latency_ms_p99",
        "head_latency_ms_p999",
        "head_latency_ms_max",
        "ex_latency_ms_avg",
        "ex_latency_ms_p50",
        "ex_latency_ms_p90",
        "ex_latency_ms_p95",
        "ex_latency_ms_p99",
        "ex_latency_ms_p999",
        "ex_latency_ms_max",
        "total_latency_ms_avg",
        "total_latency_ms_p50",
        "total_latency_ms_p90",
        "total_latency_ms_p95",
        "total_latency_ms_p99",
        "total_latency_ms_p999",
        "total_latency_ms_max",
        "disk_pages_avg",
        "disk_pages_p50",
        "disk_pages_p90",
        "disk_pages_p95",
        "disk_pages_p99",
        "disk_pages_p999",
        "disk_pages_max",
        "disk_ios_avg",
        "disk_ios_p50",
        "disk_ios_p90",
        "disk_ios_p95",
        "disk_ios_p99",
        "disk_ios_p999",
        "disk_ios_max",
        "mean_latency_ms",
        "latency_99p9_ms",
        "mean_ios",
        "mean_io_latency_us",
    ]

    # Append any remaining keys not in preferred_order
    header = preferred_order + [k for k in all_keys if k not in preferred_order]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {len(rows)} rows to {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse SPANN search logs into a CSV."
    )
    parser.add_argument(
        "--log-root",
        default="",
        help="Root directory (or glob pattern) containing <dataset>/logs/... search logs.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output CSV path (ignored if log-root is a glob pattern).",
    )
    args = parser.parse_args()

    if args.log_root == "":
        args.log_root = input("Log root directory or glob pattern (default: ./logs): ") or "./logs"
    
    # Check if log_root contains glob patterns
    has_glob = any(c in args.log_root for c in ['*', '?', '['])
    
    if has_glob:
        # Expand glob pattern to find all matching directories
        matches = sorted(glob.glob(args.log_root))
        log_dirs = [Path(m) for m in matches if Path(m).is_dir()]
        
        if not log_dirs:
            print(f"No directories match pattern: {args.log_root}")
            return
        
        print(f"Found {len(log_dirs)} directories matching pattern")
        
        # Process each directory and output to {dir_name}.csv in same parent
        for log_dir in log_dirs:
            output_csv = log_dir.parent / f"{log_dir.name}.csv"
            print(f"\n=== Processing {log_dir} ===")
            try:
                process_single_log_root(log_dir, output_csv)
            except Exception as e:
                print(f"[ERROR] Failed to process {log_dir}: {e}")
    else:
        # Single directory mode
        if args.output == "":
            args.output = input("Output CSV path (default: spann_search_results.csv): ") or "spann_search_results.csv"
        
        log_root = Path(args.log_root)
        output_csv = Path(args.output)
        process_single_log_root(log_root, output_csv)


if __name__ == "__main__":
    main()
