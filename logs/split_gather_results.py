#!/usr/bin/env python3
"""
Extract search performance logs from shard_vary_num experiments into CSV.

Parses search_release_search_split_knn_*.log files, extracts embedded internal log
references and table data, writes results to search_results.csv and errors to
search_results_errors.log.

Usage:
    python extract_shard_vary_num_logs.py [log_root] [output_csv] [error_log]

Example:
    python extract_shard_vary_num_logs.py /root/paper/DiskAnnPQ/logs/shard_vary_num
"""

import re
import csv
import pathlib
from typing import Optional, Dict, Any, List


_SEARCH_LOG_RE = re.compile(r"search_[^_]+shards_(\d+)_", re.IGNORECASE)
_TABLE_NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_TABLE_LINE_RE = re.compile(
    rf"^\s*(\d+)\s+({_TABLE_NUM})\s+({_TABLE_NUM})\s+({_TABLE_NUM})\s+({_TABLE_NUM})\s+({_TABLE_NUM})\s+({_TABLE_NUM})\s+({_TABLE_NUM})"
    rf"(?:\s+({_TABLE_NUM})\s+({_TABLE_NUM})\s+({_TABLE_NUM})\s+({_TABLE_NUM})\s+({_TABLE_NUM}))?\s*$"
)
_TABLE_HEADER_MARK = "L             QPS    Mean Latency    99.9 Latency        Mean IOs         CPU (s)   Peak Mem(MB)       Recall@10"
_TABLE_HEADER_RE = re.compile(r"^\s*L\s+QPS\s+Mean\s+Latency\s+99\.9\s+Latency\s+Mean\s+IOs", re.IGNORECASE)


def _mkdir(p: str) -> None:
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _parse_table_row(line: str) -> Optional[Dict[str, float]]:
    """Parse a single table data row from the log.

    Supports both formats:
      - Base: 8 columns (L + 7 metrics)
      - Extended: base + 5 columns (L_capacity L_init L_big L_small L_end)
    """
    m = _TABLE_LINE_RE.match(line)
    if not m:
        return None
    (
        lval,
        qps,
        mean_lat,
        p999,
        mean_ios,
        cpu_s,
        peak_mem,
        recall,
        l_capacity,
        l_init,
        l_big,
        l_small,
        l_end,
    ) = m.groups()
    row: Dict[str, float] = {
        "L": int(lval),
        "QPS": float(qps),
        "Mean Latency": float(mean_lat),
        "99.9 Latency": float(p999),
        "Mean IOs": float(mean_ios),
        "CPU (s)": float(cpu_s),
        "Peak Mem(MB)": float(peak_mem),
        "Recall@10": float(recall),
    }

    # Optional extended columns
    if l_capacity is not None:
        # These are conceptually ints, but parse as float to keep consistent dict typing.
        row["L_capacity"] = float(l_capacity)
        row["L_init"] = float(l_init)
        row["L_big"] = float(l_big)
        row["L_small"] = float(l_small)
        row["L_end"] = float(l_end)

    return row


def _parse_single_search_log(
    log_path: pathlib.Path,
    experiment: str,
    base_dir: pathlib.Path,
    errors: List[str],
) -> List[Dict[str, Any]]:
    """Parse a single search_release_search_split_knn_*.log file."""
    lines = _read_text(str(log_path)).splitlines()
    rows: List[Dict[str, Any]] = []
    current_split: Optional[int] = None
    current_search_dir: Optional[str] = None
    current_search_file: Optional[str] = None

    for idx, line in enumerate(lines):
        # Detect embedded search log file path
        path_match = _SEARCH_LOG_RE.search(line)
        if path_match:
            current_split = int(path_match.group(1))
            path_str = line.strip()
            spath = pathlib.Path(path_str)
            current_search_dir = str(spath.parent)
            current_search_file = spath.name
            continue

        # Skip table header lines (base or extended)
        if _TABLE_HEADER_MARK in line or _TABLE_HEADER_RE.search(line):
            continue

        # Try to parse data row
        parsed = _parse_table_row(line)
        if not parsed:
            continue

        if (
            current_split is None
            or current_search_dir is None
            or current_search_file is None
        ):
            errors.append(
                f"{log_path}:L{idx + 1}: data row without current search log context -> {line.strip()}"
            )
            continue

        rows.append(
            {
                "experiment": experiment,
                "top_log": str(log_path.relative_to(base_dir)),
                "search_log_dir": current_search_dir,
                "search_log_file": current_search_file,
                "search_split_num": current_split,
                **parsed,
            }
        )

    if not rows:
        errors.append(f"{log_path}: no table rows parsed")

    return rows


def extract_shard_vary_num_search_logs(
    log_root: str,
    output_csv: Optional[str] = None,
    error_log: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Extract all search_split search results under shard_vary_num experiments into one CSV.

    Args:
        log_root: Root directory containing experiment folders (e.g.,
                  /root/paper/DiskAnnPQ/logs/shard_vary_num)
        output_csv: Output CSV file path (default: log_root/search_results.csv)
        error_log: Error log file path (default: log_root/search_results_errors.log)

    Returns:
        dict with keys: rows (count), errors (count), csv (path), error_log (path)
    """
    base_dir = pathlib.Path(log_root)
    out_csv_path = (
        pathlib.Path(output_csv)
        if output_csv
        else base_dir / "search_results.csv"
    )
    err_log_path = (
        pathlib.Path(error_log)
        if error_log
        else base_dir / "search_results_errors.log"
    )

    all_rows: List[Dict[str, Any]] = []
    errors: List[str] = []

    if not base_dir.is_dir():
        raise FileNotFoundError(f"log_root not found: {base_dir}")

    # Iterate through experiment directories
    exp_dirs = sorted([p for p in base_dir.iterdir() if p.is_dir()])
    for exp_dir in exp_dirs:
        search_dir = exp_dir / "search"
        if not search_dir.is_dir():
            continue
        # Find all search_release_search_split_knn_*.log files
        log_files = sorted(search_dir.glob("search_release_search_split_knn_*.log"))
        for log_path in log_files:
            try:
                rows = _parse_single_search_log(
                    log_path, exp_dir.name, base_dir, errors
                )
                all_rows.extend(rows)
            except Exception as e:  # keep going per requirement
                errors.append(f"{log_path}: exception {type(e).__name__}: {e}")

    # Write CSV
    fieldnames = [
        "experiment",
        "top_log",
        "search_log_dir",
        "search_log_file",
        "search_split_num",
        "L",
        "QPS",
        "Mean Latency",
        "99.9 Latency",
        "Mean IOs",
        "CPU (s)",
        "Peak Mem(MB)",
        "Recall@10",
        "L_capacity",
        "L_init",
        "L_big",
        "L_small",
        "L_end",
    ]

    _mkdir(str(out_csv_path.parent))
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # Write error log
    _mkdir(str(err_log_path.parent))
    with open(err_log_path, "w", encoding="utf-8") as f:
        if errors:
            f.write("\n".join(errors) + "\n")
        else:
            f.write("no errors\n")

    return {
        "rows": len(all_rows),
        "errors": len(errors),
        "csv": str(out_csv_path),
        "error_log": str(err_log_path),
    }


if __name__ == "__main__":
    import sys

    log_root = sys.argv[1] if len(sys.argv) > 1 else input("Enter log root directory: ").strip()
    output_csv = sys.argv[2] if len(sys.argv) > 2 else None
    error_log = sys.argv[3] if len(sys.argv) > 3 else None

    print(f"Extracting logs from: {log_root}")
    result = extract_shard_vary_num_search_logs(log_root, output_csv, error_log)
    print(f"✓ Extracted {result['rows']} data rows, {result['errors']} errors")
    print(f"✓ CSV: {result['csv']}")
    print(f"✓ Error log: {result['error_log']}")
