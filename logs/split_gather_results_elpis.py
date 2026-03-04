#!/usr/bin/env python3
import os
import re
import csv
import argparse
from typing import List, Dict, Any, Optional, Tuple

NUM_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
SUBLOG_RE = re.compile(r"^\s*(/.*search_.*\.log)\s*$")

def is_header(line: str) -> bool:
    return ("QPS" in line) and ("Recall@10" in line)

def extract_split_k(sublog_path: str) -> Optional[int]:
    base = os.path.basename(sublog_path)
    m = re.search(r"_([0-9]+)_SQ", base)
    return int(m.group(1)) if m else None

def extract_elpis_ls_bs(text: str) -> Tuple[Optional[int], Optional[int]]:
    """
    从字符串中抽 ELPIS leaf_size / buffer_size：
      ...elpis-ls100000_bs80...
    返回 (ls, bs)；若无匹配则 (None, None)。
    """
    m = re.search(r"elpis-ls(\d+)_bs(\d+)", text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None

def parse_metrics_from_log(path: str) -> List[Dict[str, Any]]:
    """
    从一个聚合 search_*.log 中解析出所有块的指标行。
    每个块：sublog_path + 表头 + 数据行（可能多行）
    """
    rows: List[Dict[str, Any]] = []

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except OSError as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return rows

    i = 0
    current_sublog = None

    while i < len(lines):
        line = lines[i].strip()

        m_sub = SUBLOG_RE.match(line)
        if m_sub:
            current_sublog = m_sub.group(1)
            i += 1

            # 找表头
            header_idx = None
            while i < len(lines):
                if is_header(lines[i]):
                    header_idx = i
                    break
                if SUBLOG_RE.match(lines[i].strip()):
                    header_idx = None
                    break
                i += 1

            if header_idx is None:
                continue

            i = header_idx + 1

            # 读数据行直到遇到下一个 sublog 或 EOF
            while i < len(lines):
                s = lines[i].strip()
                if not s:
                    i += 1
                    continue
                if SUBLOG_RE.match(s):
                    break

                nums = NUM_RE.findall(s)
                if len(nums) >= 8:
                    try:
                        L = int(float(nums[0]))
                        QPS = float(nums[1])
                        mean_latency = float(nums[2])
                        latency_999 = float(nums[3])
                        mean_ios = float(nums[4])
                        cpu_s = float(nums[5])
                        peak_mem = float(nums[6])
                        recall10 = float(nums[7])
                    except ValueError:
                        i += 1
                        continue

                    rows.append({
                        "sublog_path": current_sublog,
                        "split_k": extract_split_k(current_sublog) if current_sublog else None,
                        "L": L,
                        "QPS": QPS,
                        "MeanLatency": mean_latency,
                        "Latency999": latency_999,
                        "MeanIOs": mean_ios,
                        "CPU_s": cpu_s,
                        "PeakMemMB": peak_mem,
                        "Recall10": recall10,
                    })
                i += 1

            continue

        i += 1

    return rows


def collect_search_logs(base_dir: str, output_csv: str) -> None:
    fieldnames = [
        "agg_filename",
        "agg_name_core",
        "agg_path",
        "sublog_path",
        "split_k",
        "elpis_ls",
        "elpis_bs",
        "L",
        "QPS",
        "MeanLatency",
        "Latency999",
        "MeanIOs",
        "CPU_s",
        "PeakMemMB",
        "Recall10",
    ]

    count_files = 0
    count_rows = 0

    with open(output_csv, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for root, _dirs, files in os.walk(base_dir):
            for fname in files:
                if not (fname.startswith("search_") and fname.endswith(".log")):
                    continue

                full_path = os.path.join(root, fname)
                count_files += 1

                core = fname[len("search_"):-len(".log")]

                # 先从 agg core 抽 ls/bs（优先）
                ls, bs = extract_elpis_ls_bs(core)

                rows = parse_metrics_from_log(full_path)
                if not rows:
                    continue

                for row in rows:
                    # 如果 agg core 没抽到，再试 sublog_path
                    if ls is None or bs is None:
                        sp = row.get("sublog_path") or ""
                        ls2, bs2 = extract_elpis_ls_bs(sp)
                        ls = ls if ls is not None else ls2
                        bs = bs if bs is not None else bs2

                    out_row = {
                        "agg_filename": fname,
                        "agg_name_core": core,
                        "agg_path": full_path,
                        "elpis_ls": ls,
                        "elpis_bs": bs,
                        **row,
                    }
                    writer.writerow(out_row)
                    count_rows += 1

    print(f"[INFO] Parsed {count_rows} rows from {count_files} aggregated search_*.log files.")
    print(f"[INFO] CSV written to: {output_csv}")


# def main():
#     parser = argparse.ArgumentParser(
#         description="Collect Starling aggregated search_*.log metrics into a CSV (ELPIS ls/bs supported)."
#     )
#     parser.add_argument(
#         "base_dir",
#         help="Directory containing aggregated search_*.log "
#              "(e.g. /mnt/disk1/paper/DiskAnnPQ/starling/indices)",
#     )
#     parser.add_argument(
#         "-o", "--output",
#         default="search_metrics.csv",
#         help="Output CSV path (default: search_metrics.csv)",
#     )
#     args = parser.parse_args()

#     collect_search_logs(args.base_dir, args.output)


# if __name__ == "__main__":
#     main()


def main():
    parser = argparse.ArgumentParser(
        description="Collect Starling search_*.log metrics into a CSV."
    )
    parser.add_argument(
        "--base_dir",
        default="",
        help="Directory containing search_*.log files "
             "(e.g. /mnt/disk1/paper/DiskAnnPQ/starling/indices)",
    )
    parser.add_argument(
        "-o", "--output",
        default="",
        help="Output CSV path (default: search_metrics.csv)",
    )
    args = parser.parse_args()

    if args.base_dir == "":
        args.base_dir = input("Enter base directory containing search_*.log files (default: current directory): ").strip() or "."

    if args.output == "":
        args.output = input("Enter output CSV path (default: search_metrics.csv): ").strip() or "search_metrics.csv"

    collect_search_logs(args.base_dir, args.output)


if __name__ == "__main__":
    main()
