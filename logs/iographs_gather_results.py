#!/usr/bin/env python3
import re
import csv
from pathlib import Path
from typing import Optional, Tuple

# 统一内部列
HEADER_COLS = [
    "L", "BW", "QPS", "Mean_Ltc", "Ltc_999",
    "Graph_IO", "Emb_IO", "Ext_Cmp", "PQ_Cmp",
    "Pre_T", "Disp_T", "Read_T", "Page_T",
    "Cache_T", "DiskN_T", "Post_T",
    "Mem_MB", "Recall@10",
]


def parse_log_to_rows(text: str):
    lines = text.splitlines()
    rows = []
    current_setting = None
    current_logfile = None
    current_T: Optional[int] = None

    # ---------- thread-count extraction ----------
    # pipeann: "bw=32, meml=0, pipe=0, T=1, ts=..."
    # margo/gorgeous: contains "..._T1_..." inside the logged file path
    thread_eq_pattern = re.compile(r"\bT\s*=\s*(\d+)\b")
    thread_in_path_pattern = re.compile(r"(?:^|[/_])T(\d+)(?:[_\.]|$)")

    # pipeann raw logs also include: "Search parameters: #threads: 1,  beamwidth: 8"
    pipeann_threads_pattern = re.compile(r"#threads:\s*(\d+)")

    # pipeann merged logs can include: "bw=8, meml=0, pipe=0, T=1, ts=2026-01-16 16:07:06"
    pipeann_run_line_pattern = re.compile(
        r"\bbw\s*=\s*(\d+)\s*,\s*meml\s*=\s*(\d+)\s*,\s*pipe\s*=\s*(\d+)\s*,\s*T\s*=\s*(\d+)\s*,\s*ts\s*=\s*(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})"
    )

    def _maybe_rewrite_pipeann_logfile_from_run_line(run_line: str) -> None:
        """If current_logfile is a PipeANN FILE: path, rewrite it to match bw/meml/pipe/T/ts.

        Some merged logs may keep an old FILE: path while varying T; we use the explicit
        run line to reconstruct the expected filename (including timestamp).
        """
        nonlocal current_logfile
        if not current_logfile:
            return
        m = pipeann_run_line_pattern.search(run_line)
        if not m:
            return
        bw, meml, pipe, t_str, date_str, time_str = m.groups()
        ts_compact = date_str.replace("-", "") + "_" + time_str.replace(":", "")

        prefix = ""
        path_str = current_logfile.strip()
        if path_str.startswith("FILE:"):
            prefix = "FILE: "
            path_str = path_str.split(":", 1)[1].strip()

        # only rewrite if this looks like a pipeann per-run log path
        if "/pipeann/" not in path_str or not path_str.endswith(".log"):
            return

        base_dir = Path(path_str).parent
        new_name = f"search_bw{bw}_meml{meml}_pipe{pipe}_T{t_str}_{ts_compact}.log"
        new_path = base_dir / new_name
        if new_path.exists():
            current_logfile = prefix + str(new_path)

    def maybe_update_thread(source: str) -> None:
        nonlocal current_T
        m = thread_eq_pattern.search(source)
        if m:
            current_T = int(m.group(1))
            return
        m = pipeann_threads_pattern.search(source)
        if m:
            current_T = int(m.group(1))
            return
        m = thread_in_path_pattern.search(source)
        if m:
            current_T = int(m.group(1))
            return

    num_pattern = re.compile(r'[-+]?\d+\.\d+|[-+]?\d+')
    line_starts_with_number = re.compile(r"^\s*\d")
    strict_num_pattern = re.compile(r"[-+]?\d+(?:\.\d+)?\Z")

    def split_concatenated_number_token(tok: str) -> list[str]:
        """Split tokens like '1010.78129377.96' -> ['1010.78', '129377.96'].

        这种 token 往往来自日志的 fixed-width 输出：当列宽不足时空格被吃掉，导致两个数字粘连。
        我们尝试在 token 内部找一个切分点，使得两段都能被解析成合法数字。
        """
        if tok.count(".") < 2:
            return [tok]
        if not re.fullmatch(r"[-+]?[0-9.]+", tok):
            return [tok]

        best: Optional[Tuple[int, str, str]] = None

        def score_num(s: str) -> int:
            if "." not in s:
                return 0
            dec_len = len(s.split(".", 1)[1])
            if dec_len == 2:
                return 3
            if dec_len == 3:
                return 2
            if dec_len == 1:
                return 1
            if dec_len > 6:
                return -2
            return 0

        for k in range(1, len(tok)):
            left, right = tok[:k], tok[k:]
            if strict_num_pattern.fullmatch(left) and strict_num_pattern.fullmatch(right):
                score = score_num(left) + score_num(right)
                if best is None or score > best[0]:
                    best = (score, left, right)

        if best is not None:
            return [best[1], best[2]]
        return [tok]

    def extract_numbers(line: str, expected_count: Optional[int] = None) -> list[str]:
        """Extract numbers from a line, with a fallback fix for concatenated tokens."""
        nums = num_pattern.findall(line)
        if expected_count is None or len(nums) == expected_count:
            return nums

        # fallback: split tokens that look like two numbers glued together (multiple dots)
        rebuilt: list[str] = []
        for tok in line.split():
            if tok.count(".") >= 2:
                rebuilt.extend(split_concatenated_number_token(tok))
            else:
                rebuilt.append(tok)
        fixed_line = " ".join(rebuilt)
        nums2 = num_pattern.findall(fixed_line)
        if expected_count is None or len(nums2) == expected_count:
            return nums2
        return nums

    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # ---------- PipeANN 的线程标识行 ----------
        # e.g. "bw=32, meml=0, pipe=0, T=1, ts=2026-01-15 16:49:44"
        if ("bw=" in stripped) and ("meml=" in stripped) and ("pipe=" in stripped) and ("T=" in stripped):
            maybe_update_thread(stripped)
            _maybe_rewrite_pipeann_logfile_from_run_line(stripped)
            i += 1
            continue

        # ---------- 识别 setting ----------
        if stripped.startswith("Running "):
            rest = stripped[len("Running "):].strip()
            # 有些汇总 log 会打印 "Running Command:" 作为占位标题，下一行才是真正的命令。
            # 这不是算法/方法名，不能写进 setting（否则 CSV 的 _quant 会变成 "Command:"）。
            first = rest.split()[0].rstrip(":").lower() if rest else ""
            if first == "command":
                i += 1
                continue
            if rest.upper() == "MARGO":
                current_setting = "MARGO"
            else:
                current_setting = rest  # 包含 "DiskANN Search", "PipeANN (Mem Nav)", "PipeANN" 等
            i += 1
            continue

        # ---------- 识别 log 文件路径 ----------
        # 兼容：MARGO 日志里常见 'Searching... log file: /path/to/xxx.log'
        if stripped.startswith("Searching... log file:"):
            current_logfile = stripped.split("Searching... log file:", 1)[1].strip()
            maybe_update_thread(current_logfile)
            i += 1
            continue

        if stripped.endswith(".log") and not stripped.startswith("Searching..."):
            current_logfile = stripped
            maybe_update_thread(current_logfile)
            i += 1
            continue

        # ---------- MARGO (新输出) 格式 ----------
        # 典型表头：
        #  L  QPS  Mean Latency  99.9 Latency  Mean IOs  CPU (s)  Peak Mem(MB)  Recall@10
        if (
            "Mean Latency" in line
            and "99.9" in line
            and "Peak Mem" in line
            and "Recall@10" in line
            and "QPS" in line
            and "Beamwidth" not in line
        ):
            i += 1
            while i < n:
                data_line = lines[i]
                stripped2 = data_line.strip()

                if stripped2.startswith("Running "):
                    break

                # 表格块中可能夹杂 log 路径行
                if stripped2.startswith("Searching... log file:"):
                    current_logfile = stripped2.split("Searching... log file:", 1)[1].strip()
                    maybe_update_thread(current_logfile)
                    i += 1
                    continue
                if stripped2.endswith(".log") and not stripped2.startswith("Searching..."):
                    current_logfile = stripped2
                    maybe_update_thread(current_logfile)
                    i += 1
                    continue

                if not line_starts_with_number.match(data_line):
                    i += 1
                    continue

                nums = extract_numbers(data_line, expected_count=8)
                if len(nums) == 0:
                    i += 1
                    continue
                if len(nums) != 8:
                    i += 1
                    continue

                L_val = int(nums[0])
                qps = float(nums[1])
                mean_lat = float(nums[2])
                lat_999 = float(nums[3])
                mean_ios = float(nums[4])
                cpu_s = float(nums[5])
                peak_mem = float(nums[6])
                recall = float(nums[7])

                row = {
                    "setting": current_setting,
                    "log_file": current_logfile,
                    "T": current_T,
                    "L": L_val,
                    "BW": None,
                    "QPS": qps,
                    "Mean_Ltc": mean_lat,
                    "Ltc_999": lat_999,
                    "Graph_IO": mean_ios,
                    "Emb_IO": None,
                    "Ext_Cmp": None,
                    "PQ_Cmp": None,
                    "Pre_T": None,
                    "Disp_T": None,
                    "Read_T": None,
                    "Page_T": None,
                    "Cache_T": None,
                    "DiskN_T": None,
                    "Post_T": None,
                    "Mem_MB": peak_mem,
                    "Recall@10": recall,
                    "CPU_s": cpu_s,
                }
                rows.append(row)
                i += 1
            continue

        # ---------- 旧格式：DiskANN / Starling / Gorgeous ----------
        if ("L  BW" in line or "L   BW" in line) and "Recall@10" in line:
            i += 1
            while i < n:
                data_line = lines[i]
                stripped2 = data_line.strip()

                # 遇到新一轮 Running，说明下一个 block 开始了
                if stripped2.startswith("Running "):
                    break

                # 有些日志会在表头和数据之间/中间插入 log 文件路径，这行包含大量数字（如 CUT4096）
                # 如果不处理，会被当成数据行导致 Recall@10=4096 之类的错误。
                if stripped2.endswith(".log") and not stripped2.startswith("Searching..."):
                    current_logfile = stripped2
                    maybe_update_thread(current_logfile)
                    i += 1
                    continue

                # 只接受真正的数据行：通常是以 L 开头的纯数字行（允许前导空格）。
                # 这样可以避免把“路径/配置行”的数字抓进来。
                if not line_starts_with_number.match(data_line):
                    i += 1
                    continue

                nums = extract_numbers(data_line, expected_count=18)

                # 空行 / 分隔线直接跳过
                if len(nums) == 0:
                    i += 1
                    continue

                # 数据行应该有 18 个数字
                if len(nums) != 18:
                    i += 1
                    continue

                # Recall@10 合理范围保护：有的日志 Recall 用百分比（0~100），也可能是 0~1。
                # 但不可能是 4096 这种量级。
                try:
                    recall_candidate = float(nums[-1])
                except ValueError:
                    i += 1
                    continue
                if recall_candidate > 100.0:
                    i += 1
                    continue

                L_val = int(nums[0])
                BW_val = int(nums[1])
                rest_nums = [float(x) for x in nums[2:]]
                values = [L_val, BW_val] + rest_nums

                row = {"setting": current_setting, "log_file": current_logfile}
                row["T"] = current_T
                for col_name, val in zip(HEADER_COLS, values):
                    row[col_name] = val
                rows.append(row)
                i += 1
            continue

        # ---------- MARGO 格式 ----------
        # 兼容旧解析分支：如果某些 MARGO 输出确实包含 Beamwidth 列，仍然走这里
        if "Beamwidth" in line and "QPS" in line and "Recall@10" in line:
            i += 1
            while i < n:
                data_line = lines[i]
                stripped2 = data_line.strip()

                if stripped2.startswith("Running "):
                    break

                nums = extract_numbers(data_line, expected_count=11)
                if len(nums) == 0:
                    i += 1
                    continue
                if len(nums) != 11:
                    i += 1
                    continue

                L_val = int(nums[0])
                beamwidth = int(nums[1])
                qps = float(nums[2])
                mean_lat = float(nums[3])
                lat_999 = float(nums[4])
                mean_ios = float(nums[5])
                cpu_s = float(nums[6])
                b4_in_mem = float(nums[7])
                after_cache = float(nums[8])
                peak_mem = float(nums[9])
                recall = float(nums[10])

                row = {
                    "setting": current_setting,
                    "log_file": current_logfile,
                    "T": current_T,
                    "L": L_val,
                    "BW": beamwidth,
                    "QPS": qps,
                    "Mean_Ltc": mean_lat,
                    "Ltc_999": lat_999,
                    "Graph_IO": mean_ios,
                    "Emb_IO": None,
                    "Ext_Cmp": None,
                    "PQ_Cmp": None,
                    "Pre_T": None,
                    "Disp_T": None,
                    "Read_T": None,
                    "Page_T": None,
                    "Cache_T": None,
                    "DiskN_T": None,
                    "Post_T": None,
                    "Mem_MB": peak_mem,
                    "Recall@10": recall,
                    # 额外信息保留
                    "CPU_s": cpu_s,
                    "B4_Load_InMem": b4_in_mem,
                    "After_Load_Cache": after_cache,
                }
                rows.append(row)
                i += 1
            continue

        # ---------- PipeANN 格式 ----------
        if "I/O Width" in line and "QPS" in line and "Recall@10" in line:
            i += 1
            while i < n:
                data_line = lines[i]
                stripped2 = data_line.strip()

                if stripped2.startswith("Running "):
                    break

                # 只接受真正的数值数据行，避免把 FILE 路径、START/END 时间戳等误解析成结果
                if not line_starts_with_number.match(data_line):
                    i += 1
                    continue

                nums = extract_numbers(data_line, expected_count=8)
                if len(nums) == 0:
                    i += 1
                    continue
                if len(nums) != 8:
                    i += 1
                    continue

                # 合法性保护：Recall@10 只可能在 [0,100]（或极少 [0,1]）范围内
                # 避免把文件名中的日期时间（如 20260203、215930）写入结果列
                try:
                    recall_candidate = float(nums[-1])
                except ValueError:
                    i += 1
                    continue
                if recall_candidate > 100.0:
                    i += 1
                    continue

                L_val = int(nums[0])
                width = int(nums[1])
                qps = float(nums[2])
                mean_lat = float(nums[3])
                lat_999 = float(nums[4])
                mean_hops = float(nums[5])
                mean_ios = float(nums[6])
                recall = float(nums[7])

                row = {
                    "setting": current_setting,
                    "log_file": current_logfile,
                    "T": current_T,
                    "L": L_val,
                    "BW": width,
                    "QPS": qps,
                    "Mean_Ltc": mean_lat,
                    "Ltc_999": lat_999,
                    "Graph_IO": mean_ios,
                    "Emb_IO": None,
                    "Ext_Cmp": None,
                    "PQ_Cmp": None,
                    "Pre_T": None,
                    "Disp_T": None,
                    "Read_T": None,
                    "Page_T": None,
                    "Cache_T": None,
                    "DiskN_T": None,
                    "Post_T": None,
                    "Mem_MB": None,
                    "Recall@10": recall,
                    "Mean_Hops": mean_hops,
                }
                rows.append(row)
                i += 1
            continue

        i += 1

    return rows


def main():
    import argparse
    import csv
    import os
    import glob
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Parse DiskANN/Gorgeous/Starling/MARGO logs into a unified CSV."
    )

    # 改成 nargs="+"：允许传入多个路径/模式
    parser.add_argument(
        "--log_path",
        nargs="+",
        help="Input log file path OR glob pattern(s) OR directory. "
             "Examples: /a/b/*.log  /a/**/x_*.log  /a/b/logs_dir",
        default=[],
    )
    parser.add_argument("--csv_path", help="output CSV file path", default="")
    args = parser.parse_args()

    if (not args.log_path) or args.csv_path == "":
        # 兼容交互输入：允许用户一次输入一个pattern（里面可含*）
        lp = input("Enter input log file path / glob / directory: ").strip()
        cp = input("Enter output CSV file path: ").strip()
        args.log_path = [lp] if lp else []
        args.csv_path = cp

    if not args.log_path or args.csv_path == "":
        raise SystemExit("Both --log_path and --csv_path are required.")

    csv_path = Path(args.csv_path)

    def expand_one(p: str) -> list[Path]:
        """Expand a single input: file / dir / glob pattern -> list of files."""
        p = os.path.expanduser(os.path.expandvars(p))

        # 1) 如果是目录：默认取目录下 *.log
        pp = Path(p)
        if pp.exists() and pp.is_dir():
            files = sorted(pp.glob("*.log"))
            return [f for f in files if f.is_file()]

        # 2) 如果是普通文件路径
        if pp.exists() and pp.is_file():
            return [pp]

        # 3) 否则按 glob 处理（支持 ** 递归）
        matches = glob.glob(p, recursive=True)
        files = [Path(m) for m in matches if Path(m).is_file()]
        files.sort()
        return files

    # 展开所有输入（支持多个pattern）
    all_files: list[Path] = []
    for item in args.log_path:
        all_files.extend(expand_one(item))

    # 去重且保持顺序
    seen = set()
    log_files: list[Path] = []
    for f in all_files:
        fp = str(f.resolve())
        if fp not in seen:
            seen.add(fp)
            log_files.append(f)

    if not log_files:
        raise SystemExit(f"No log files matched: {args.log_path}")

    # 维持你目前画图脚本用的列名
    fieldnames = [
        "_quant",
        "log_file",
        "L",
        "_W",
        "T",
        "QPS",
        "Mean_Ltc",
        "Ltc_999",
        "Graph_IO",
        "Emb_IO",
        "Ext_Cmp",
        "PQ_Cmp",
        "Pre_T",
        "Disp_T",
        "Read_T",
        "Page_T",
        "Cache_T",
        "DiskN_T",
        "Post_T",
        "Mem_MB",
        "Recall@10",
    ]

    total_rows = 0
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for log_path in log_files:
            # 大 log：用容错编码，避免某些奇怪字符直接崩
            text = log_path.read_text(encoding="utf-8", errors="replace")

            rows = parse_log_to_rows(text)  # 你现有逻辑不动
            if not rows:
                continue

            for r in rows:
                out_row = {
                    "_quant": r.get("setting"),
                    "log_file": r.get("log_file") or str(log_path),
                    "L": r.get("L"),
                    "_W": r.get("BW"),
                    "T": r.get("T"),
                    "QPS": r.get("QPS"),
                    "Mean_Ltc": r.get("Mean_Ltc"),
                    "Ltc_999": r.get("Ltc_999"),
                    "Graph_IO": r.get("Graph_IO"),
                    "Emb_IO": r.get("Emb_IO"),
                    "Ext_Cmp": r.get("Ext_Cmp"),
                    "PQ_Cmp": r.get("PQ_Cmp"),
                    "Pre_T": r.get("Pre_T"),
                    "Disp_T": r.get("Disp_T"),
                    "Read_T": r.get("Read_T"),
                    "Page_T": r.get("Page_T"),
                    "Cache_T": r.get("Cache_T"),
                    "DiskN_T": r.get("DiskN_T"),
                    "Post_T": r.get("Post_T"),
                    "Mem_MB": r.get("Mem_MB"),
                    "Recall@10": r.get("Recall@10"),
                }
                writer.writerow(out_row)
                total_rows += 1

    print(f"Matched {len(log_files)} log files, parsed {total_rows} rows into {csv_path}")


if __name__ == "__main__":
    main()
