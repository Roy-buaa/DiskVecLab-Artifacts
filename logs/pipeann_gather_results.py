#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os
import re
from datetime import datetime
from typing import Optional

# 例：
# - search_bw1_meml0_pipe2_20251217_173813.log
# - search_bw1_meml0_20251217_173813.log  (pipe 可选，默认按 pipe=2 处理)
# - search_bw1_meml0_pipe2_T32_20260115_171941.log (新增可选 T)
FNAME_RE = re.compile(
    r".*search_bw(?P<bw>\d+)_meml(?P<meml>\d+)"
    r"(?:_pipe(?P<pipe>\d+))?"
    r"(?:_T(?P<T>\d+))?"
    r"_(?P<date>\d{8})_(?P<time>\d{6})\.log$"
)

# 尝试从 log 内容里捞“运行时间”的常见写法（匹配到就用）
TIME_PATTERNS = [
    re.compile(r"\b(total|elapsed)\s*time\b\s*[:=]\s*(?P<val>[\d.]+)\s*(?P<unit>ms|s|sec|secs|seconds)\b", re.I),
    re.compile(r"\bqps\b\s*[:=]\s*(?P<val>[\d.]+)\b", re.I),
    re.compile(r"\bavg\s*(latency|time)\b\s*[:=]\s*(?P<val>[\d.]+)\s*(?P<unit>ms|s)\b", re.I),
]

def extract_runtime_hint(log_path: str) -> Optional[str]:
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            # 只扫前 2000 行，足够快；你也可以改大
            for i, line in enumerate(f):
                if i > 2000:
                    break
                for pat in TIME_PATTERNS:
                    m = pat.search(line)
                    if m:
                        gd = m.groupdict()
                        # 有 unit 的优先展示 unit
                        if "unit" in gd and gd.get("unit"):
                            return f"{gd.get('val')}{gd.get('unit')}"
                        return gd.get("val")
    except Exception:
        pass
    return None

def parse_meta_from_name(path: str):
    base = os.path.basename(path)
    m = FNAME_RE.match(base)
    if not m:
        return None

    bw = int(m.group("bw"))
    meml = int(m.group("meml"))
    pipe_str = m.groupdict().get("pipe")
    pipe = int(pipe_str) if pipe_str is not None else 2

    t_str = m.groupdict().get("T")
    t = int(t_str) if t_str is not None else None
    dt = datetime.strptime(m.group("date") + m.group("time"), "%Y%m%d%H%M%S")
    return bw, meml, pipe, t, dt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="", help="log 目录，比如 /root/paper/DiskAnnPQ/logs/main/pipeann_deep100m_search")
    ap.add_argument("--glob", default="search_bw*_meml*_*.log", help="文件匹配模式（相对 --dir）")
    ap.add_argument("--out", default="", help="输出拼接后的文件路径")
    ap.add_argument("--no-runtime-scan", action="store_true", help="不从内容扫描时间，只用文件名时间戳")
    args = ap.parse_args()

    if args.dir == "":
        args.dir = input("请输入 log 目录路径: ").strip()
    if args.out == "":
        args.out = input("请输入输出文件路径: ").strip()

    pattern = os.path.join(args.dir, args.glob)
    files = sorted(glob.glob(pattern))

    # 只保留能解析出 bw/meml/时间戳 的文件，并按时间戳排序
    metas = []
    for fp in files:
        meta = parse_meta_from_name(fp)
        if meta is not None:
            bw, meml, pipe, t, dt = meta
            metas.append((dt, bw, meml, pipe, t, fp))

    metas.sort(key=lambda x: x[0])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.out, "w", encoding="utf-8", errors="replace") as out:
        for dt, bw, meml, pipe, t, fp in metas:
            is_mem_nav = (meml != 0)
            runtime = None if args.no_runtime_scan else extract_runtime_hint(fp)

            out.write("=" * 90 + "\n")
            out.write(f"FILE: {fp}\n")
            if pipe == 0: # DiskANN
                method = "Pipe-DiskANN"
            elif pipe == 2: # PipeANN
                method = "PipeANN"
            else:
                raise ValueError(f"Unknown pipe value: {pipe} in file {fp}")
    
            out.write(f"Running {method} (Mem Nav)\n" if is_mem_nav else f"Running {method}\n")
            header = f"bw={bw}, meml={meml}, pipe={pipe}"
            if t is not None:
                header += f", T={t}"
            header += f", ts={dt.strftime('%Y-%m-%d %H:%M:%S')}"
            out.write(header)
            if runtime:
                out.write(f", time={runtime}")
            out.write("\n")
            out.write("=" * 90 + "\n\n")

            # 追加原始 log（流式）
            with open(fp, "r", encoding="utf-8", errors="replace") as fin:
                for line in fin:
                    out.write(line)

            if not out.tell() or not str(out).endswith("\n"):
                out.write("\n")
            out.write("\n")

    print(f"[OK] merged {len(metas)} logs -> {args.out}")

if __name__ == "__main__":
    main()
