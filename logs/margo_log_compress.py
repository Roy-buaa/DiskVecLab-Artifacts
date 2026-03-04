#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from typing import Optional, TextIO

PAIR_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s*$")

def flush_run(
    out: TextIO,
    start_n: int,
    end_n: int,
    v: int,
    count: int,
    first_line: str,
    last_line: str,
    min_run: int,
    style: str,
):
    """
    style:
      - "head-tail": 输出首行 + 省略行 + 尾行
      - "summary":   只输出一行 summary
    """
    if count < min_run:
        # 不压缩：直接输出全部行（这里无法重放逐行内容），所以我们只在“不会压缩”的情况下不走 run 聚合；
        # 因此 flush_run 只会在 run 已经聚合的情况下被调用，count < min_run 说明你把 min_run 设得太大。
        # 为了保持正确性，我们至少输出首尾，避免丢失信息。
        out.write(first_line)
        if count >= 2:
            out.write(last_line)
        return

    if style == "summary":
        out.write(f"{start_n}..{end_n} {v}    [compressed: {count} lines]\n")
        return

    # 默认 head-tail
    out.write(first_line)
    omitted = max(0, count - 2)
    if omitted > 0:
        out.write(f"... omitted {omitted} lines: {start_n+1}..{end_n-1} {v} ...\n")
    if count >= 2:
        out.write(last_line)

def shrink_file(in_path: str, out_path: str, min_run: int, style: str):
    with open(in_path, "r", encoding="utf-8", errors="replace") as fin, \
         open(out_path, "w", encoding="utf-8", errors="replace") as out:

        # 当前 run 状态
        in_run = False
        run_start_n: Optional[int] = None
        run_end_n: Optional[int] = None
        run_v: Optional[int] = None
        run_count = 0
        run_first_line = ""
        run_last_line = ""

        def end_current_run_if_needed():
            nonlocal in_run, run_start_n, run_end_n, run_v, run_count, run_first_line, run_last_line
            if not in_run:
                return
            flush_run(
                out,
                start_n=run_start_n,  # type: ignore
                end_n=run_end_n,      # type: ignore
                v=run_v,              # type: ignore
                count=run_count,
                first_line=run_first_line,
                last_line=run_last_line,
                min_run=min_run,
                style=style,
            )
            in_run = False
            run_start_n = run_end_n = run_v = None
            run_count = 0
            run_first_line = run_last_line = ""

        for line in fin:
            m = PAIR_RE.match(line)
            if not m:
                # 非目标行：先把之前的 run 刷出去，再原样写
                end_current_run_if_needed()
                out.write(line)
                continue

            n = int(m.group(1))
            v = int(m.group(2))

            if not in_run:
                # 开启新 run
                in_run = True
                run_start_n = run_end_n = n
                run_v = v
                run_count = 1
                run_first_line = line
                run_last_line = line
                continue

            # 如果能并入当前 run：n 连续递增且 v 相同
            if run_v == v and run_end_n is not None and n == run_end_n + 1:
                run_end_n = n
                run_count += 1
                run_last_line = line
                continue

            # 否则：当前 run 结束，刷新；然后用当前行开启新 run
            end_current_run_if_needed()
            in_run = True
            run_start_n = run_end_n = n
            run_v = v
            run_count = 1
            run_first_line = line
            run_last_line = line

        # 文件结束：刷新最后一个 run
        end_current_run_if_needed()

def main():
    ap = argparse.ArgumentParser(description="Compress consecutive '<int> <int>' lines in large logs.")
    ap.add_argument("--in", dest="in_path", required=True, help="input log path")
    ap.add_argument("--out", dest="out_path", required=True, help="output (shrunk) log path")
    ap.add_argument("--min-run", type=int, default=50,
                    help="only compress runs with at least this many lines (default: 50)")
    ap.add_argument("--style", choices=["head-tail", "summary"], default="head-tail",
                    help="compression style (default: head-tail)")
    args = ap.parse_args()

    if args.min_run < 3:
        raise SystemExit("--min-run must be >= 3 (otherwise compression is meaningless).")

    shrink_file(args.in_path, args.out_path, min_run=args.min_run, style=args.style)
    print(f"[OK] wrote: {args.out_path}")

if __name__ == "__main__":
    main()
