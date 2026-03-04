#!/usr/bin/env python3
"""Sum `time` command outputs of the form: `real 12m34.567s`.

Examples:
  python sum_real_time.py DiskAnnPQ/logs/main/1b/gorgeous_sift1b-5shard.log
  python sum_real_time.py DiskAnnPQ/logs/main/1b/  # scans common log extensions recursively

By default, if an input path is a directory, this script scans it recursively and
includes files with extensions: .log .txt .out
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence


REAL_MS_RE = re.compile(r"^\s*real\s+(?P<m>\d+)m(?P<s>\d+(?:\.\d+)?)s\s*$")
REAL_HMS_RE = re.compile(
    r"^\s*real\s+(?P<h>\d+)h(?P<m>\d+)m(?P<s>\d+(?:\.\d+)?)s\s*$"
)
REAL_S_RE = re.compile(r"^\s*real\s+(?P<s>\d+(?:\.\d+)?)s\s*$")

RUNNING_RE = re.compile(r"^\s*running\s+(?P<label>.+?)\s*$", re.IGNORECASE)
TIME_LINE_RE = re.compile(r"^\s*(real|user|sys)\b", re.IGNORECASE)


@dataclass(frozen=True)
class ParseStat:
    files: int
    lines: int
    matches: int
    seconds: float


@dataclass
class RunningAgg:
    matches: int = 0
    seconds: float = 0.0


def _normalize_context_line(line: str, max_len: int) -> str:
    s = " ".join(line.strip().split())
    if max_len > 0 and len(s) > max_len:
        return s[: max_len - 1] + "…"
    return s


def _prev_key(line: str, mode: str, max_len: int) -> str:
    norm = _normalize_context_line(line, max_len=0)
    if mode == "full":
        return _normalize_context_line(norm, max_len=max_len)

    # category mode: try to map common "Running ..." lines to stable buckets.
    rm = RUNNING_RE.match(norm)
    if rm:
        return _running_key(rm.group("label"), mode="category")

    # Otherwise, shorten noisy lines (paths / long suffixes) into a stable prefix.
    cut = norm
    for sep in ("...", ":"):
        if sep in cut:
            cut = cut.split(sep, 1)[0].strip()
    return _normalize_context_line(cut, max_len=max_len)


def _running_key(label: str, mode: str) -> str:
    raw = label.strip()
    if mode == "full":
        return raw

    low = raw.lower()
    if low.startswith("benchmarks for"):
        return "benchmarks"
    if low.startswith("benchmark for"):
        return "benchmark"
    if low.startswith("graph partition"):
        return "graph partition"
    if low.startswith("relayout"):
        return "relayout"

    # Fallback: shorten noisy lines.
    cut = raw
    for sep in ("...", ":"):
        if sep in cut:
            cut = cut.split(sep, 1)[0].strip()
    return cut


def _iter_files(inputs: Sequence[str], exts: set[str]) -> Iterator[Path]:
    for raw in inputs:
        p = Path(raw)
        if p.is_file():
            yield p
            continue
        if p.is_dir():
            for root, _, files in os.walk(p):
                for name in files:
                    fp = Path(root) / name
                    if fp.suffix.lower() in exts:
                        yield fp
            continue
        raise FileNotFoundError(raw)


def _parse_real_seconds(line: str) -> float | None:
    m = REAL_HMS_RE.match(line)
    if m:
        h = int(m.group("h"))
        mm = int(m.group("m"))
        ss = float(m.group("s"))
        return h * 3600.0 + mm * 60.0 + ss

    m = REAL_MS_RE.match(line)
    if m:
        mm = int(m.group("m"))
        ss = float(m.group("s"))
        return mm * 60.0 + ss

    m = REAL_S_RE.match(line)
    if m:
        return float(m.group("s"))

    return None


def _format_hms(total_seconds: float) -> str:
    # Avoid negative due to float rounding.
    total_seconds = max(0.0, total_seconds)
    whole = int(total_seconds)
    frac = total_seconds - whole

    h = whole // 3600
    whole %= 3600
    m = whole // 60
    s = whole % 60

    # Keep milliseconds-ish precision if present.
    s_with_frac = s + frac
    return f"{h}h{m:02d}m{s_with_frac:06.3f}s"


def sum_real_times(
    paths: Sequence[str],
    exts: set[str],
    per_file: bool,
    by_running: bool,
    running_mode: str,
    by_prev_line: bool,
    prev_mode: str,
    prev_max_len: int,
) -> tuple[ParseStat, dict[str, RunningAgg], dict[str, RunningAgg]]:
    files = 0
    lines = 0
    matches = 0
    seconds = 0.0

    running_aggs: dict[str, RunningAgg] = {}
    prev_aggs: dict[str, RunningAgg] = {}

    for fp in _iter_files(paths, exts):
        files += 1
        file_matches = 0
        file_seconds = 0.0
        current_running: str | None = None
        current_context: str | None = None

        try:
            with fp.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    lines += 1

                    if by_prev_line:
                        # Track the most recent "meaningful" line prior to a time output.
                        if line.strip() and not TIME_LINE_RE.match(line):
                            current_context = _prev_key(line, mode=prev_mode, max_len=prev_max_len)

                    if by_running:
                        rm = RUNNING_RE.match(line)
                        if rm:
                            current_running = _running_key(rm.group("label"), running_mode)
                            continue

                    v = _parse_real_seconds(line)
                    if v is None:
                        continue
                    matches += 1
                    file_matches += 1
                    seconds += v
                    file_seconds += v

                    if by_running:
                        key = current_running if current_running is not None else "<unknown>"
                        agg = running_aggs.get(key)
                        if agg is None:
                            agg = RunningAgg()
                            running_aggs[key] = agg
                        agg.matches += 1
                        agg.seconds += v

                    if by_prev_line:
                        key = current_context if current_context is not None else "<unknown>"
                        agg = prev_aggs.get(key)
                        if agg is None:
                            agg = RunningAgg()
                            prev_aggs[key] = agg
                        agg.matches += 1
                        agg.seconds += v
        except IsADirectoryError:
            # Should not happen due to _iter_files.
            continue

        if per_file and file_matches:
            print(f"{fp}\t{file_matches}\t{_format_hms(file_seconds)}\t({file_seconds:.3f}s)")

    return ParseStat(files=files, lines=lines, matches=matches, seconds=seconds), running_aggs, prev_aggs


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description="Sum time outputs like: real 12m34.567s")
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more files or directories (directories scanned recursively)",
    )
    parser.add_argument(
        "--ext",
        default=".log,.txt,.out",
        help="Comma-separated extensions to scan when a path is a directory (default: .log,.txt,.out)",
    )
    parser.add_argument(
        "--per-file",
        action="store_true",
        help="Print per-file subtotal lines (only files with matches)",
    )
    parser.add_argument(
        "--by-running",
        action="store_true",
        help="Group and sum by the most recent 'Running ...' line before each 'real ...'",
    )
    parser.add_argument(
        "--by-prev-line",
        action="store_true",
        help="Group and sum by the nearest previous non-empty line before each 'real ...' (not limited to Running)",
    )
    parser.add_argument(
        "--running-mode",
        choices=("category", "full"),
        default="category",
        help="How to build the grouping key from 'Running ...' (default: category)",
    )
    parser.add_argument(
        "--prev-max-len",
        type=int,
        default=120,
        help="Max length of the context line when using --by-prev-line (default: 120, 0 means unlimited)",
    )
    parser.add_argument(
        "--prev-mode",
        choices=("category", "full"),
        default="category",
        help="How to build the grouping key for --by-prev-line (default: category)",
    )

    args = parser.parse_args(argv)

    exts = {e.strip().lower() for e in args.ext.split(",") if e.strip()}
    exts = {e if e.startswith(".") else f".{e}" for e in exts}

    try:
        stat, running_aggs, prev_aggs = sum_real_times(
            args.paths,
            exts=exts,
            per_file=args.per_file,
            by_running=args.by_running,
            running_mode=args.running_mode,
            by_prev_line=args.by_prev_line,
            prev_mode=args.prev_mode,
            prev_max_len=args.prev_max_len,
        )
    except FileNotFoundError as e:
        print(f"ERROR: path not found: {e}", file=sys.stderr)
        return 2

    print("---")
    print(f"files_scanned: {stat.files}")
    print(f"lines_scanned: {stat.lines}")
    print(f"real_matches:  {stat.matches}")
    print(f"total_real:    {_format_hms(stat.seconds)}")
    print(f"total_seconds: {stat.seconds:.3f}")

    if args.by_running:
        print("---")
        print("by_running (sorted by total_seconds desc):")
        for key, agg in sorted(running_aggs.items(), key=lambda kv: kv[1].seconds, reverse=True):
            print(f"{key}\t{agg.matches}\t{_format_hms(agg.seconds)}\t({agg.seconds:.3f}s)")

    if args.by_prev_line:
        print("---")
        print("by_prev_line (sorted by total_seconds desc):")
        for key, agg in sorted(prev_aggs.items(), key=lambda kv: kv[1].seconds, reverse=True):
            print(f"{key}\t{agg.matches}\t{_format_hms(agg.seconds)}\t({agg.seconds:.3f}s)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
