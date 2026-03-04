#!/usr/bin/env python3
"""
Batch runner for DiskANN search experiments.
- Sweeps quantization types: ["PQ", "RABITQ", "LSQ"]
- Sweeps search worker counts W: [1, 2, 4, 8]
- Rebuilds the project once at the beginning (make -j)
- Drops Linux page cache before each run (requires sudo)
- Streams stdout live to console and a per-run log file (tee-like)
- Writes a CSV summary at the end

Edit the CONFIG section below to match your environment.
"""
# from __future__ import annotations
import argparse
import csv
import datetime as dt
import itertools
import os
from pathlib import Path
import subprocess
import sys
import time

# ===================== CONFIG (edit me) =====================
APP = "/root/paper/DiskAnnPQ/DiskAnn/build/apps/search_disk_index"
# # SIFT-1B example
# INDEX_PREFIX = "/root/paper/sift1b/disk_index_sift_learn_R128_L125_A1.2"
# QUERY_FILE = "/root/paper/sift1b/bigann_query.bin"
# GT_FILE = "/root/paper/sift1b/sift_query_learn_gt100"
# DATA_PATH = "/root/paper/sift1b/bigann_base.bin"
# OUT_DIR = Path("/root/paper/sift1b")  # result .res goes here

# # Compression-ratio directories; each contains a quick_link.sh that retargets symlinks
# RATIO_DIRS = [
#     "/root/paper/sift1b/quant8B",
#     "/root/paper/sift1b/quant16B",
#     "/root/paper/sift1b/quant32B",
#     "/root/paper/sift1b/quant64B",
#     "/root/paper/sift1b/quant128B",
# ]

# SIFT-100M example
INDEX_PREFIX = "/root/paper/sift100m/disk_index_sift_learn_100M_R128_L125_A1.2"
QUERY_FILE = "/root/paper/sift100m/query.public.1K.u8bin"
GT_FILE = "/root/paper/sift100m/sift_query_learn_1kquery_gt100"
DATA_PATH = "/root/paper/sift100m/learn.100M.u8bin"
OUT_DIR = Path("/root/paper/sift100m")  # result .res goes here

# Compression-ratio directories; each contains a quick_link.sh that retargets symlinks
RATIO_DIRS = [
    # "/root/paper/sift100m/quant8B",
    "/root/paper/sift100m/quant16B",
    # "/root/paper/sift100m/quant32B",
    # "/root/paper/sift100m/quant64B",
    # "/root/paper/sift100m/quant128B",
]

# # SpaceV1B example
# INDEX_PREFIX = "/root/paper/spacev1b/diskann/disk_index_spacev1b_R128_L125_"
# QUERY_FILE = "/root/paper/spacev1b/query.bin"
# GT_FILE = "/root/paper/spacev1b/spacev1b_gt100.bin"
# DATA_PATH = "/root/paper/spacev1b/base_vectors_all.bin"
# OUT_DIR = Path("/root/paper/spacev1b")  # result .res goes here

# # Compression-ratio directories; each contains a quick_link.sh that retargets symlinks
# RATIO_DIRS = [
#     "/root/paper/spacev1b/diskann/quant16B",
#     "/root/paper/spacev1b/diskann/quant32B",
#     "/root/paper/spacev1b/diskann/quant64B",
# ]


# Common parameters
DATA_TYPE = "uint8"
DIST_FN = "l2"
K = 10
NUM_NODES_TO_CACHE = 10000
# L_VALUES = [10, 20, 40, 80, 100, 200, 300, 400, 600, 800, 1600, 3200]               # each run uses a single -L
L_VALUES = [15, 25, 30, 60, 140, 170, 240, 270, 340, 370, 500, 700, 900, 1000, 1200, 2400]               # each run uses a single -L

# L_VALUES = [10]               # each run uses a single -L
THREADS = 32                          # -T

QUANT_TYPES = [
    "PQ", 
    # "RABITQ", 
    # "LSQ"
]
W_LIST = [1, 4]

#-----------------------------------------------------------
LOG_DIR = OUT_DIR / "logs" / "0214test_1kquery"           # log files go here

# If your sudo is passwordless for dropping caches, leave as-is. Otherwise this
# command will prompt on the terminal.
DROP_CACHES_CMD = ["bash", "-lc", "sync; echo 3 | sudo tee /proc/sys/vm/drop_caches"]

# Rebuild once before all runs
BUILD_CMD = ["make", "-j"]
# ============================================================


def shlex_join(cmd: list[str]) -> str:
    # Python <3.8 compatibility; good enough for our controlled args
    import shlex
    return " ".join(shlex.quote(x) for x in cmd)


def ensure_dirs() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def run(cmd: list[str], *, cwd: str | None = None, env: dict | None = None) -> int:
    print(f"\n$ {shlex_join(cmd)}\n", flush=True)
    proc = subprocess.Popen(cmd, cwd=cwd, env=env)
    return proc.wait()


def run_and_tee(cmd: list[str], log_path: Path, *, cwd: str | None = None, env: dict | None = None) -> int:
    print(f"\n$ {shlex_join(cmd)}\n", flush=True)
    with log_path.open("wb") as lf:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert proc.stdout is not None
        try:
            for chunk in iter(lambda: proc.stdout.read(8192), b""):
                sys.stdout.buffer.write(chunk)
                sys.stdout.flush()
                lf.write(chunk)
        finally:
            proc.stdout.close()
        return proc.wait()


def drop_caches() -> None:
    print("Dropping Linux page caches (requires sudo)...")
    rc = run(DROP_CACHES_CMD)
    if rc != 0:
        print(f"WARNING: drop_caches returned code {rc}. Continuing anyway.")

def ensure_aio_nr_files_limit() -> None:
    """Ensure /proc/sys/fs/aio-nr and aio-max-nr are set high enough."""
    try:
        # echo 20971520  tee /proc/sys/fs/aio-max-nr
        with open("/proc/sys/fs/aio-max-nr", "r+") as f:
            cur = int(f.readline().split()[0])
            if cur < 20971520:
                print(f"Setting aio-max-nr from {cur} to 20971520")
                f.seek(0)
                f.write("20971520\n")
                f.truncate()
    except Exception as e:
        print(f"WARNING: Could not set aio-max-nr: {e}")

def build_once() -> None:
    print("Building project with make -j ...")
    rc = run(BUILD_CMD)
    if rc != 0:
        print("\nERROR: Build failed. Aborting.")
        sys.exit(rc)


def normalize_ratio_dirs(raw: list[str]) -> list[Path]:
    dirs: list[Path] = []
    for s in raw:
        p = Path(s)
        if not p.is_absolute():
            p = OUT_DIR / s
        dirs.append(p)
    return dirs


def ratio_label(p: Path) -> str:
    return p.name


def run_quick_link(dir_path: Path) -> int:
    print(f"Switching compression ratio via quick_link.sh in {dir_path} ...")
    # Use bash to run regardless of executable bit
    return run(["bash", "-lc", "bash quick_link.sh"], cwd=str(dir_path))


def cur_stamp() -> str:
    # e.g., 0911_1430
    return dt.datetime.now().strftime("%m%d_%H%M")


def make_cur_name(qtype: str, W: int, L: int, ratio: str) -> str:
    return f"{cur_stamp()}_{ratio}_W{W}_L{L}_{qtype}"


def build_search_cmd(qtype: str, W: int, Ls: list[int], result_path: Path) -> list[str]:
    cmd = [
        APP,
        "--data_type", DATA_TYPE,
        "--dist_fn", DIST_FN,
        "--index_path_prefix", INDEX_PREFIX,
        "--query_file", QUERY_FILE,
        "--gt_file", GT_FILE,
        "-K", str(K),
        "--result_path", str(result_path),
        "--num_nodes_to_cache", str(NUM_NODES_TO_CACHE),
        "--data_path", DATA_PATH,
        "-W", str(W),
        "--quantification_type", qtype,
        "-T", str(THREADS),
    ]

    # IMPORTANT: pass multiple L values as: -L l1 l2 l3 ...
    cmd += ["-L"] + [str(x) for x in Ls]
    return cmd

def format_L_label(Ls: list[int]) -> str:
    if len(Ls) == 1:
        return f"L{Ls[0]}"
    # e.g. L10-20-40-80
    label = "L" + "-".join(map(str, Ls))
    # avoid ultra-long filenames
    if len(label) > 60:
        import hashlib
        h = hashlib.md5(label.encode("utf-8")).hexdigest()[:8]
        label = f"L{Ls[0]}-{Ls[-1]}_n{len(Ls)}_{h}"
    return label


def make_cur_name(qtype: str, W: int, Ls: list[int], ratio: str) -> str:
    return f"{cur_stamp()}_{ratio}_W{W}_{format_L_label(Ls)}_{qtype}"

def main() -> None:
    parser = argparse.ArgumentParser(description="Batch DiskANN search sweeper")
    parser.add_argument("--dry", action="store_true", help="Print commands without running")
    parser.add_argument("--no-build", action="store_true", help="Skip the initial make -j build step")
    parser.add_argument("--no-drop", action="store_true", help="Skip dropping caches before each run")
    parser.add_argument("--quant", nargs="*", default=QUANT_TYPES, help="Override quant types to run")
    parser.add_argument("--w", nargs="*", type=int, default=W_LIST, help="Override W list to run")
    parser.add_argument("--L", dest="Ls", nargs="*", type=int, default=L_VALUES, help="Override L list to run; each run uses a single L")
    parser.add_argument("--ratios", dest="ratio_dirs", nargs="*", default=RATIO_DIRS, help="Compression-ratio directories; each must contain quick_link.sh. Accept absolute paths or names under OUT_DIR.")
    parser.add_argument("--tag", default="", help="Optional tag to append to CURNAME")
    parser.add_argument("--multi-L", action="store_true", help="If set, run once per (ratio, quant, W) and pass ALL Ls in a single command: -L l1 l2 ...; otherwise run one L per run.")

    args = parser.parse_args()

    ratio_dirs = normalize_ratio_dirs(args.ratio_dirs)

    ensure_dirs()
    ensure_aio_nr_files_limit()

    if not args.no_build and not args.dry:
        build_once()

    summary_rows = []
    total_start = time.time()

    for ratio_dir in ratio_dirs:
        ratio = ratio_label(ratio_dir)
        print("=" * 80)
        print(f"Switching ratio: {ratio} ({ratio_dir})")
        print("=" * 80)
        if not args.dry:
            rc_link = run_quick_link(ratio_dir)
            if rc_link != 0:
                print(f"WARNING: quick_link.sh failed with code {rc_link}; skipping this ratio.")
                continue

        if args.multi_L:
            # One run per (ratio, quant, W) with all Ls
            settings = itertools.product(args.quant, args.w)
            settings = ((qtype, W, args.Ls) for qtype, W in settings)
        else:
            # One run per (ratio, quant, W, L)
            settings = ((qtype, W, [L]) for qtype, W, L in itertools.product(args.quant, args.w, args.Ls))

        for qtype, W, L in settings:
            cname = make_cur_name(qtype, W, L, ratio)
            if args.tag:
                cname += f"_{args.tag}"
            res_path = OUT_DIR / f"{cname}.res"
            log_path = LOG_DIR / f"{cname}.log"

            cmd = build_search_cmd(qtype, W, L, res_path)

            print("=" * 80)
            print(f"Running: ratio={ratio}  quant={qtype}  W={W}  L={L}")
            print(f"Result: {res_path}")
            print(f"Log:    {log_path}")
            print("=" * 80)

            start_ts = dt.datetime.now().isoformat(timespec="seconds")
            start_time = time.time()

            if args.dry:
                print("(dry-run) ", shlex_join(cmd))
                rc = 0
            else:
                if not args.no_drop:
                    drop_caches()
                rc = run_and_tee(cmd, log_path)

            dur_sec = time.time() - start_time
            end_ts = dt.datetime.now().isoformat(timespec="seconds")

            summary_rows.append({
                "ratio": ratio,
                "quant": qtype,
                "W": W,
                "L": L,
                "result_path": str(res_path),
                "log_path": str(log_path),
                "return_code": rc,
                "start": start_ts,
                "end": end_ts,
                "duration_sec": f"{dur_sec:.1f}",
            })

            print(f"\nFinished run (quant={qtype}, W={W}) with rc={rc}, took {dur_sec:.1f}s\n")
            # Small gap between runs
            time.sleep(1)

    # Write CSV summary
    summary_csv = OUT_DIR / f"batch_summary_{dt.datetime.now().strftime('%m%d_%H%M%S')}.csv"
    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "ratio", "quant", "W", "L", "result_path", "log_path", "return_code", "start", "end", "duration_sec"
        ])
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    total_dur = time.time() - total_start
    print("\nAll runs complete.")
    print(f"Summary CSV: {summary_csv}")
    print(f"Total duration: {total_dur/60:.1f} minutes\n")


if __name__ == "__main__":
    main()
