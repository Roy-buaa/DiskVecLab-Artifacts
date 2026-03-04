#!/usr/bin/env python3
"""
PipeANN experiment orchestrator.

- 每组实验分为两大步：build（构建）和 search（搜索）。
- build 里再细分为：
    1) build_disk_index
    2) build_mem_index（包含 gen_random_slice + build_memory_index）
- 每个小步结束会写一个 .done 文件，后续自动跳过。
- 构建 / 搜索命令的 stdout/stderr 全量写入 log 文件中，同时在终端实时打印。
- 默认只配置了 SIFT100M，一般放在 /root/paper/sift100m 下。
  如需支持其他数据集，可以仿照 SIFT100M 的配置再扩展一个 Experiment。

假设本脚本放在 /root/paper/DiskAnnPQ/PipeANN 目录下使用：
    python run_pipeann.py --dataset sift100m --phase both
"""

import argparse
import os
import sys
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional


@dataclass
class Experiment:
    name: str
    dtype: str
    disk_R: int
    dataset_base: Path
    index_prefix: str  # 原始 C++ 程序直接拼接用，保持字符串
    query_base: Path
    gt_base: Path
    sample_rate: float
    mem_r: int
    mem_build_l: int
    mem_alpha: float
    mem_build_threads: int
    thread_num: int
    ks: List[int]
    beamwidths: List[int]
    
    search_thread_num: int | List[int] = 32  # 搜索时使用的线程数
    disk_pq_bytes: Optional[int] = None  # 可选参数，暂未使用
    disk_memory_budget: Optional[int] = None  # 可选参数，暂未使用
    disk_build_threads: Optional[int] = None  # 可选参数，暂未使用
    memls: List[int] = field(default_factory=lambda: [0, 10])


def run_and_log(cmd, log_path: Path, cwd: Optional[Path] = None):
    """运行子进程，实时打印，同时把 stdout+stderr 全量追加到日志文件。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd_str = " ".join(str(c) for c in cmd)
    print(f"==> RUN: {cmd_str}")
    print(f"    LOG: {log_path}")
    start_ts = datetime.now().isoformat(timespec="seconds")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n\n===== COMMAND: {cmd_str}\n")
        f.write(f"===== START: {start_ts}\n")
        f.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
        ret = proc.wait()
        end_ts = datetime.now().isoformat(timespec="seconds")
        f.write(f"===== END: {end_ts}, RETURN CODE: {ret}\n")
        f.flush()
        if ret != 0:
            raise subprocess.CalledProcessError(ret, cmd_str)


def ensure_binaries_built(root: Path, force: bool, log_root: Path):
    """
    保证 PipeANN 的 CMake + make 已经跑过。
    简单判断 build/tests/build_disk_index 是否存在；如不存在或 force=True 则重新编译。
    """
    build_dir = root / "build"
    tests_dir = build_dir / "tests"
    disk_bin = tests_dir / "build_disk_index"

    if disk_bin.exists() and not force:
        print(f"[SKIP] PipeANN binaries 已存在：{disk_bin}")
        return

    build_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_root / "pipeann_build_binaries.log"

    # cmake ..
    run_and_log(["cmake", ".."], log_path, cwd=build_dir)
    # make -j
    run_and_log(["make", "-j"], log_path, cwd=build_dir)

    if not disk_bin.exists():
        raise RuntimeError(f"PipeANN binary 构建后仍未找到: {disk_bin}")


def get_paths(root: Path):
    """根据 PipeANN 根目录，返回几个常用路径。"""
    build_dir = root / "build"
    tests_dir = build_dir / "tests"
    utils_dir = tests_dir / "utils"

    return {
        "build_dir": build_dir,
        "tests_dir": tests_dir,
        "utils_dir": utils_dir,
        "build_disk_bin": tests_dir / "build_disk_index",
        "build_mem_bin": tests_dir / "build_memory_index",
        "gen_random_bin": utils_dir / "gen_random_slice",
        "search_disk_bin": tests_dir / "search_disk_index",
    }


def build_disk_index(exp: Experiment, paths, log_root: Path, meta_root: Path, force: bool):
    done_file = meta_root / exp.name / "disk_index.done"
    done_file.parent.mkdir(parents=True, exist_ok=True)
    if done_file.exists() and not force:
        print(f"[SKIP] disk_index 已完成（存在 {done_file}）")
        return

    cmd = [
        str(paths["build_disk_bin"]),
        exp.dtype,
        str(exp.dataset_base),
        exp.index_prefix,
        str(exp.disk_R),
        str(exp.disk_R), # disk_L
        "32" if exp.disk_pq_bytes is None else str(exp.disk_pq_bytes),
        "128" if exp.disk_memory_budget is None else str(exp.disk_memory_budget),
        "128" if exp.disk_build_threads is None else str(exp.disk_build_threads),
        "l2",
        "0",
    ]
    log_path = log_root / exp.name / "build_disk_index.log"
    run_and_log(cmd, log_path)

    done_file.write_text(
        f"{datetime.now().isoformat(timespec='seconds')} disk_index OK\n",
        encoding="utf-8",
    )
    print(f"[DONE] disk_index 完成，写入 {done_file}")


def build_mem_index(exp: Experiment, paths, log_root: Path, meta_root: Path, force: bool):
    done_file = meta_root / exp.name / "mem_index.done"
    done_file.parent.mkdir(parents=True, exist_ok=True)
    if done_file.exists() and not force:
        print(f"[SKIP] mem_index 已完成（存在 {done_file}）")
        return

    log_path = log_root / exp.name / "build_mem_index.log"

    # 1. gen_random_slice
    slice_prefix = f"{exp.index_prefix}_SAMPLE_RATE_{exp.sample_rate}"
    cmd_slice = [
        str(paths["gen_random_bin"]),
        exp.dtype,
        str(exp.dataset_base),
        slice_prefix,
        str(exp.sample_rate),
    ]
    run_and_log(cmd_slice, log_path)

    # 2. build_memory_index
    data_bin = f"{slice_prefix}_data.bin"
    ids_bin = f"{slice_prefix}_ids.bin"
    mem_index_path = f"{exp.index_prefix}_mem.index"

    cmd_mem = [
        str(paths["build_mem_bin"]),
        exp.dtype,
        data_bin,
        ids_bin,
        mem_index_path,
        "0",
        "0",
        str(exp.mem_r),
        str(exp.mem_build_l),
        str(exp.mem_alpha),
        str(exp.mem_build_threads),
        "l2",
    ]
    run_and_log(cmd_mem, log_path)

    done_file.write_text(
        f"{datetime.now().isoformat(timespec='seconds')} mem_index OK\n",
        encoding="utf-8",
    )
    print(f"[DONE] mem_index 完成，写入 {done_file}")


def search(exp: Experiment, paths, log_root: Path, meta_root: Path, force: bool):
    done_file = meta_root / exp.name / "search.done"
    done_file.parent.mkdir(parents=True, exist_ok=True)
    if done_file.exists() and not force:
        print(f"[SKIP] search 已完成（存在 {done_file}）")
        return

    ks_strs = [str(k) for k in exp.ks]
    if isinstance(exp.search_thread_num, list):
        thread_nums = exp.search_thread_num
    else:
        thread_nums = [exp.search_thread_num]


    for bw in exp.beamwidths:
        for meml in exp.memls:
            for mode in [0, 2]:
                for search_thread_num in thread_nums:
                    cmd = [
                        str(paths["search_disk_bin"]),
                        exp.dtype,
                        exp.index_prefix,
                        str(search_thread_num),
                        str(bw),
                        str(exp.query_base),
                        str(exp.gt_base),
                        "10",
                        "l2",
                        "pq",
                        str(mode),
                        str(meml),
                        *ks_strs,
                    ]
                    log_path = log_root / exp.name / f"search_bw{bw}_meml{meml}_pipe{mode}_T{search_thread_num}_{datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')}.log"
                    run_and_log(cmd, log_path)

    done_file.write_text(
        f"{datetime.now().isoformat(timespec='seconds')} search OK\n",
        encoding="utf-8",
    )
    print(f"[DONE] search 完成，写入 {done_file}")

def create_sift1b_experiment() -> Experiment:
    """
    按你现有脚本，把 SIFT100M 的路径和参数集中到一个 Experiment 里。
    如需改路径或参数，只要改这里。
    """
    ks = list(range(10, 300, 5))  # 10,20,...,200
    beamwidths = [32, 16, 8, 4, 2, 1]

    return Experiment(
        name="sift1b",
        dtype="uint8",
        disk_R=128,
        dataset_base=Path("/root/paper/sift1b/bigann_base.bin"),
        index_prefix="/root/paper/sift1b/pipeann/",
        query_base=Path("/root/paper/sift1b/bigann_query.og1k.u8bin"),
        gt_base=Path("/root/paper/sift1b/bigann_query.og1k.u8bin.1bbase.gt"),
        sample_rate=0.001,
        mem_r=24,
        mem_build_l=128,
        mem_alpha=1.2,
        mem_build_threads=64,
        thread_num=128,
        ks=ks,
        beamwidths=beamwidths,
        memls=[0, 10],
        disk_memory_budget=400,
        disk_build_threads=128,
    )

def create_sift100m_experiment() -> Experiment:
    """
    按你现有脚本，把 SIFT100M 的路径和参数集中到一个 Experiment 里。
    如需改路径或参数，只要改这里。
    """
    # ks = list(range(10, 501, 5))  # 10,20,...,200
    ks = [28, 40, 102]
    beamwidths = [32, 16, 8, 4, 2, 1]
    # thread_nums = [i for i in range(1, 33)]

    return Experiment(
        name="sift100m",
        dtype="uint8",
        disk_R=128,
        dataset_base=Path("/root/paper/sift100m/learn.100M.u8bin"),
        index_prefix="/root/paper/sift100m/pipeann/",
        query_base=Path("/root/paper/sift100m/query.public.10K.u8bin"),
        gt_base=Path("/root/paper/sift100m/sift_query_learn_100m_gt100"),
        sample_rate=0.001,
        mem_r=24,
        mem_build_l=128,
        mem_alpha=1.2,
        mem_build_threads=64,
        thread_num=128,
        ks=ks,
        beamwidths=beamwidths,
        memls=[0, 10],
        # search_thread_num=thread_nums,
    )

def create_spacev1b_experiment() -> Experiment:
    """
    按你现有脚本，把 SpaceV1B 的路径和参数集中到一个 Experiment 里。
    如需改路径或参数，只要改这里。
    """
    ks = list(range(10, 300, 10))  # 10,20,...,200
    beamwidths = [32, 16, 8, 4, 2, 1]

    return Experiment(
        name="spacev1b",
        dtype="uint8",
        disk_R=128,
        dataset_base=Path("/root/paper/spacev1b/base_vectors_1b.bin"),
        index_prefix="/root/paper/spacev1b/pipeann/",
        query_base=Path("/root/paper/spacev1b/query1k.bin"),
        gt_base=Path("/root/paper/spacev1b/query1k.bin.1bbase.k100.gt"),
        sample_rate=0.001,
        mem_r=24,
        mem_build_l=128,
        mem_alpha=1.2,
        mem_build_threads=128,
        thread_num=128,
        disk_memory_budget=25,
        ks=ks,
        beamwidths=beamwidths,
    )

def create_laion50m768d_experiment() -> Experiment:
    """
    按你现有脚本，把 LAION100m 的路径和参数集中到一个 Experiment 里。
    如需改路径或参数，只要改这里。
    """
    ks = list(range(10, 501, 5))  # 10,20,...,200
    beamwidths = [32, 16, 8, 4, 2, 1]

    return Experiment(
        name="laion50m768d",
        dtype="float",
        disk_R=128,
        dataset_base=Path("/root/paper/laion100m/laion50m768d.fbin"),
        index_prefix="/root/paper/laion100m/pipeann768d/",
        query_base=Path("/root/paper/laion100m/laion1k768d.fbin"),
        gt_base=Path("/root/paper/laion100m/laion1k768d.fbin.50mbase.k100.gt"),
        sample_rate=0.001,
        mem_r=24,
        mem_build_l=128,
        mem_alpha=1.2,
        mem_build_threads=128,
        thread_num=128,
        ks=ks,
        beamwidths=beamwidths,
        disk_pq_bytes=192,
        disk_memory_budget=450,
        disk_build_threads=128,
        memls=[0, 10],
    )

def create_laion50m512d_experiment() -> Experiment:
    """
    按你现有脚本，把 laion50m512d 的路径和参数集中到一个 Experiment 里。
    如需改路径或参数，只要改这里。
    """
    ks = list(range(10, 201, 10)) + [300, 500, 750, 1000, 1500, 2500, 4000] # 10,20,...,200
    beamwidths = [32, 16, 8, 4, 2, 1]

    return Experiment(
        name="laion50m512d",
        dtype="float",
        disk_R=128,
        dataset_base=Path("/root/paper/laion100m/laion_512dim_50m.fbin"),
        index_prefix="/root/paper/laion100m/pipeann512d/",
        query_base=Path("/root/paper/laion100m/laion_512dim_1k_ood.fbin"),
        gt_base=Path("/root/paper/laion100m/laion_512dim_1k_ood.fbin.50m.k100.gt"),
        sample_rate=0.001,
        mem_r=24,
        mem_build_l=128,
        mem_alpha=1.2,
        mem_build_threads=128,
        thread_num=128,
        ks=ks,
        beamwidths=beamwidths,
        disk_pq_bytes=192,
        disk_memory_budget=400,
        disk_build_threads=128,
    )

def create_deep100m_experiment() -> Experiment:
    """
    按你现有脚本，把 LAION50m 的路径和参数集中到一个 Experiment 里。
    如需改路径或参数，只要改这里。
    """
    ks = list(range(10, 201, 10)) + [250, 300, 400, 500, 750, 1000, 1500]  # 10,20,...,200
    beamwidths = [32, 16, 8, 4, 2, 1]

    return Experiment(
        name="deep100m",
        dtype="float",
        disk_R=128,
        dataset_base=Path("/root/paper/deep1b/base100M.fbin"),
        index_prefix="/root/paper/deep1b/pipeann/",
        query_base=Path("/root/paper/deep1b/base1k.fbin"),
        gt_base=Path("/root/paper/deep1b/base1k.fbin.base100m.k100.gt"),
        sample_rate=0.001,
        mem_r=24,
        mem_build_l=128,
        mem_alpha=1.2,
        mem_build_threads=128,
        thread_num=128,
        ks=ks,
        beamwidths=beamwidths,
        disk_pq_bytes=24,
        disk_memory_budget=400,
        disk_build_threads=128,
    )



def parse_args():
    p = argparse.ArgumentParser(
        description="PipeANN 构建 + 搜索脚本（带步骤跳过和日志记录）"
    )
    p.add_argument(
        "--root",
        type=str,
        default="/root/paper/DiskAnnPQ/PipeANN",
        help="PipeANN 源码根目录（不指定则用脚本所在目录）",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="",
        choices=["sift1b", "sift100m", "spacev1b", "laion50m768d", "laion50m512d", "deep100m", ""],
        help="数据集名称，目前只内置 sift100m，如需其他数据集可自己扩展",
    )
    p.add_argument(
        "--phase",
        type=str,
        default="both",
        choices=["build", "search", "both"],
        help="执行阶段：build（只构建索引）/ search（只搜索）/ both（先构建再搜索）",
    )
    p.add_argument(
        "--force-binaries",
        action="store_true",
        help="强制重新 cmake + make（忽略已存在的二进制）",
    )
    p.add_argument(
        "--force-build",
        action="store_true",
        help="强制重新构建 disk_index 和 mem_index（忽略 .done）",
    )
    p.add_argument(
        "--force-search",
        action="store_true",
        help="强制重新搜索（忽略 search.done）",
    )
    return p.parse_args()

def main():
    args = parse_args()

    if args.dataset == "":
        args.dataset = input("请输入数据集名称（sift100m / spacev1b）：").strip().lower()

    # 1) PipeANN 源码根目录：只用于 cmake / make 和找 build/tests 下的二进制
    if args.root is None:
        pipeann_root = Path(__file__).resolve().parent
    else:
        pipeann_root = Path(args.root).expanduser().resolve()

    print(f"[INFO] PipeANN root = {pipeann_root}")

    # 2) 先根据 dataset 选出实验配置
    if args.dataset == "sift1b":
        exp = create_sift1b_experiment()
    elif args.dataset == "sift100m":
        exp = create_sift100m_experiment()
    elif args.dataset == "spacev1b":
        exp = create_spacev1b_experiment()
    elif args.dataset == "laion50m768d":
        exp = create_laion50m768d_experiment()
    elif args.dataset == "laion50m512d":
        exp = create_laion50m512d_experiment()
    elif args.dataset == "deep100m":
        exp = create_deep100m_experiment()
    else:
        raise ValueError(f"未支持的数据集：{args.dataset}")

    # 3) 以 index_prefix 对应目录作为“实验根目录”
    #    所有和具体数据集相关的 log / meta（.done）都写到这里
    exp_root = Path(exp.index_prefix).expanduser().resolve()
    log_root = exp_root / "logs"
    meta_root = exp_root / "meta"

    # 4) 编译 PipeANN 二进制：和数据集无关，日志统一放在 pipeann_root/logs 下
    ensure_binaries_built(
        pipeann_root,
        force=args.force_binaries,
        log_root=pipeann_root / "logs",
    )
    paths = get_paths(pipeann_root)

    # 5) build 阶段：写入 {index_prefix}/logs 和 {index_prefix}/meta
    if args.phase in ("build", "both"):
        print(f"\n==== [BUILD] {exp.name} ====")
        build_disk_index(exp, paths, log_root, meta_root, force=args.force_build)
        build_mem_index(exp, paths, log_root, meta_root, force=args.force_build)

    # 6) search 阶段：同样写入 {index_prefix}/logs 和 {index_prefix}/meta
    if args.phase in ("search", "both"):
        print(f"\n==== [SEARCH] {exp.name} ====")
        search(exp, paths, log_root, meta_root, force=args.force_search)


if __name__ == "__main__":
    main()
