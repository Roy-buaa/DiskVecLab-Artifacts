import os
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import argparse

# ===================== 参数解析：总线程数 & 每任务线程数 =====================

parser = argparse.ArgumentParser()
parser.add_argument(
    "--max-threads",
    type=int,
    default=128,
    help="总线程预算 y（所有 quantizer 进程总共最多占用多少线程）",
)
parser.add_argument(
    "--threads-per-task",
    type=int,
    default=16,
    help="每个 quantizer 进程内部使用多少线程 x（设置 OMP_NUM_THREADS）",
)
parser.add_argument(
    "--force-rebuild",
    action="store_true",
    help="忽略 .done 标记，强制重跑所有任务",
)

args = parser.parse_args()
MAX_THREADS = args.max_threads       # y
THREADS_PER_TASK = args.threads_per_task  # x
FORCE_REBUILD = args.force_rebuild

# 能并行的最大任务数： floor(y / x)，至少为 1
MAX_PARALLEL_JOBS = max(1, MAX_THREADS // THREADS_PER_TASK)

print(
    f"[CONF] max_threads={MAX_THREADS}, "
    f"threads_per_task={THREADS_PER_TASK}, "
    f"max_parallel_jobs={MAX_PARALLEL_JOBS}"
)

# 每个 quantizer 进程用这么多 OMP 线程
os.environ["OMP_NUM_THREADS"] = str(THREADS_PER_TASK)

# ===================== 基本配置 =====================

# 在这里切换数据集
DATASET = "sift100m"  # 'sift1m' 或 'sift1b' 或 'sift100m'

if DATASET == "sift1m":
    DATA_PATH = "/root/paper/sift1m/sift_base.fvecs"
    DATA_TYPE = "float"
elif DATASET == "sift1b":
    DATA_PATH = "/root/paper/sift1b/bigann_base.bvecs"
    DATA_TYPE = "uint8"
elif DATASET == "sift100m":
    DATA_PATH = "/root/paper/sift100m/learn.100M.fbin"
    DATA_TYPE = "float"
else:
    raise ValueError(f"Unknown DATASET: {DATASET}")

# quantizer 可执行文件路径
QUANTIZER_BIN = "/root/paper/DiskAnnPQ/SPTAG/Release/quantizer"

# 数据集根目录，例如 /root/paper/sift1b
DATASET_ROOT = Path("/root/paper") / DATASET

# log 目录：/root/paper/{DATASET}/logs/quantizer/
LOG_ROOT = DATASET_ROOT / "logs" / "quantizer"
LOG_ROOT.mkdir(parents=True, exist_ok=True)

# ===================== 量化配置 =====================

# 要跑的 quant_size 和 quant_type 组合
QUANT_SIZES = [16, 32, 64]
QUANT_TYPES = [
    "RabitQ",
    "PQQuantizer",
    "LSQ",
]

# ===================== 帮助函数 =====================


def run_cmd_and_log(cmd, log_path: Path) -> int:
    """
    运行命令：
    - stdout+stderr 实时打印到控制台
    - 同时写入 log_path
    - 返回进程的 returncode
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[RUN] {' '.join(cmd)}")
    with log_path.open("w", encoding="utf-8") as f:
        header = (
            f"# CMD: {' '.join(cmd)}\n"
            f"# TIME: {datetime.now().isoformat()}\n\n"
        )
        f.write(header)
        f.flush()

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # 行缓冲
            env=dict(os.environ),  # 带上 OMP_NUM_THREADS 等
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            # 多任务并行时，这里输出会互相穿插，这是预期行为
            print(line, end="")
            f.write(line)
            f.flush()

        proc.wait()
        returncode = proc.returncode

        f.write(f"\n# RETURN_CODE: {returncode}\n")
        f.flush()

    print(f"[LOG] {log_path} (ret={returncode})")
    if returncode != 0:
        print(f"[WARN] Non-zero return code for {log_path.name}")

    return returncode


# ===================== Task 定义 & 执行函数 =====================

@dataclass
class QuantTask:
    quant_size: int
    quant_type: str
    rbq_bits: int
    pq_chunk: int
    lsq_chunk: int
    lsq_chunk_bits: int
    oq_path: Path
    o_path: Path
    log_path: Path
    marker_path: Path


def run_one_task(task: QuantTask) -> int:
    # 单个任务开始时间
    task_start = datetime.now()
    print(
        f"[TASK-START] {DATASET} quant{task.quant_size}B "
        f"{task.quant_type} @ {task_start.isoformat()}"
    )

    cmd = [
        QUANTIZER_BIN,
        "-d", "128",
        "-qd", str(task.pq_chunk),
        "-rbqbits", str(task.rbq_bits),
        "-lsq_subvector", str(task.lsq_chunk),
        "-lsqbits", str(task.lsq_chunk_bits),
        "-v", DATA_TYPE,
        "-i", DATA_PATH,
        "-oq", str(task.oq_path),
        "-o", str(task.o_path),
        "-f", "XVEC" if DATA_PATH.endswith("vecs") else "DEFAULT",
        "--quantizer", task.quant_type,
        "--debug", "true",
        "--train_samples", "1000000",
    ]

    ret = run_cmd_and_log(cmd, task.log_path)

    task_end = datetime.now()
    elapsed_sec = (task_end - task_start).total_seconds()

    print(
        f"[TASK-END] {DATASET} quant{task.quant_size}B {task.quant_type} "
        f"@ {task_end.isoformat()} | elapsed={elapsed_sec:.2f}s"
    )

    if ret == 0:
        task.marker_path.write_text(
            (
                f"OK {task_end.isoformat()}\n"
                f"DATASET={DATASET}\n"
                f"QUANT_SIZE={task.quant_size}\n"
                f"QUANT_TYPE={task.quant_type}\n"
                f"OQ={task.oq_path}\n"
                f"O={task.o_path}\n"
                f"ELAPSED_SEC={elapsed_sec:.3f}\n"
            ),
            encoding="utf-8",
        )
        print(f"[MARK] Done marker written: {task.marker_path}")
    else:
        if task.marker_path.exists():
            task.marker_path.unlink()
        print(f"[MARK] Failed; removed marker if existed: {task.marker_path}")

    return ret


# ===================== 主逻辑：收集任务 + 并行执行 =====================

def main():
    tasks: list[QuantTask] = []

    for quant_size in QUANT_SIZES:
        # 量化参数
        rbq_bits = quant_size // 16
        pq_chunk = quant_size
        lsq_chunk = quant_size * 2
        lsq_chunk_bits = 4

        quant_dir = DATASET_ROOT / f"quant{quant_size}B"
        quant_dir.mkdir(parents=True, exist_ok=True)

        for quant_type in QUANT_TYPES:
            # LSQ 限制
            if quant_type == "LSQ" and quant_size > 32:
                continue

            # RabitQ 限制
            if quant_type == "RabitQ" and quant_size < 16:
                continue

            oq_path = quant_dir / f"OQ_{quant_type}.spannvec"
            o_path = quant_dir / f"O_{quant_type}.spannvec"

            log_path = LOG_ROOT / f"quant_{quant_size}B_{quant_type}.log"
            marker_path = LOG_ROOT / f"quant_{quant_size}B_{quant_type}.done"

            if marker_path.exists() and not FORCE_REBUILD:
                print(
                    f"[SKIP] {DATASET} quant{quant_size}B {quant_type} already done "
                    f"({marker_path})"
                )
                continue

            tasks.append(
                QuantTask(
                    quant_size=quant_size,
                    quant_type=quant_type,
                    rbq_bits=rbq_bits,
                    pq_chunk=pq_chunk,
                    lsq_chunk=lsq_chunk,
                    lsq_chunk_bits=lsq_chunk_bits,
                    oq_path=oq_path,
                    o_path=o_path,
                    log_path=log_path,
                    marker_path=marker_path,
                )
            )

    if not tasks:
        print("[INFO] 没有需要执行的任务（可能都已经有 .done 标记了）")
        return

    print(
        f"[INFO] 待执行任务数量: {len(tasks)}, "
        f"最大并行数: {MAX_PARALLEL_JOBS}"
    )

    # 并行执行
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_JOBS) as executor:
        future_to_task = {executor.submit(run_one_task, t): t for t in tasks}

        for future in as_completed(future_to_task):
            t = future_to_task[future]
            try:
                ret = future.result()
                status = "OK" if ret == 0 else f"FAIL({ret})"
                print(
                    f"[DONE] {DATASET} quant{t.quant_size}B {t.quant_type}: {status}"
                )
            except Exception as e:
                print(
                    f"[ERROR] {DATASET} quant{t.quant_size}B {t.quant_type}: {e!r}"
                )


if __name__ == "__main__":
    # 全局开始时间
    global_start = datetime.now()
    print(f"[GLOBAL-START] {global_start.isoformat()}")

    main()

    # 全局结束时间 & 总耗时
    global_end = datetime.now()
    total_elapsed_sec = (global_end - global_start).total_seconds()

    print(f"[GLOBAL-END]   {global_end.isoformat()}")
    print(f"[GLOBAL-ELAPSED] {total_elapsed_sec:.2f}s")

    # 把总耗时写入一个 summary 文件（追加）
    summary_path = LOG_ROOT / "quantizer_build_summary.log"
    with summary_path.open("a", encoding="utf-8") as f:
        f.write(
            "START={start} END={end} ELAPSED_SEC={elapsed:.3f} "
            "DATASET={dataset} MAX_THREADS={max_threads} "
            "THREADS_PER_TASK={threads_per_task} MAX_PARALLEL_JOBS={max_jobs}\n".format(
                start=global_start.isoformat(),
                end=global_end.isoformat(),
                elapsed=total_elapsed_sec,
                dataset=DATASET,
                max_threads=MAX_THREADS,
                threads_per_task=THREADS_PER_TASK,
                max_jobs=MAX_PARALLEL_JOBS,
            )
        )
    print(f"[GLOBAL-SUMMARY] written to {summary_path}")
