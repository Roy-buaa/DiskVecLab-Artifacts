#!/usr/bin/env python3
"""
Experiment runner for SPANN-style index build + search.

特性：
- (dataset, quant_size, quant_type) 自动生成:
    - VectorPath: <root>/quant{quant_size}B/O_{quant_type}.spannvec
    - QuantizerFilePath: <root>/quant{quant_size}B/OQ_{quant_type}.spannvec
    - IndexDirectory: <root>/spann/<exp_name>
    - Config: <root>/configs/<exp_name>/
    - Log: <root>/logs/<exp_name>/
- 自动写入 build.done 标记，后续自动跳过 build
- 支持 --force-build 强制重建
- 支持 grid search: MaxCheck × InternalResultNum

用法示例：

  # 构建所有实验（已建好的会自动跳过）
  python run_spann_experiments.py --phase build

  # 强制重建所有索引
  python run_spann_experiments.py --phase build --force-build

  # 只跑搜索（会警告没 build 的实验）
  python run_spann_experiments.py --phase search

  # 只跑某几个实验（名字见下面 Experiment.name）
  python run_spann_experiments.py --phase both --exps sift1m_q16B_lxh

  # dry-run 看命令和路径，不实际运行
  python run_spann_experiments.py --phase both --dry-run
"""

import argparse
import itertools
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


# ========= 需要你改/确认的部分：binary 路径 =========

# Path to your SPANN binary
BINARY_PATH = "/root/paper/DiskAnnPQ/SPTAG/Release/ssdserving"


# ========= Dataset / Experiment 定义 =========

@dataclass
class Dataset:
    """
    每个数据集的信息。
    root_dir 决定本数据集下 quant / spann / configs / logs 的根目录。
    """
    name: str
    root_dir: str

    value_type: str
    dim: int

    full_vector_path: str
    full_vector_type: str
    query_path: str
    query_type: str
    truth_path: str
    truth_type: str


@dataclass
class Experiment:
    """
    一组构建实验由 (dataset, quant_size, quant_type) 唯一确定。

    量化器 & 量化码路径根据这三个量自动算，不需要你手写。
    搜索实验再叠加 (MaxCheck, InternalResultNum) 形成 grid search。
    """
    dataset: str
    quant_size: int      # quant{quant_size}B
    quant_type: str      # 用在 OQ_{quant_type}.spannvec / O_{quant_type}.spannvec 里

    # grid for search
    maxcheck_list: List[int] = field(default_factory=lambda: [2048])
    internal_list: List[int] = field(default_factory=lambda: [1000])

    # 可选：覆盖 build / search 里的某些 config 参数
    # key 形式: "Section.Param"，例如 "BuildHead.NumberOfThreads"
    build_overrides: Dict[str, str] = field(default_factory=dict)
    search_overrides: Dict[str, str] = field(default_factory=dict)

    @property
    def name(self) -> str:
        """
        实验名，用在：
        - IndexDirectory: <root>/spann/<name>
        - configs: <root>/configs/<name>/
        - logs: <root>/logs/<name>/
        """
        return f"{self.dataset}_q{self.quant_size}B_{self.quant_type}"


# ====== 定义数据集 ======

DATASETS: Dict[str, Dataset] = {
    "sift1m": Dataset(
        name="sift1m",
        root_dir="/root/paper/sift1m",  # 这一行是关键
        value_type="Float",
        dim=128,
        full_vector_path="/root/paper/sift1m/sift_base.fvecs",
        full_vector_type="XVEC",
        query_path="/root/paper/sift1m/sift_query.fvecs",
        query_type="XVEC",
        truth_path="/root/paper/sift1m/sift_query_gt100",
        truth_type="DEFAULT",
    ),
    "sift1b": Dataset(
        name="sift1b",
        root_dir="/root/paper/sift1b",
        value_type="UInt8",
        dim=128,
        full_vector_path="/root/paper/sift1b/bigann_base.bvecs",
        full_vector_type="XVEC",
        query_path="/root/paper/sift1b/bigann_query.bvecs",
        query_type="XVEC",
        truth_path="/root/paper/sift1b/sift_query_learn_gt100",
        truth_type="DEFAULT",
    ),
    "sift100m": Dataset(
        name="sift100m",
        root_dir="/root/paper/sift100m",
        # value_type="UInt8",
        value_type="Float",
        dim=128,
        # full_vector_path="/root/paper/sift100m/learn.100M.u8bin",
        full_vector_path="/root/paper/sift100m/learn.100M.fbin",
        full_vector_type="DEFAULT",
        query_path="/root/paper/sift100m/query.public.1K.fbin",
        # query_path="/root/paper/sift100m/query.public.10K.u8bin",
        query_type="DEFAULT",
        truth_path="/root/paper/sift100m/sift_query_learn_1kquery_gt100",
        truth_type="DEFAULT",
    ),
    # 如果有别的数据集，在这里再加，例如:
    # "deep1b": Dataset(
    #     name="deep1b",
    #     root_dir="/root/paper/deep1b",
    #     ...
    # )
}


# ====== 定义所有实验（只需要列出 quant_size / quant_type） ======

EXPERIMENTS_LIST: List[Experiment] = []

quant_sizes = [
    16, 
    32, 64, 
    # 128
    ]
quant_types = [
    "RabitQ", 
    # "LSQ", 
    "PQQuantizer"
]
maxcheck_options = [512, 2048]
internal_options = [i for i in range(10, 201, 10)] + [i for i in range(200, 1201, 100)] + [i for i in range(1500, 3001, 500)]
for qs in quant_sizes:
    for qt in quant_types:
        # if qt == "LSQ":
        if qt == "LSQ" and qs > 32:
            continue  # LSQ 不跑大于 32B 的
        EXPERIMENTS_LIST.append(
            Experiment(
                dataset="sift100m",
                quant_size=qs,
                quant_type=qt,
                maxcheck_list=maxcheck_options,
                internal_list=internal_options,
                build_overrides={"BuildHead.NumberOfThreads": "124"},
                search_overrides={"SearchSSDIndex.NumberOfThreads": "32"},
            )
        )


# 方便通过名字 --exps 指定
EXPERIMENTS: Dict[str, Experiment] = {e.name: e for e in EXPERIMENTS_LIST}


# ====== 默认 config section 模板（和你给的类似） ======

DEFAULT_BUILD_SECTIONS: Dict[str, Dict[str, str]] = {
    "SelectHead": {
        "isExecute": "true",
        "TreeNumber": "1",
        "BKTKmeansK": "32",
        "BKTLeafSize": "8",
        "SamplesNumber": "1000",
        "SaveBKT": "false",
        "SelectThreshold": "10",
        "SplitFactor": "6",
        "SplitThreshold": "25",
        "Ratio": "0.1",
        "NumberOfThreads": "124",
        "BKTLambdaFactor": "0",
    },
    "BuildHead": {
        "isExecute": "true",
        "NeighborhoodSize": "32",
        "TPTNumber": "32",
        "TPTLeafSize": "2000",
        "MaxCheck": "16324",
        "MaxCheckForRefineGraph": "16324",
        "RefineIterations": "3",
        "NumberOfThreads": "124",
        "BKTLambdaFactor": "0",
    },
    "BuildSSDIndex": {
        "isExecute": "true",
        "BuildSsdIndex": "true",
        "InternalResultNum": "64",
        "ReplicaCount": "8",
        "PostingPageLimit": "3",
        "NumberOfThreads": "124",
        "MaxCheck": "16324",
        "TmpDir": "/tmp/",
    },
}

DEFAULT_SEARCH_SECTION: Dict[str, str] = {
    "isExecute": "true",
    "BuildSsdIndex": "false",
    "InternalResultNum": "1000",
    "NumberOfThreads": "100",
    "HashTableExponent": "4",
    "ResultNum": "10",
    "MaxCheck": "2048",
    "MaxDistRatio": "8.0",
    "SearchPostingPageLimit": "12",
    "Rerank": "10",
}


# ====== 路径辅助函数 ======

def dataset_root(ds: Dataset) -> Path:
    return Path(ds.root_dir)


def get_quant_dir(ds: Dataset, exp: Experiment) -> Path:
    """
    /root/paper/<dataset>/quant{quant_size}B
    """
    return dataset_root(ds) / f"quant{exp.quant_size}B"


def get_vector_path(ds: Dataset, exp: Experiment) -> str:
    """
    O 是量化后的 base 向量:
    <root>/quant{quant_size}B/O_{quant_type}.spannvec
    """
    return str(get_quant_dir(ds, exp) / f"O_{exp.quant_type}.spannvec")


def get_quantizer_path(ds: Dataset, exp: Experiment) -> str:
    """
    OQ 是量化器:
    <root>/quant{quant_size}B/OQ_{quant_type}.spannvec
    """
    return str(get_quant_dir(ds, exp) / f"OQ_{exp.quant_type}.spannvec")


def get_index_dir(ds: Dataset, exp: Experiment) -> Path:
    """
    <root>/spann/<exp_name>
    """
    return dataset_root(ds) / "spann" / exp.name


def get_config_dir(ds: Dataset, exp: Experiment) -> Path:
    """
    <root>/configs/<exp_name>/
    """
    return dataset_root(ds) / "configs" / exp.name


def get_log_dir(ds: Dataset, exp: Experiment) -> Path:
    """
    <root>/logs/<exp_name>/
    """
    return dataset_root(ds) / "logs" / exp.name


def get_build_marker_path(ds: Dataset, exp: Experiment) -> Path:
    """
    <root>/logs/<exp_name>/build.done
    """
    return get_log_dir(ds, exp) / "build.done"


# ====== config 渲染辅助 ======

def _apply_overrides(section_name: str, params: Dict[str, str], overrides: Dict[str, str]) -> Dict[str, str]:
    """Apply Section.Param overrides to a param dict and return a copy."""
    out = dict(params)
    for key, val in overrides.items():
        try:
            sec, param = key.split(".", 1)
        except ValueError:
            continue
        if sec == section_name:
            out[param] = str(val)
    return out


def render_config(
    ds: Dataset,
    exp: Experiment,
    phase: str,
    maxcheck: Optional[int] = None,
    internal: Optional[int] = None,
) -> str:
    """
    phase: "build" or "search".
    搜索阶段需要提供 maxcheck / internal。
    """
    if phase not in {"build", "search"}:
        raise ValueError(f"Unknown phase: {phase}")

    lines: List[str] = []

    # ---- [Base] ----
    vec_path = get_vector_path(ds, exp)
    quantizer_path = get_quantizer_path(ds, exp)
    index_dir = get_index_dir(ds, exp)

    lines.append("[Base]")
    base_params = {
        "ValueType": ds.value_type,
        "DistCalcMethod": "L2",
        "IndexAlgoType": "BKT",
        "Dim": str(ds.dim),
        "FullVectorPath": ds.full_vector_path,
        "FullVectorType": ds.full_vector_type,
        "VectorPath": vec_path,
        "VectorType": "DEFAULT",  # 量化后的 spannvec，一般用 DEFAULT
        "QueryPath": ds.query_path,
        "QueryType": ds.query_type,
        "TruthPath": ds.truth_path,
        "TruthType": ds.truth_type,
        "IndexDirectory": str(index_dir),
        "QuantizerFilePath": quantizer_path,
    }
    for k, v in base_params.items():
        lines.append(f"{k}={v}")
    lines.append("")

    if phase == "build":
        # ---- 构建阶段：3 个 build section 不注释 ----
        for sec_name in ("SelectHead", "BuildHead", "BuildSSDIndex"):
            params = DEFAULT_BUILD_SECTIONS[sec_name]
            params = _apply_overrides(sec_name, params, exp.build_overrides)
            lines.append(f"[{sec_name}]")
            for k, v in params.items():
                lines.append(f"{k}={v}")
            lines.append("")

        # SearchSSDIndex 在 build 配置里默认不执行
        search_params = _apply_overrides("SearchSSDIndex", DEFAULT_SEARCH_SECTION, exp.search_overrides)
        search_params = dict(search_params)
        search_params["isExecute"] = "false"
        lines.append("[SearchSSDIndex]")
        for k, v in search_params.items():
            lines.append(f"{k}={v}")
        lines.append("")

    else:  # phase == "search"
        if maxcheck is None or internal is None:
            raise ValueError("For phase='search', maxcheck and internal must be provided")

        # ---- 搜索阶段：3 个 build section 全部注释掉 ----
        for sec_name in ("SelectHead", "BuildHead", "BuildSSDIndex"):
            params = DEFAULT_BUILD_SECTIONS[sec_name]
            lines.append(f"; [{sec_name}]")
            for k, v in params.items():
                lines.append(f"; {k}={v}")
            lines.append(";")
            lines.append("")

        # ---- 真正执行的 SearchSSDIndex ----
        search_params = _apply_overrides("SearchSSDIndex", DEFAULT_SEARCH_SECTION, exp.search_overrides)
        search_params = dict(search_params)
        search_params["MaxCheck"] = str(maxcheck)
        search_params["InternalResultNum"] = str(internal)

        lines.append("[SearchSSDIndex]")
        for k, v in search_params.items():
            lines.append(f"{k}={v}")
        lines.append("")

    return "\n".join(lines)


# ====== 执行 binary 并写日志 ======

def run_one(binary_path: str, config_path: Path, log_path: Path, dry_run: bool = False) -> int:
    """Run binary with a given config, log output, and return the exit code."""
    cmd = [binary_path, str(config_path)]
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print(f"[DRY-RUN] CMD: {' '.join(cmd)}")
        print(f"[DRY-RUN] Config: {config_path}")
        print(f"[DRY-RUN] Log will be: {log_path}")
        return 0

    print(f"[RUN] {' '.join(cmd)}")

    # Launch subprocess and stream output to both stdout and log file
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"# CMD: {' '.join(cmd)}\n")
        f.write(f"# TIME_START: {datetime.now().isoformat()}\n\n")
        f.flush()

        # Start process
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, text=True)

        # Stream output line by line
        if proc.stdout is not None:
            try:
                for line in proc.stdout:
                    # Print to console immediately
                    print(line, end="")
                    # Write to log
                    f.write(line)
            except Exception:
                # In case of any streaming errors, fall back to waiting for process
                pass

        proc.wait()
        return_code = proc.returncode

        f.write(f"\n# RETURN_CODE: {return_code}\n")
        f.write(f"# TIME_END: {datetime.now().isoformat()}\n")

    print(f"[LOG] Saved to {log_path} (ret={return_code})")
    if return_code != 0:
        print(f"[WARN] Non-zero return code for {config_path}")

    return return_code


# ====== 高层执行逻辑 ======

def get_selected_experiments(names: List[str]) -> List[Experiment]:
    if not names or names == ["all"]:
        return list(EXPERIMENTS.values())
    out: List[Experiment] = []
    for n in names:
        if n not in EXPERIMENTS:
            raise KeyError(f"Unknown experiment name: {n}")
        out.append(EXPERIMENTS[n])
    return out


def run_build_phase(
    exps: List[Experiment],
    binary_path: str,
    dry_run: bool = False,
    force_build: bool = False,
) -> None:
    for exp in exps:
        ds = DATASETS[exp.dataset]
        marker = get_build_marker_path(ds, exp)

        if marker.exists() and not force_build:
            print(f"[SKIP-BUILD] {ds.name}/{exp.name} already built (marker: {marker})")
            continue

        cfg_text = render_config(ds, exp, phase="build")
        cfg_dir = get_config_dir(ds, exp)
        cfg_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = cfg_dir / "build.cfg"
        cfg_path.write_text(cfg_text, encoding="utf-8")

        # 确保索引目录存在
        index_dir = get_index_dir(ds, exp)
        index_dir.mkdir(parents=True, exist_ok=True)
        print(f"[BUILD] Making index dir: {index_dir}")

        import time
        timestr = time.strftime("%Y%m%d_%H%M%S")
        log_dir = get_log_dir(ds, exp)
        log_path = log_dir / f"build_{timestr}.log"
        print(f"[BUILD] Log path: {log_path}, Now building...")
        ret = run_one(binary_path, cfg_path, log_path, dry_run=dry_run)
        print(f"[BUILD] Finished building {ds.name}/{exp.name} with ret={ret}")

        if dry_run:
            # dry-run 不写 marker
            continue

        if ret == 0:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(
                f"BUILD_OK {datetime.now().isoformat()}\n"
                f"INDEX_DIR={index_dir}\n",
                encoding="utf-8",
            )
            print(f"[MARK] Build marker written: {marker}")
        else:
            if marker.exists():
                marker.unlink()
            print(f"[MARK] Build failed, marker removed (if existed): {marker}")


def run_search_phase(exps: List[Experiment], binary_path: str, dry_run: bool = False) -> None:
    for exp in exps:
        ds = DATASETS[exp.dataset]
        marker = get_build_marker_path(ds, exp)
        if not marker.exists():
            print(
                f"[WARN] Build marker not found for {ds.name}/{exp.name}. "
                f"Make sure index exists in {get_index_dir(ds, exp)}."
            )

        for maxcheck, internal in itertools.product(exp.maxcheck_list, exp.internal_list):
            cfg_text = render_config(
                ds,
                exp,
                phase="search",
                maxcheck=maxcheck,
                internal=internal,
            )

            cfg_dir = get_config_dir(ds, exp)
            cfg_dir.mkdir(parents=True, exist_ok=True)
            cfg_name = f"search_mc{maxcheck}_ir{internal}.cfg"
            cfg_path = cfg_dir / cfg_name
            cfg_path.write_text(cfg_text, encoding="utf-8")
            
            import time
            timestr = time.strftime("%Y%m%d_%H%M%S")

            log_dir = get_log_dir(ds, exp)
            log_time_str = datetime.now().strftime("%Y%m%d_%H%M%S_")
            log_path = log_dir / (log_time_str + f"search_mc{maxcheck}_ir{internal}_{timestr}.log")
            run_one(binary_path, cfg_path, log_path, dry_run=dry_run)


# ====== CLI ======

def main() -> None:
    parser = argparse.ArgumentParser(description="Run SPANN index build + search experiments.")
    parser.add_argument(
        "--phase",
        choices=["build", "search", "both"],
        default="both",
        help="Which phase to run.",
    )
    parser.add_argument(
        "--exps",
        nargs="*",
        default=["all"],
        help="Experiment names to run, or 'all'. (names come from Experiment.name)",
    )
    parser.add_argument(
        "--binary",
        default=BINARY_PATH,
        help="Path to binary (default is BINARY_PATH in this script).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not actually run binary, just print commands and paths.",
    )
    parser.add_argument(
        "--force-build",
        action="store_true",
        help="Ignore build markers and rebuild indexes even if they appear done.",
    )

    args = parser.parse_args()
    binary_path = args.binary

    selected_exps = get_selected_experiments(args.exps)

    if args.phase in {"build", "both"}:
        run_build_phase(
            selected_exps,
            binary_path,
            dry_run=args.dry_run,
            force_build=args.force_build,
        )

    if args.phase in {"search", "both"}:
        run_search_phase(selected_exps, binary_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
