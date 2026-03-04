#!/usr/bin/env python3
"""
Search-only runner for prebuilt SPANN/SPTAG SSD indexes.

目标
----
- 基于“别人离线构建好的 IndexDirectory”，只改 SearchSSDIndex 参数做搜索
- 不引入“量化”概念：实验 = 数据集 + 预构建索引（index_dir）+ 基础配置模板（base_cfg）
- 每次运行生成：
  - configs/<exp_name>/search_mc{MaxCheck}_ir{InternalResultNum}.cfg
  - logs/<exp_name>/{timestamp}_search_mc{...}_ir{...}.log

你需要做的事情
--------------
1) 填好 BINARY_PATH
2) 在 DATASETS / INDEX_SPECS 里登记你已有的预构建索引：
   - dataset: 选择数据集名（用于确定 root_dir 放 configs/logs）
   - name: 实验名（目录名）
   - index_dir: 预构建索引目录（真实存在）
   - base_cfg: 一个“基础配置模板”，里面 Base/Query/Truth/VectorPath/QuantizerFilePath 等都已经正确
3) 运行：
   python run_spann_search_prebuilt.py --exps all
   python run_spann_search_prebuilt.py --exps sift100m_spann_xxx --maxcheck 100 1000 --internal 100 200 500
"""

from __future__ import annotations

import argparse
import itertools
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ========= 需要你改的：binary 路径 =========
BINARY_PATH = "/root/paper/DiskAnnPQ/SPANN-original/SPTAG/Release/ssdserving"

# ========= Search-only：我们会强制做这些安全修改 =========
BUILD_SECTIONS_TO_DISABLE = ["SelectHead", "BuildHead", "BuildSSDIndex"]

# 你也可以在 --set 里改更多 SearchSSDIndex 参数
DEFAULT_FORCE_SEARCH_KV: Dict[str, str] = {
    "isExecute": "true",
    "BuildSsdIndex": "false",
}

# ====== 数据集 & 索引定义（无量化概念）======

@dataclass
class Dataset:
    name: str
    root_dir: str  # 用于放 configs/logs 的根目录（不强依赖数据文件路径）


@dataclass
class IndexSpec:
    """
    一个预构建索引实验。
    - base_cfg: 基础配置模板（必须包含正确的 Base 路径、Query/Truth、QuantizerFilePath、VectorPath 等）
    - index_dir: 预构建好的 IndexDirectory（真实索引文件所在）
    """
    dataset: str
    name: str

    index_dir: str
    base_cfg: str

    # grid
    maxcheck_list: List[int] = field(default_factory=lambda: [2048])
    internal_list: List[int] = field(default_factory=lambda: [1000])

    # 默认搜索覆盖（等价于 --set SearchSSDIndex.X=Y 的“按索引固化版本”）
    search_overrides: Dict[str, str] = field(default_factory=dict)

    # 是否强制将 build sections isExecute=false（推荐一直 true）
    disable_build_sections: bool = True


# ====== 你在这里登记数据集（只用于 root_dir）======

DATASETS: Dict[str, Dataset] = {
    "sift1m": Dataset(name="sift1m", root_dir="/root/paper/sift1m"),
    "sift1b": Dataset(name="sift1b", root_dir="/root/paper/sift1b"),
    "sift100m": Dataset(name="sift100m", root_dir="/root/paper/sift100m"),
    "laion50m512d": Dataset(name="laion50m512d", root_dir="/root/paper/laion100m"),
    "laion50m768d": Dataset(name="laion50m768d", root_dir="/root/paper/laion100m"),
    "t2i100m": Dataset(name="t2i100m", root_dir="/root/paper/text2image1b"),
    "deep100m": Dataset(name="deep100m", root_dir="/root/paper/deep1b"),
    "spacev1b": Dataset(name="spacev1b", root_dir="/root/paper/spacev1b"),
}

# ====== 你在这里登记“别人已经建好的索引”======

INDEX_SPECS_LIST: List[IndexSpec] = [
    # 示例（你改成自己的真实路径）
    # IndexSpec(
    #     dataset="laion50m512d",
    #     name="laion50m512dood_spann",
    #     index_dir="/root/paper/laion100m/spann-original512dood",         # 预构建索引目录（里面应有 ssd index 文件）
    #     base_cfg="/root/paper/DiskAnnPQ/SPANN-original/config.laion50m512d.ini",    # 基础配置模板（Base/Query/Truth/...写对）
    #     maxcheck_list=[2048],
    #     internal_list=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 400, 500, 600, 800],
    #     search_overrides={"NumberOfThreads": "32", "Rerank": "10", "ResultNum": "10"},
    # ),
    # IndexSpec(
    #     dataset="laion50m768d",
    #     name="laion50m768d_spann",
    #     index_dir="/root/paper/laion100m/spann-original768dnew",         # 预构建索引目录（里面应有 ssd index 文件）
    #     base_cfg="/root/paper/DiskAnnPQ/SPANN-original/config.laion50m768d.ini",    # 基础配置模板（Base/Query/Truth/...写对）
    #     maxcheck_list=[2048],
    #     internal_list=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 400, 500, 600, 800],
    #     search_overrides={"NumberOfThreads": "32", "Rerank": "10", "ResultNum": "10"},
    # ),
    # IndexSpec(
    #     dataset="laion50m512d",
    #     name="laion50m512dood_spann",
    #     index_dir="/root/paper/laion100m/spann-original512d",         # 预构建索引目录（里面应有 ssd index 文件）
    #     base_cfg="/root/paper/DiskAnnPQ/SPANN-original/config.laion50m512d.ini",    # 基础配置模板（Base/Query/Truth/...写对）
    #     maxcheck_list=[2048],
    #     internal_list=[100, 200, 400, 800, 1000, 1200, 1600, 2000, 4000, 8000],
    #     search_overrides={"NumberOfThreads": "32", "Rerank": "10", "ResultNum": "10"},
    # ),
    # IndexSpec(
    #     dataset="laion50m512d",
    #     name="laion50m512dood_spann_16t",
    #     index_dir="/root/paper/laion100m/spann-original512d",         # 预构建索引目录（里面应有 ssd index 文件）
    #     base_cfg="/root/paper/DiskAnnPQ/SPANN-original/config.laion50m512d.ini",    # 基础配置模板（Base/Query/Truth/...写对）
    #     maxcheck_list=[8000],
    #     internal_list=[100, 200, 400, 800, 1000, 1200, 1600, 2000, 4000, 8000],
    #     search_overrides={"NumberOfThreads": "16", "Rerank": "10", "ResultNum": "10"},
    # ),
    # IndexSpec(
    #     dataset="laion50m512d",
    #     name="laion50m512dood_spann_64t",
    #     index_dir="/root/paper/laion100m/spann-original512d",         # 预构建索引目录（里面应有 ssd index 文件）
    #     base_cfg="/root/paper/DiskAnnPQ/SPANN-original/config.laion50m512d.ini",    # 基础配置模板（Base/Query/Truth/...写对）
    #     maxcheck_list=[2048],
    #     internal_list=[100, 200, 400, 800, 1000, 1200, 1600, 2000, 4000, 8000],
    #     search_overrides={"NumberOfThreads": "64", "Rerank": "10", "ResultNum": "10"},
    # ),
    # IndexSpec(
    #     dataset="laion50m512d",
    #     name="laion50m512dood_spann_128t",
    #     index_dir="/root/paper/laion100m/spann-original512d",         # 预构建索引目录（里面应有 ssd index 文件）
    #     base_cfg="/root/paper/DiskAnnPQ/SPANN-original/config.laion50m512d.ini",    # 基础配置模板（Base/Query/Truth/...写对）
    #     maxcheck_list=[2048],
    #     internal_list=[100, 200, 400, 800, 1000, 1200, 1600, 2000, 4000, 8000],
    #     search_overrides={"NumberOfThreads": "128", "Rerank": "10", "ResultNum": "10"},
    # ),
    IndexSpec(
        dataset="laion50m768d",
        name="laion50m768d_spann_16t",
        index_dir="/root/paper/laion100m/spann-original768d",         # 预构建索引目录（里面应有 ssd index 文件）
        base_cfg="/root/paper/DiskAnnPQ/SPANN-original/config.laion50m768d.ini",    # 基础配置模板（Base/Query/Truth/...写对）
        maxcheck_list=[2048],
        internal_list=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 400, 500, 600, 800, 1200, 1600, 2000],
        search_overrides={"NumberOfThreads": "16", "Rerank": "10", "ResultNum": "10"},
    ),
    IndexSpec(
        dataset="laion50m768d",
        name="laion50m768d_spann_32t",
        index_dir="/root/paper/laion100m/spann-original768d",         # 预构建索引目录（里面应有 ssd index 文件）
        base_cfg="/root/paper/DiskAnnPQ/SPANN-original/config.laion50m768d.ini",    # 基础配置模板（Base/Query/Truth/...写对）
        maxcheck_list=[2048],
        internal_list=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 400, 500, 600, 800, 1200, 1600, 2000],
        search_overrides={"NumberOfThreads": "32", "Rerank": "10", "ResultNum": "10"},
    ),
    IndexSpec(
        dataset="laion50m768d",
        name="laion50m768d_spann_64t",
        index_dir="/root/paper/laion100m/spann-original768d",         # 预构建索引目录（里面应有 ssd index 文件）
        base_cfg="/root/paper/DiskAnnPQ/SPANN-original/config.laion50m768d.ini",    # 基础配置模板（Base/Query/Truth/...写对）
        maxcheck_list=[2048],
        internal_list=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 400, 500, 600, 800, 1200, 1600, 2000],
        search_overrides={"NumberOfThreads": "64", "Rerank": "10", "ResultNum": "10"},
    ),
    IndexSpec(
        dataset="laion50m768d",
        name="laion50m768d_spann_128t",
        index_dir="/root/paper/laion100m/spann-original768d",         # 预构建索引目录（里面应有 ssd index 文件）
        base_cfg="/root/paper/DiskAnnPQ/SPANN-original/config.laion50m768d.ini",    # 基础配置模板（Base/Query/Truth/...写对）
        maxcheck_list=[2048],
        internal_list=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 400, 500, 600, 800, 1200, 1600, 2000],
        search_overrides={"NumberOfThreads": "128", "Rerank": "10", "ResultNum": "10"},
    ),
    # IndexSpec(
    #     dataset="t2i100m",
    #     name="t2i100m_spann",
    #     index_dir="/root/paper/text2image1b/spann-original",
    #     base_cfg="/root/paper/DiskAnnPQ/SPANN-original/config.t2i100m.ini",
    #     maxcheck_list=[2048],
    #     internal_list=[10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 250, 300, 400, 500, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
    #     search_overrides={"NumberOfThreads": "32", "Rerank": "10", "ResultNum": "10"},
    # ),
    # IndexSpec(
    #     dataset="deep100m",
    #     name="deep100m_spann",
    #     index_dir="/root/paper/deep1b/spann-original",
    #     base_cfg="/root/paper/DiskAnnPQ/SPANN-original/config.deep100m.ini",
    #     maxcheck_list=[512, 2048],
    #     internal_list=[10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 250, 300, 400, 500, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
    #     search_overrides={"NumberOfThreads": "32", "Rerank": "10", "ResultNum": "10"},
    # ),
    # IndexSpec(
    #     dataset="spacev1b",
    #     name="spacev1b_spann",
    #     index_dir="/root/paper/spacev1b/spann-original",
    #     base_cfg="/root/paper/DiskAnnPQ/SPANN-original/config.spacev1b.ini",
    #     maxcheck_list=[2048],
    #     internal_list=[10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 250, 300, 400, 500, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
    #     search_overrides={"NumberOfThreads": "32", "Rerank": "10", "ResultNum": "10"},
    # ),
    # IndexSpec(
    #     dataset="sift1b",
    #     name="sift1b_spann",
    #     index_dir="/root/paper/sift1b/spann-original",
    #     base_cfg="/root/paper/DiskAnnPQ/SPANN-original/config.sift1b.ini",
    #     maxcheck_list=[2048],
    #     internal_list=[10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 250, 300, 400, 500, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
    #     search_overrides={"NumberOfThreads": "32", "Rerank": "10", "ResultNum": "10"},
    # ),
    # IndexSpec(
    #     dataset="deep100m",
    #     name="deep100m_spann",
    #     index_dir="/root/paper/deep1b/spann-original",
    #     base_cfg="/root/paper/DiskAnnPQ/SPANN-original/config.deep100m.ini",
    #     maxcheck_list=[512, 2048],
    #     internal_list=[10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 250, 300, 400, 500, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
    #     search_overrides={"NumberOfThreads": "32", "Rerank": "10", "ResultNum": "10"},
    # ),
    # IndexSpec(
    #     dataset="spacev1b",
    #     name="spacev1b_spann",
    #     index_dir="/root/paper/spacev1b/spann-original",
    #     base_cfg="/root/paper/DiskAnnPQ/SPANN-original/config.spacev1b.ini",
    #     maxcheck_list=[512, 2048],
    #     internal_list=[10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 250, 300, 400, 500, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
    #     search_overrides={"NumberOfThreads": "32", "Rerank": "10", "ResultNum": "10"},
    # ),
]

INDEX_SPECS: Dict[str, IndexSpec] = {s.name: s for s in INDEX_SPECS_LIST}


# ===================== cfg 解析 & patch =====================

_SECTION_RE = re.compile(r"^\s*\[(?P<name>[^\]]+)\]\s*$")
_KV_RE = re.compile(r"^\s*(?P<k>[^=;#\s][^=]*)\s*=\s*(?P<v>.*)\s*$")

def _is_comment_line(line: str) -> bool:
    s = line.strip()
    return s.startswith(";") or s.startswith("#")

def parse_cfg(text: str) -> Tuple[List[str], Dict[str, List[str]], List[str]]:
    """
    轻量解析：
    - 返回 (preamble_lines, sections{name->raw_lines}, section_order)
    - 不保留“注释 section”（如 '; [BuildHead]'），但保留其中的行在 raw_lines 里没意义，所以忽略。
    """
    preamble: List[str] = []
    sections: Dict[str, List[str]] = {}
    order: List[str] = []

    cur: Optional[str] = None
    for line in text.splitlines():
        # 忽略被注释掉的 section header（以 ; 开头）
        if line.lstrip().startswith(";"):
            if cur is None:
                preamble.append(line)
            else:
                sections[cur].append(line)
            continue

        m = _SECTION_RE.match(line)
        if m:
            cur = m.group("name").strip()
            if cur not in sections:
                sections[cur] = []
                order.append(cur)
            continue

        if cur is None:
            preamble.append(line)
        else:
            sections[cur].append(line)

    return preamble, sections, order

def set_kv_in_section(lines: List[str], key: str, value: str) -> List[str]:
    """
    在某个 section 的 raw_lines 中：
    - 替换所有出现的 key=...（非注释行）
    - 若不存在则追加一行 key=value
    """
    out: List[str] = []
    found = False
    for line in lines:
        if _is_comment_line(line):
            out.append(line)
            continue
        m = _KV_RE.match(line)
        if m and m.group("k").strip() == key:
            out.append(f"{key}={value}")
            found = True
        else:
            out.append(line)

    if not found:
        # 保持风格：section 末尾追加
        out.append(f"{key}={value}")
    return out

def patch_cfg_for_search(
    base_cfg_text: str,
    index_dir: str,
    disable_build_sections: bool,
    maxcheck: int,
    internal: int,
    search_kv: Dict[str, str],
) -> str:
    """
    基于 base_cfg_text：
    1) Base.IndexDirectory 强制指向 index_dir（避免 template 写错）
    2) build sections: isExecute=false（可选）
    3) SearchSSDIndex: isExecute=true + BuildSsdIndex=false + MaxCheck/InternalResultNum + 用户覆盖
    """
    preamble, sections, order = parse_cfg(base_cfg_text)

    # ---- Base：强制 IndexDirectory ----
    if "Base" not in sections:
        raise ValueError("Base section not found in base_cfg")
    sections["Base"] = set_kv_in_section(sections["Base"], "IndexDirectory", index_dir)

    # ---- Disable build sections ----
    if disable_build_sections:
        for sec in BUILD_SECTIONS_TO_DISABLE:
            if sec in sections:
                sections[sec] = set_kv_in_section(sections[sec], "isExecute", "false")

    # ---- SearchSSDIndex section ----
    if "SearchSSDIndex" not in sections:
        sections["SearchSSDIndex"] = []
        order.append("SearchSSDIndex")

    merged = dict(DEFAULT_FORCE_SEARCH_KV)
    merged.update(search_kv)
    merged["MaxCheck"] = str(max(maxcheck, internal))
    merged["InternalResultNum"] = str(internal)

    # 写入 merged kv
    for k, v in merged.items():
        sections["SearchSSDIndex"] = set_kv_in_section(sections["SearchSSDIndex"], k, str(v))

    # ---- Render back ----
    out_lines: List[str] = []
    out_lines.extend(preamble)
    if out_lines and out_lines[-1].strip() != "":
        out_lines.append("")

    for sec in order:
        out_lines.append(f"[{sec}]")
        out_lines.extend(sections.get(sec, []))
        out_lines.append("")

    return "\n".join(out_lines).rstrip() + "\n"


# ===================== 路径辅助 =====================

def dataset_root(ds: Dataset) -> Path:
    return Path(ds.root_dir)

def get_config_dir(ds: Dataset, spec: IndexSpec) -> Path:
    return dataset_root(ds) / "configs" / spec.name

def get_log_dir(ds: Dataset, spec: IndexSpec) -> Path:
    return dataset_root(ds) / "logs" / spec.name


# ===================== 执行 binary 并写日志 =====================

def run_one(binary_path: str, config_path: Path, log_path: Path, dry_run: bool = False) -> int:
    cmd = [binary_path, str(config_path)]
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print(f"[DRY-RUN] CMD: {' '.join(cmd)}")
        print(f"[DRY-RUN] Config: {config_path}")
        print(f"[DRY-RUN] Log: {log_path}")
        return 0

    print(f"[RUN] {' '.join(cmd)}")
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"# CMD: {' '.join(cmd)}\n")
        f.write(f"# TIME_START: {datetime.now().isoformat()}\n\n")
        f.flush()

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, text=True)
        if proc.stdout is not None:
            for line in proc.stdout:
                print(line, end="")
                f.write(line)

        proc.wait()
        ret = proc.returncode
        f.write(f"\n# RETURN_CODE: {ret}\n")
        f.write(f"# TIME_END: {datetime.now().isoformat()}\n")

    print(f"[LOG] {log_path} (ret={ret})")
    return ret


# ===================== 选择实验 =====================

def get_selected_specs(names: List[str]) -> List[IndexSpec]:
    if not names or names == ["all"]:
        return list(INDEX_SPECS.values())
    out: List[IndexSpec] = []
    for n in names:
        if n not in INDEX_SPECS:
            raise KeyError(f"Unknown exp name: {n}")
        out.append(INDEX_SPECS[n])
    return out


# ===================== CLI main =====================

def main() -> None:
    parser = argparse.ArgumentParser(description="Search-only runner for prebuilt SPANN/SPTAG SSD indexes.")
    parser.add_argument("--binary", default=BINARY_PATH, help="Path to ssdserving binary.")
    parser.add_argument("--exps", nargs="*", default=["all"], help="Experiment names to run, or 'all'.")
    parser.add_argument("--dry-run", action="store_true", help="Only print commands/paths, do not run.")

    # 可全局覆盖 grid（如果不传，就用 IndexSpec 自带的列表）
    parser.add_argument("--maxcheck", nargs="*", type=int, default=[], help="Override MaxCheck list for ALL selected exps.")
    parser.add_argument("--internal", nargs="*", type=int, default=[], help="Override InternalResultNum list for ALL selected exps.")

    # 额外覆盖 SearchSSDIndex 参数（全局）
    # 用法：--set NumberOfThreads=100 --set Rerank=10
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        help="Extra SearchSSDIndex overrides, e.g. --set NumberOfThreads=100 Rerank=10 ResultNum=10",
    )

    args = parser.parse_args()

    binary = args.binary
    specs = get_selected_specs(args.exps)

    # parse --set
    cli_search_overrides: Dict[str, str] = {}
    for item in args.set:
        if "=" not in item:
            raise ValueError(f"Bad --set item (expect k=v): {item}")
        k, v = item.split("=", 1)
        cli_search_overrides[k.strip()] = v.strip()

    for spec in specs:
        if spec.dataset not in DATASETS:
            raise KeyError(f"Dataset not found in DATASETS: {spec.dataset}")

        ds = DATASETS[spec.dataset]

        index_dir = Path(spec.index_dir)
        if not index_dir.exists():
            print(f"[WARN] index_dir not found: {index_dir} (still will try run)")
        base_cfg_path = Path(spec.base_cfg)
        if not base_cfg_path.exists():
            raise FileNotFoundError(f"base_cfg not found: {base_cfg_path}")

        base_text = base_cfg_path.read_text(encoding="utf-8", errors="replace")

        maxchecks = args.maxcheck if args.maxcheck else spec.maxcheck_list
        internals = args.internal if args.internal else spec.internal_list

        # per-exp overrides: spec.search_overrides + cli overrides
        merged_search_overrides = dict(spec.search_overrides)
        merged_search_overrides.update(cli_search_overrides)

        cfg_dir = get_config_dir(ds, spec)
        log_dir = get_log_dir(ds, spec)
        cfg_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        for mc, ir in itertools.product(maxchecks, internals):
            patched = patch_cfg_for_search(
                base_cfg_text=base_text,
                index_dir=str(index_dir),
                disable_build_sections=spec.disable_build_sections,
                maxcheck=mc,
                internal=ir,
                search_kv=merged_search_overrides,
            )

            cfg_name = f"search_mc{mc}_ir{ir}.cfg"
            cfg_path = cfg_dir / cfg_name
            cfg_path.write_text(patched, encoding="utf-8")

            t = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = log_dir / f"{t}_search_mc{mc}_ir{ir}.log"

            run_one(binary, cfg_path, log_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
