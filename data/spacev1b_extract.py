#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPACEV1B -> fvecs / bvecs / ivecs converter (streaming & sharded).

- vectors.bin/ contains multiple parts: vectors_1.bin has 8-byte header:
    [int32 total_count][int32 dim], followed by raw int8 data.
  Subsequent parts contain contiguous raw int8 bytes (no header).
- query.bin, query_log.bin:
    [int32 count][int32 dim][count*dim int8 bytes]
- truth.bin:
    [int32 q_count][int32 topk][q_count*topk int32 vids][q_count*topk float32 dists]

Outputs:
- Vectors: .fvecs (float32) or .bvecs (uint8, each vector +128 from int8 input).
- Queries / Query log: same as vectors (format selectable).
- Truth: vids -> .ivecs (int32 with per-vector leading dim=topk), dists -> .fvecs.

Author: ChatGPT (GPT-5 Thinking)
"""

import argparse
import os
import re
from pathlib import Path
import struct
import sys
import numpy as np
from typing import Generator, Iterable, List, Tuple, Optional

# -----------------------------
# Utilities
# -----------------------------

def sorted_vector_parts(vectors_dir: Path) -> List[Path]:
    """
    Return [vectors_1.bin, vectors_2.bin, ...] sorted by numeric suffix.
    """
    parts = []
    pat = re.compile(r"^vectors_(\d+)\.bin$")
    for p in vectors_dir.iterdir():
        m = pat.match(p.name)
        if m:
            parts.append((int(m.group(1)), p))
    parts.sort(key=lambda x: x[0])
    return [p for _, p in parts]


def read_total_count_and_dim(first_part: Path) -> Tuple[int, int]:
    """
    Read [int32 count][int32 dim] header from first part.
    """
    with open(first_part, "rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise RuntimeError("vectors_1.bin header is incomplete.")
        total_count = struct.unpack("<i", header[:4])[0]
        dim = struct.unpack("<i", header[4:8])[0]
    if dim <= 0 or total_count <= 0:
        raise ValueError(f"Bad header: count={total_count}, dim={dim}")
    return total_count, dim


def stream_int8_vectors(parts: List[Path], dim: int, chunk_bytes: int = 64 * 1024 * 1024
                       ) -> Generator[np.ndarray, None, None]:
    """
    Stream int8 vectors (N x dim) across multiple part files.
    Skips the first 8 bytes header in vectors_1.bin.

    Yields numpy arrays of shape (n_block, dim), dtype=int8.
    """
    b_per_vec = dim  # int8 per vector
    rem = bytearray()

    for idx, part in enumerate(parts):
        with open(part, "rb") as f:
            if idx == 0:
                # Skip 8-byte header
                _ = f.read(8)
            while True:
                chunk = f.read(chunk_bytes)
                if not chunk:
                    break
                rem.extend(chunk)

                n_vec = len(rem) // b_per_vec
                if n_vec > 0:
                    n_bytes = n_vec * b_per_vec

                    # IMPORTANT: create memoryview, build array FROM it, COPY, then RELEASE
                    mv = memoryview(rem)
                    buf = mv[:n_bytes]                      # a memoryview slice
                    arr = np.frombuffer(buf, dtype=np.int8).reshape(n_vec, dim).copy()

                    # release views BEFORE resizing rem
                    try:
                        buf.release()
                    except AttributeError:
                        pass
                    try:
                        mv.release()
                    except AttributeError:
                        pass

                    # Now safe to resize rem
                    del rem[:n_bytes]

                    # yield the copied block
                    yield arr

    if len(rem) != 0:
        # There should be no leftover if files are well-formed
        raise RuntimeError(f"Leftover bytes ({len(rem)}) not forming full vectors.")



def _writev(fd: int, bufs: List[bytes]) -> None:
    """
    writev if available, else fallback to loop writes.
    """
    if hasattr(os, "writev"):
        os.writev(fd, bufs)
    else:
        for b in bufs:
            os.write(fd, b)

# 放在工具函数区：替换你原来的 _writev，并新增两个小工具
def _get_iov_max() -> int:
    # Linux 通常 1024；不可用时保守取 1024
    try:
        return int(os.sysconf("SC_IOV_MAX"))
    except Exception:
        return 1024

def _writev_chunked(fd: int, bufs: list[bytes], iov_max = None) -> None:
    """writev 分块写，确保每次调用的 iovec 数不超过 IOV_MAX。"""
    if iov_max is None:
        iov_max = _get_iov_max()
    if hasattr(os, "writev"):
        i = 0
        n = len(bufs)
        while i < n:
            j = min(i + iov_max, n)
            os.writev(fd, bufs[i:j])
            i = j
    else:
        # 平台不支持 writev 时的兜底
        for b in bufs:
            os.write(fd, b)

def write_sharded_vectors(
    out_dir: Path,
    base_name: str,
    vec_iter: Iterable[np.ndarray],
    dim: int,
    vectors_per_shard: int = 2_000_000,
    output_format: str = "fvecs",
    writev_batch: int = 256,
) -> List[Path]:
    """
    Write vectors in shards:
      - fvecs: per-vector [int32 dim][dim * float32]
      - bvecs: per-vector [int32 dim][dim * uint8]  with uint8 = int8 + 128

    vec_iter yields blocks of int8 arrays with shape (n_block, dim).
    Returns list of output shard paths.
    """
    assert output_format in ("fvecs", "bvecs")
    out_paths = []
    shard_idx = 0
    within_shard = 0
    fd: Optional[int] = None
    path: Optional[Path] = None

    def open_new_shard():
        nonlocal fd, path, shard_idx, within_shard
        if fd is not None:
            os.close(fd)
        shard_idx += 1
        within_shard = 0
        ext = ".fvecs" if output_format == "fvecs" else ".bvecs"
        fname = f"{base_name}_{shard_idx:06d}{ext}"
        p = out_dir / fname
        # O_BINARY is not required/available on POSIX; guarded get
        flags = os.O_CREAT | os.O_WRONLY | os.O_TRUNC
        if hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY
        fd_local = os.open(str(p), flags, 0o644)
        out_paths.append(p)
        return p, fd_local

    path, fd = open_new_shard()
    header_bytes = struct.pack("<i", dim)

    for block in vec_iter:
        n = block.shape[0]
        start = 0
        while start < n:
            room = vectors_per_shard - within_shard
            take = min(room, n - start)
            sub = block[start:start + take]  # int8 [take, dim]
            start += take

            if output_format == "fvecs":
                # Convert once
                subf = sub.astype(np.float32, copy=False)
                # Batch write using writev
                bufs: List[bytes] = []
                for i in range(subf.shape[0]):
                    bufs.append(header_bytes)
                    bufs.append(subf[i].tobytes(order="C"))
                    if len(bufs) >= 2 * writev_batch:
                        _writev_chunked(fd, bufs); bufs.clear()
                if bufs:
                    _writev_chunked(fd, bufs)
            else:
                # bvecs: add 128, cast to uint8
                # safe path: promote to int16 to avoid wraparound then cast to uint8
                subv = (sub.astype(np.int16) + 128).astype(np.uint8, copy=False)
                bufs: List[bytes] = []
                for i in range(subv.shape[0]):
                    bufs.append(header_bytes)
                    bufs.append(subv[i].tobytes(order="C"))
                    if len(bufs) >= 2 * writev_batch:
                        _writev_chunked(fd, bufs); bufs.clear()
                if bufs:
                    _writev_chunked(fd, bufs);

            within_shard += take
            if within_shard >= vectors_per_shard:
                path, fd = open_new_shard()

    if fd is not None:
        os.close(fd)
    return out_paths


def convert_vectors(
    src_root: Path, out_root: Path,
    vectors_per_shard: int, output_format: str,
    chunk_bytes: int
) -> None:
    vectors_dir = src_root / "vectors.bin"
    parts = sorted_vector_parts(vectors_dir)
    if not parts:
        raise FileNotFoundError(f"No vector parts found in {vectors_dir}")
    total_count, dim = read_total_count_and_dim(parts[0])
    print(f"[vectors] total={total_count:,} dim={dim} parts={len(parts)} format={output_format}")

    iter_blocks = stream_int8_vectors(parts, dim, chunk_bytes=chunk_bytes)
    out_root.mkdir(parents=True, exist_ok=True)
    out_paths = write_sharded_vectors(
        out_dir=out_root,
        base_name="vectors",
        vec_iter=iter_blocks,
        dim=dim,
        vectors_per_shard=vectors_per_shard,
        output_format=output_format
    )
    print(f"[vectors] wrote {len(out_paths)} shards:")
    for p in out_paths:
        print("  -", p)


def convert_query_like_file(
    bin_path: Path, out_path: Path, output_format: str
) -> None:
    with open(bin_path, "rb") as f:
        hdr = f.read(8)
        if len(hdr) != 8:
            raise RuntimeError(f"Incomplete header in {bin_path}")
        count = struct.unpack("<i", hdr[:4])[0]
        dim = struct.unpack("<i", hdr[4:8])[0]
        raw = f.read(count * dim)
        if len(raw) != count * dim:
            raise RuntimeError(f"Data truncated in {bin_path}")
    arr = np.frombuffer(raw, dtype=np.int8).reshape(count, dim)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    header_bytes = struct.pack("<i", dim)

    flags = os.O_CREAT | os.O_WRONLY | os.O_TRUNC
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    fd = os.open(str(out_path), flags, 0o644)

    try:
        if output_format == "fvecs":
            arrf = arr.astype(np.float32, copy=False)
            bufs: List[bytes] = []
            for i in range(count):
                bufs.append(header_bytes)
                bufs.append(arrf[i].tobytes())
                if len(bufs) >= 16384:
                    _writev_chunked(fd, bufs); bufs.clear()
            if bufs:
                _writev_chunked(fd, bufs)
        else:
            arru = (arr.astype(np.int16) + 128).astype(np.uint8, copy=False)
            bufs: List[bytes] = []
            for i in range(count):
                bufs.append(header_bytes)
                bufs.append(arru[i].tobytes())
                if len(bufs) >= 16384:
                    _writev_chunked(fd, bufs); bufs.clear()
            if bufs:
                _writev_chunked(fd, bufs)
    finally:
        os.close(fd)

    print(f"[{bin_path.name}] -> {out_path.name}  ({count} x {dim}, format={output_format})")


def convert_truth(bin_path: Path, out_dir: Path) -> None:
    with open(bin_path, "rb") as f:
        hdr = f.read(8)
        if len(hdr) != 8:
            raise RuntimeError("Incomplete truth header.")
        q_count = struct.unpack("<i", hdr[:4])[0]
        topk = struct.unpack("<i", hdr[4:8])[0]
        vids_bytes = f.read(q_count * topk * 4)
        dists_bytes = f.read(q_count * topk * 4)
        if len(vids_bytes) != q_count * topk * 4 or len(dists_bytes) != q_count * topk * 4:
            raise RuntimeError("truth.bin truncated.")

    vids = np.frombuffer(vids_bytes, dtype=np.int32).reshape(q_count, topk)
    dists = np.frombuffer(dists_bytes, dtype=np.float32).reshape(q_count, topk)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Write vids as ivecs: per-vector [int32 dim=topk][topk * int32]
    ivecs_path = out_dir / "truth_vids.ivecs"
    flags = os.O_CREAT | os.O_WRONLY | os.O_TRUNC
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    fd = os.open(str(ivecs_path), flags, 0o644)
    try:
        h = struct.pack("<i", topk)
        bufs: List[bytes] = []
        for i in range(q_count):
            bufs.append(h)
            bufs.append(vids[i].tobytes())
            if len(bufs) >= 16384:
                _writev_chunked(fd, bufs); bufs.clear()
        if bufs:
            _writev_chunked(fd, bufs)
    finally:
        os.close(fd)

    # Write distances as fvecs: per-vector [int32 dim=topk][topk * float32]
    fvecs_path = out_dir / "truth_distances.fvecs"
    fd2 = os.open(str(fvecs_path), flags, 0o644)
    try:
        h = struct.pack("<i", topk)
        bufs: List[bytes] = []
        for i in range(q_count):
            bufs.append(h)
            bufs.append(dists[i].astype(np.float32, copy=False).tobytes())
            if len(bufs) >= 16384:
                _writev_chunked(fd2, bufs); bufs.clear()
        if bufs:
            _writev_chunked(fd2, bufs)
    finally:
        os.close(fd2)

    print(f"[truth] -> {ivecs_path.name} (vids), {fvecs_path.name} (distances) ; q={q_count} topk={topk}")


# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert SPACEV1B dataset to fvecs/bvecs/ivecs.")
    parser.add_argument("--src", type=Path, required=True, help="Path to SPACEV1B root (contains vectors.bin/, query.bin, truth.bin...)")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--what", type=str, default="all",
                        choices=["vectors", "queries", "query_log", "truth", "all"],
                        help="What to convert")
    parser.add_argument("--output-format", type=str, default="fvecs", choices=["fvecs", "bvecs"],
                        help="Output format for vectors/queries/query_log")
    parser.add_argument("--vectors-per-shard", type=int, default=2_000_000,
                        help="Vectors per output shard (for base vectors).")
    parser.add_argument("--chunk-bytes", type=int, default=64 * 1024 * 1024,
                        help="Read chunk size for streaming base vectors.")
    args = parser.parse_args()

    src = args.src
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    if args.what in ("vectors", "all"):
        convert_vectors(
            src_root=src,
            out_root=out,
            vectors_per_shard=args.vectors_per_shard,
            output_format=args.output_format,
            chunk_bytes=args.chunk_bytes,
        )

    if args.what in ("queries", "all"):
        convert_query_like_file(
            src / "query.bin",
            out / f"query.{args.output_format}",
            output_format=args.output_format
        )

    if args.what in ("query_log", "all"):
        qlog = src / "query_log.bin"
        if qlog.exists():
            convert_query_like_file(
                qlog,
                out / f"query_log.{args.output_format}",
                output_format=args.output_format
            )
        else:
            print("[query_log] query_log.bin not found; skipping.")

    if args.what in ("truth", "all"):
        convert_truth(src / "truth.bin", out)

if __name__ == "__main__":
    main()
