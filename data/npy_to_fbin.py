import numpy as np
import os
import glob
import argparse
from typing import Dict, List, Tuple


def verify_npy_files(emb_folder: str, expected_dim: int = None) -> Dict:
    """
    校验目录下所有 img_emb_*.npy 文件的完整性。
    
    Args:
        emb_folder (str): 存放 .npy 向量文件的目录
        expected_dim (int, optional): 期望的向量维度，None表示从第一个有效文件中获取
    
    Returns:
        dict: 包含校验结果的字典，包括：
            - total_files: 总文件数
            - valid_files: 有效文件数
            - broken_files: 损坏的文件列表
            - dimension_mismatch: 维度不匹配的文件列表
            - has_nan: 包含NaN的文件列表
            - has_inf: 包含Inf的文件列表
            - file_details: 每个文件的详细信息
    """
    file_pattern = os.path.join(emb_folder, "img_emb_*.npy")
    npy_files = sorted(glob.glob(file_pattern))
    
    if not npy_files:
        print(f"⚠️ 错误：在目录 '{emb_folder}' 下没有找到匹配 '{file_pattern}' 的文件。")
        return {}
    
    print(f"\n{'='*60}")
    print(f"🔍 开始校验 {len(npy_files)} 个 .npy 文件...")
    print(f"{'='*60}\n")
    
    result = {
        'total_files': len(npy_files),
        'valid_files': 0,
        'broken_files': [],
        'dimension_mismatch': [],
        'has_nan': [],
        'has_inf': [],
        'file_details': []
    }
    
    detected_dim = expected_dim
    
    for idx, filepath in enumerate(npy_files, 1):
        filename = os.path.basename(filepath)
        file_info = {
            'filename': filename,
            'filepath': filepath,
            'status': 'unknown',
            'shape': None,
            'has_nan': False,
            'has_inf': False,
            'error': None
        }
        
        print(f"[{idx}/{len(npy_files)}] 检查: {filename} ... ", end="", flush=True)
        
        try:
            # 尝试加载文件
            data = np.load(filepath, mmap_mode='r')
            
            # 检查维度
            if data.ndim != 2:
                file_info['status'] = 'invalid_shape'
                file_info['shape'] = data.shape
                file_info['error'] = f"维度不是2D，实际为 {data.shape}"
                result['dimension_mismatch'].append(file_info)
                print(f"⚠️  维度错误: {data.shape}")
                continue
            
            num_vecs, cur_dim = data.shape
            file_info['shape'] = (num_vecs, cur_dim)
            
            # 确定或检查维度
            if detected_dim is None:
                detected_dim = cur_dim
                print(f"✓ (首个有效文件，维度={cur_dim}, 向量数={num_vecs})")
            else:
                if cur_dim != detected_dim:
                    file_info['status'] = 'dimension_mismatch'
                    file_info['error'] = f"维度 {cur_dim} 与期望的 {detected_dim} 不一致"
                    result['dimension_mismatch'].append(file_info)
                    print(f"⚠️  维度不匹配: {cur_dim} != {detected_dim}")
                    continue
                else:
                    print(f"✓ (维度={cur_dim}, 向量数={num_vecs})", end="")
            
            # 检查NaN和Inf（采样检查以提高速度）
            # 对于大文件，只检查前1000个向量
            sample_size = min(1000, num_vecs)
            sample_data = data[:sample_size, :]
            
            has_nan = np.isnan(sample_data).any()
            has_inf = np.isinf(sample_data).any()
            
            file_info['has_nan'] = bool(has_nan)
            file_info['has_inf'] = bool(has_inf)
            
            warnings = []
            if has_nan:
                result['has_nan'].append(file_info)
                warnings.append("含NaN")
            if has_inf:
                result['has_inf'].append(file_info)
                warnings.append("含Inf")
            
            if warnings:
                print(f" ⚠️  {', '.join(warnings)}")
            else:
                if detected_dim is not None and idx > 1:  # 不是第一个文件时才打印
                    print()
            
            file_info['status'] = 'valid'
            result['valid_files'] += 1
            
        except Exception as e:
            file_info['status'] = 'broken'
            file_info['error'] = str(e)
            result['broken_files'].append(file_info)
            print(f"❌ 损坏: {e}")
        
        result['file_details'].append(file_info)
    
    # 打印摘要
    print(f"\n{'='*60}")
    print("📊 校验结果摘要:")
    print(f"{'='*60}")
    print(f"总文件数:       {result['total_files']}")
    print(f"✅ 有效文件:    {result['valid_files']}")
    print(f"❌ 损坏文件:    {len(result['broken_files'])}")
    print(f"⚠️  维度不匹配:  {len(result['dimension_mismatch'])}")
    print(f"⚠️  包含NaN:     {len(result['has_nan'])}")
    print(f"⚠️  包含Inf:     {len(result['has_inf'])}")
    
    if detected_dim:
        print(f"\n检测到的向量维度: {detected_dim}")
    
    # 详细列出问题文件
    if result['broken_files']:
        print(f"\n❌ 损坏的文件列表:")
        for f in result['broken_files']:
            print(f"   - {f['filename']}: {f['error']}")
    
    if result['dimension_mismatch']:
        print(f"\n⚠️  维度不匹配的文件:")
        for f in result['dimension_mismatch']:
            print(f"   - {f['filename']}: {f['error']}")
    
    if result['has_nan']:
        print(f"\n⚠️  包含NaN的文件:")
        for f in result['has_nan']:
            print(f"   - {f['filename']}")
    
    if result['has_inf']:
        print(f"\n⚠️  包含Inf的文件:")
        for f in result['has_inf']:
            print(f"   - {f['filename']}")
    
    print(f"\n{'='*60}\n")
    
    return result


def export_first_n_vectors_skip_broken_fbin(
    target_vector_count: int,
    emb_folder: str,
    output_filename: str = "laion100m.fbin",
):
    """
    从 emb_folder 下按文件名顺序读取 img_emb_*.npy，
    跳过损坏/无法读取的文件，直到凑够 target_vector_count 个向量，
    并以 fbin 格式写出。
    
    fbin格式：
    - 前4字节：向量数量 (uint32)
    - 接下来4字节：向量维度 (uint32)
    - 然后是所有向量数据 (float32)

    Args:
        target_vector_count (int): 目标导出的向量总数（例如 100_000_000）。
        emb_folder (str): 存放 .npy 向量文件的目录。
        output_filename (str): 输出的 fbin 文件路径。
    """

    file_pattern = os.path.join(emb_folder, "*_emb_*.npy")
    npy_files = sorted(glob.glob(file_pattern))

    if not npy_files:
        print(f"⚠️ 错误：在目录 '{emb_folder}' 下没有找到匹配 '{file_pattern}' 的文件。")
        return

    print(f"✅ 找到 {len(npy_files)} 个 .npy 文件。")
    print(f"🔄 目标导出向量总数：{target_vector_count}")
    print(f"📝 输出格式：fbin")

    total_written = 0      # 已成功写出的向量数量
    dim = None             # 向量维度（第一次读取后锁定）
    file_idx = 0

    # 先写入文件头（占位），稍后回填实际数量
    with open(output_filename, "wb") as fout:
        # 占位：写入数量和维度（稍后更新）
        np.array([0], dtype=np.uint32).tofile(fout)  # 数量占位
        np.array([0], dtype=np.uint32).tofile(fout)  # 维度占位
        
        for filepath in npy_files:
            if total_written >= target_vector_count:
                print("🛑 已达到目标数量，停止读取后续文件。")
                break

            file_idx += 1
            print(f"\n📂 处理第 {file_idx}/{len(npy_files)} 个文件: {filepath}")

            # 读取 .npy，损坏文件会在这里抛异常
            try:
                data = np.load(filepath, mmap_mode="r")
            except Exception as e:
                print(f"❌ 无法加载文件 {filepath}，跳过。错误：{e}")
                continue

            if data.ndim != 2:
                print(f"⚠️ 警告：文件 {filepath} 的维度不是 (N, D)，实际 {data.shape}，跳过。")
                continue

            num_vecs, cur_dim = data.shape

            if dim is None:
                dim = cur_dim
                print(f"ℹ️ 设定向量维度 dim = {dim}")
            else:
                if cur_dim != dim:
                    print(
                        f"⚠️ 警告：文件 {filepath} 的维度 {cur_dim} 与之前的 {dim} 不一致，跳过。"
                    )
                    continue

            remaining = target_vector_count - total_written
            take = min(num_vecs, remaining)

            if take <= 0:
                print("ℹ️ 不再需要更多向量。")
                break

            # 只取前 take 个向量
            batch = data[:take, :]

            # 写 fbin: 直接写入向量数据（不需要每个向量前写维度）
            try:
                # 向量数据转 float32 后写出
                batch = batch.astype("float32", copy=False)
                batch.tofile(fout)

                total_written += take

                print(
                    f"   -> 从该文件写出 {take} 个向量，"
                    f"当前总计：{total_written}/{target_vector_count}"
                )

            except Exception as e:
                print(f"❌ 写入文件 {output_filename} 时发生错误，停止：{e}")
                break

    # 回填正确的数量和维度到文件头
    if dim is not None and total_written > 0:
        with open(output_filename, "r+b") as fout:
            fout.seek(0)
            np.array([total_written], dtype=np.uint32).tofile(fout)
            np.array([dim], dtype=np.uint32).tofile(fout)

    print("\n✅ 处理结束。")
    if total_written < target_vector_count:
        print(
            f"⚠️ 仅写出 {total_written} 个向量，"
            f"未能达到目标 {target_vector_count}（可能是文件数量不足或很多文件损坏）。"
        )
    else:
        print(f"🎉 已成功写出 {total_written} 个向量到 {output_filename}。")
    
    print(f"📊 文件信息: 向量数={total_written}, 维度={dim}")


def export_first_n_vectors_skip_broken(
    target_vector_count: int,
    emb_folder: str,
    output_filename: str = "laion100m.fvecs",
):
    """
    从 emb_folder 下按文件名顺序读取 img_emb_*.npy，
    跳过损坏/无法读取的文件，直到凑够 target_vector_count 个向量，
    并以 fvecs 格式写出。

    Args:
        target_vector_count (int): 目标导出的向量总数（例如 100_000_000）。
        emb_folder (str): 存放 .npy 向量文件的目录。
        output_filename (str): 输出的 fvecs 文件路径。
    """

    file_pattern = os.path.join(emb_folder, "img_emb_*.npy")
    npy_files = sorted(glob.glob(file_pattern))

    if not npy_files:
        print(f"⚠️ 错误：在目录 '{emb_folder}' 下没有找到匹配 '{file_pattern}' 的文件。")
        return

    print(f"✅ 找到 {len(npy_files)} 个 .npy 文件。")
    print(f"🔄 目标导出向量总数：{target_vector_count}")

    total_written = 0      # 已成功写出的向量数量
    dim = None             # 向量维度（第一次读取后锁定）
    file_idx = 0

    # 直接边读边写，避免把 100M 全放到内存里
    with open(output_filename, "wb") as fout:
        for filepath in npy_files:
            if total_written >= target_vector_count:
                print("🛑 已达到目标数量，停止读取后续文件。")
                break

            file_idx += 1
            print(f"\n📂 处理第 {file_idx}/{len(npy_files)} 个文件: {filepath}")

            # 读取 .npy，损坏文件会在这里抛异常
            try:
                # mmap_mode='r' 可以减少内存峰值（视情况使用）
                data = np.load(filepath, mmap_mode="r")
            except Exception as e:
                print(f"❌ 无法加载文件 {filepath}，跳过。错误：{e}")
                continue

            if data.ndim != 2:
                print(f"⚠️ 警告：文件 {filepath} 的维度不是 (N, D)，实际 {data.shape}，跳过。")
                continue

            num_vecs, cur_dim = data.shape

            if dim is None:
                dim = cur_dim
                print(f"ℹ️ 设定向量维度 dim = {dim}")
            else:
                if cur_dim != dim:
                    print(
                        f"⚠️ 警告：文件 {filepath} 的维度 {cur_dim} 与之前的 {dim} 不一致，跳过。"
                    )
                    continue

            remaining = target_vector_count - total_written
            take = min(num_vecs, remaining)

            if take <= 0:
                print("ℹ️ 不再需要更多向量。")
                break

            # 只取前 take 个向量
            batch = data[:take, :]

            # 写 fvecs: 每个向量前写一个 int32 的维度，然后写 float32 向量
            # 为了效率，这里做成批写：dims_block + batch_block
            try:
                # dims_block: [take, 1] 全是 dim
                dims_block = np.full((take, 1), dim, dtype="int32")
                dims_block.tofile(fout)

                # 向量数据转 float32 后写出
                batch = batch.astype("float32", copy=False)
                batch.tofile(fout)

                total_written += take

                print(
                    f"   -> 从该文件写出 {take} 个向量，"
                    f"当前总计：{total_written}/{target_vector_count}"
                )

            except Exception as e:
                print(f"❌ 写入文件 {output_filename} 时发生错误，停止：{e}")
                break

    print("\n✅ 处理结束。")
    if total_written < target_vector_count:
        print(
            f"⚠️ 仅写出 {total_written} 个向量，"
            f"未能达到目标 {target_vector_count}（可能是文件数量不足或很多文件损坏）。"
        )
    else:
        print(f"🎉 已成功写出 {total_written} 个向量到 {output_filename}。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="校验或提取 LAION npy 向量文件"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["verify", "extract"],
        default="extract",
        help="操作模式：verify=校验文件完整性，extract=提取向量"
    )
    parser.add_argument(
        "--emb_folder",
        type=str,
        default="/root/paper/laion500m",
        help="存放 img_emb_*.npy 文件的目录"
    )
    parser.add_argument(
        "--target",
        type=int,
        default=100_020_000,
        help="提取模式：目标向量数量"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/root/paper/laion500m/laion101m.fvecs",
        help="提取模式：输出文件路径"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["fvecs", "fbin"],
        default=None,
        help="提取模式：输出格式 (fvecs或fbin)，默认根据输出文件扩展名自动判断"
    )
    parser.add_argument(
        "--expected_dim",
        type=int,
        default=None,
        help="校验模式：期望的向量维度（可选）"
    )
    
    args = parser.parse_args()
    
    if args.mode == "verify":
        # 校验模式
        print(f"🔍 校验模式")
        print(f"目录: {args.emb_folder}")
        if args.expected_dim:
            print(f"期望维度: {args.expected_dim}")
        
        result = verify_npy_files(args.emb_folder, args.expected_dim)
        
        # 可选：将结果保存到JSON文件
        # import json
        # with open('verification_result.json', 'w') as f:
        #     json.dump(result, f, indent=2)
        
    elif args.mode == "extract":
        # 提取模式
        print(f"📤 提取模式")
        print(f"目录: {args.emb_folder}")
        print(f"目标向量数: {args.target:,}")
        print(f"输出文件: {args.output}")
        
        if args.target <= 0:
            print("❌ 错误：目标向量数量必须 > 0。")
        else:
            # 自动判断输出格式
            output_format = args.format
            if output_format is None:
                # 根据文件扩展名自动判断
                if args.output.endswith('.fbin'):
                    output_format = 'fbin'
                elif args.output.endswith('.fvecs'):
                    output_format = 'fvecs'
                else:
                    # 默认使用fvecs
                    output_format = 'fvecs'
                    print(f"⚠️  未识别的文件扩展名，默认使用 {output_format} 格式")
            
            print(f"📝 输出格式: {output_format}")
            
            # 根据格式调用相应的函数
            if output_format == 'fbin':
                export_first_n_vectors_skip_broken_fbin(
                    target_vector_count=args.target,
                    emb_folder=args.emb_folder,
                    output_filename=args.output,
                )
            else:  # fvecs
                export_first_n_vectors_skip_broken(
                    target_vector_count=args.target,
                    emb_folder=args.emb_folder,
                    output_filename=args.output,
                )


