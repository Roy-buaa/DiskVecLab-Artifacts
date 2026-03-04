#!/usr/bin/env python3
"""
日志文件压缩工具 - 去除重复和冗余内容（改进版）
"""
import glob
import re
from pathlib import Path

def compress_log(input_file, output_file):
    """压缩日志文件"""
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    compressed_lines = []
    last_pattern = None
    pattern_count = 0
    posting_list_count = 0
    delete_workspace_count = 0
    tptree_split_count = 0
    
    for line in lines:
        # 检测 TpTree split node 模式
        if 'TpTree split node' in line:
            tptree_split_count += 1
            continue
        
        # 检测 PostingList 模式
        if re.match(r'\[\d+\] PostingList\[\d+\]:\s+\d+ vectors', line):
            posting_list_count += 1
            # if posting_list_count % 1000000 == 0:
            #     compressed_lines.append(f"[1] ... (已跳过 {posting_list_count} 条 PostingList 日志)\n")
            continue
        
        # 检测 Delete workspace 模式
        if 'Delete workspace happens!' in line:
            delete_workspace_count += 1
            if delete_workspace_count == 1:
                compressed_lines.append(line)
            continue
        else:
            if delete_workspace_count > 1:
                compressed_lines.append(f"[1] ... (前面有 {delete_workspace_count} 条 'Delete workspace happens!' 消息)\n")
            delete_workspace_count = 0
        
        # 检测 Lambda 模式（重复的统计数据）
        if re.match(r'\[\d+\] Lambda:min\(', line):
            if last_pattern == 'lambda':
                pattern_count += 1
                continue
            else:
                if pattern_count > 1:
                    compressed_lines.append(f"[1] ... (前面有 {pattern_count} 条相似的 Lambda 统计行)\n")
                pattern_count = 0
                last_pattern = 'lambda'
                compressed_lines.append(line)
                continue
        else:
            if pattern_count > 1:
                compressed_lines.append(f"[1] ... (前面有 {pattern_count} 条相似的 Lambda 统计行)\n")
            pattern_count = 0
            last_pattern = None
        
        # 其他行保留
        compressed_lines.append(line)
    
    # 处理最后的统计
    if posting_list_count > 0:
        compressed_lines.append(f"[1] === 共处理 {posting_list_count} 条 PostingList 日志，已去重 ===\n")
    if delete_workspace_count > 1:
        compressed_lines.append(f"[1] ... (最后有 {delete_workspace_count} 条 'Delete workspace happens!' 消息)\n")
    if tptree_split_count > 0:
        compressed_lines.append(f"[1] === 已移除 {tptree_split_count} 条 TpTree split node 日志 ===\n")
    
    # 写入输出文件
    with open(output_file, 'w') as f:
        f.writelines(compressed_lines)
    
    original_lines = len(lines)
    compressed_size = len(compressed_lines)
    reduction_rate = (1 - compressed_size / original_lines) * 100
    
    print(f"原始行数: {original_lines}")
    print(f"压缩后行数: {compressed_size}")
    print(f"压缩率: {reduction_rate:.1f}%")
    print(f"\n移除详情:")
    print(f"  - TpTree split node: {tptree_split_count} 行")
    print(f"  - PostingList: {posting_list_count} 行")
    print(f"  - Delete workspace: {delete_workspace_count} 行")
    print(f"\n已保存到: {output_file}")

def _build_output_path(path: Path) -> Path:
    """生成输出路径: name.compressed.log 或 name.compressed.<suffix>"""
    if path.suffix:
        return path.with_suffix(f'.compressed{path.suffix}')
    return path.with_name(f'{path.name}.compressed.log')


if __name__ == '__main__':
    pattern = input('请输入要压缩的日志文件通配符（如 /path/to/*.log）: ').strip()
    matches = sorted(glob.glob(pattern))

    if not matches:
        print('未找到匹配的日志文件')
    else:
        for file_path in matches:
            log_path = Path(file_path)
            if not log_path.is_file():
                print(f'跳过非文件: {log_path}')
                continue

            output_path = _build_output_path(log_path)
            print(f'处理: {log_path} -> {output_path}')
            try:
                compress_log(str(log_path), str(output_path))
            except Exception as exc:
                print(f'处理 {log_path} 时出错: {exc}')
