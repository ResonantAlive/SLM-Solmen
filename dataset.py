# dataset.py
# 用途：预训练数据集加载
# 支持多个 .jsonl 文件顺序读取，记录训练进度（当前文件索引 + 行号），支持断点续读
# .jsonl 格式要求：每行是一个 JSON 对象，包含 "text" 字段

import json                          # 解析每行的 JSON 字符串
import torch                         # 用于构造 tensor 返回给 DataLoader
from torch.utils.data import IterableDataset  # 流式数据集基类，适合超大数据，不需要提前全部加载进内存
from pathlib import Path             # 面向对象的路径操作，比字符串拼接更安全
from typing import Iterator          # 类型标注：Iterator[X] 表示"可以 yield X 的迭代器"


class PretrainDataset(IterableDataset):
    """
    流式预训练数据集
    - 顺序遍历 file_list 中的所有 .jsonl 文件
    - 将文本 token 化后切分为固定长度的训练样本
    - 记录当前训练到的 (文件索引, 行号)，支持断点续读
    """

    def __init__(
        self,
        file_list: list[str],        # .jsonl 文件路径列表，按顺序读取
        tokenizer,                   # HuggingFace tokenizer 实例，负责文本→token id 转换
        seq_len: int,                # 每个训练样本的 token 长度（与 max_seq_len 一致）
        resume_file_idx: int = 0,    # 断点续读：从第几个文件开始（0-indexed）
        resume_line: int = 0,        # 断点续读：在 resume_file_idx 文件中跳过多少行
    ):
        """
        file_list       : .jsonl 文件路径列表，按顺序训练
        tokenizer       : HuggingFace tokenizer 实例
        seq_len         : 每个训练样本的 token 长度（与 ModelConfig.max_seq_len 保持一致）
        resume_file_idx : 断点续读：从第几个文件开始（0-indexed）
        resume_line     : 断点续读：跳过该文件前多少行
        """
        # 将字符串路径全部转为 Path 对象，方便后续做 .exists()、.name 等操作
        self.file_list = [Path(p) for p in file_list]
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.resume_file_idx = resume_file_idx  # 用于 _iter_tokens 中决定从哪里开始
        self.resume_line = resume_line

        # current_file_idx / current_line 在迭代过程中实时更新，供外部查询训练进度
        self.current_file_idx = resume_file_idx
        self.current_line = resume_line

    def _iter_tokens(self) -> Iterator[int]:
        """逐文件、逐行读取并 yield token id"""
        for file_idx, path in enumerate(self.file_list):  # 遍历数据文件列表，file_idx 从 0 开始
            # 跳过已经训练完的文件（恢复断点时用）
            if file_idx < self.resume_file_idx:  # 跳过断点之前的文件
                continue                    # 不处理已训练完的文件

            self.current_file_idx = file_idx  # 实时更新当前文件进度
            # 只有起始文件才需要跳过行，后续文件从第 0 行开始
            skip_lines = self.resume_line if file_idx == self.resume_file_idx else 0

            if not path.exists():              # 检查文件是否真的存在于磁盘上
                # 文件不存在时给出警告并跳过，不中断训练
                print(f"[警告] 文件不存在，已跳过：{path}")
                continue

            print(f"[数据] 读取文件 [{file_idx + 1}/{len(self.file_list)}]：{path.name}"
                  + (f"（跳过前 {skip_lines} 行）" if skip_lines > 0 else ""))

            with open(path, "r", encoding="utf-8") as f:  # 以 UTF-8 编码打开数据文件
                for line_no, line in enumerate(f):       # 逐行枚举，line_no 从 0 开始
                    if line_no < skip_lines:
                        continue     # 跳过已训练的行（断点续读）

                    self.current_line = line_no  # 更新当前行号进度
                    line = line.strip()          # 去掉行首行尾空白字符（包括换行符）
                    if not line:
                        continue     # 跳过空行

                    try:
                        obj = json.loads(line)   # 将 JSON 字符串解析为字典
                        text = obj.get("text", "")  # 取 "text" 字段，不存在则返回空串
                    except (json.JSONDecodeError, AttributeError):
                        continue     # JSON 格式损坏时跳过该行，不崩溃

                    if not text:
                        continue     # text 为空则跳过

                    # Bug fix: 部分 tokenizer 配置的 encode() 默认 add_special_tokens=True，
                    # 会在序列末尾自动追加 EOS，而下方代码又手动 append(eos_id)，
                    # 导致每篇文档以双 EOS 结尾，污染训练序列边界信号。
                    # 修复：显式传入 add_special_tokens=False，由代码统一控制 EOS 追加。
                    ids = self.tokenizer.encode(text, add_special_tokens=False)
                    # 将文本转为 token id 列表，禁止 tokenizer 自动加特殊 token

                    eos_id = self.tokenizer.eos_token_id  # 获取结束符 id
                    if eos_id is not None:
                        ids.append(eos_id)  # 每篇文档末尾手动加一个 EOS，标记文档边界

                    yield from ids  # 逐个 yield token id（生成器语法，节省内存）

            self.resume_line = 0    # 当前文件处理完，下一个文件从第 0 行开始

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """
        将 token 流切分为 (input_ids, labels) 对
        labels 是 input_ids 错位一位：labels[i] = input_ids[i+1]（标准 next-token prediction）
        """
        buf: list[int] = []          # 临时缓冲区，积累 token 直到够一个完整样本
        chunk = self.seq_len + 1     # 多取 1 个 token 用于构造 labels（错位需要多一个）

        for token_id in self._iter_tokens():  # 从生成器逐个取出 token id
            buf.append(token_id)     # 不断往缓冲区里加 token
            if len(buf) >= chunk:              # 缓冲区的 token 够了，可以切一个样本
                sample = buf[:chunk]         # 取出 seq_len+1 个 token
                buf = buf[chunk - 1:]        # 滑窗重叠一个 token，保证上下文连续（避免硬切割丢信息）
                # 例如 seq_len=4：第一个样本用 token [0,1,2,3,4]，第二个从 [3,4,5,6,7] 开始
                # 这样每个样本的开头都接上了上一个样本的尾巴，不会丢失边界处的信息
                input_ids = torch.tensor(sample[:-1], dtype=torch.long)  # 前 seq_len 个作为输入
                labels    = torch.tensor(sample[1:],  dtype=torch.long)  # 后 seq_len 个作为标签（错位一位）
                # yield 返回一个字典，DataLoader 会自动将多个样本拼成 batch
                yield {"input_ids": input_ids, "labels": labels}


def get_progress(dataset: PretrainDataset) -> dict:
    """获取当前数据集训练进度，用于写入 checkpoint"""
    return {
        "file_idx": dataset.current_file_idx,    # 当前正在读取的文件编号
        "line":     dataset.current_line,         # 当前文件读到第几行
        "file_name": str(dataset.file_list[dataset.current_file_idx])
                     if dataset.file_list else "", # 当前文件名（方便人工核对）
    }