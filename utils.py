# utils.py
# 用途：公共工具函数
# 包含：日志打印、余弦学习率调度、checkpoint 保存/加载（HF 格式 + 优化器状态）

import json                          # 保存 config.json 用
import math                          # 余弦退火需要 math.cos 和 math.pi
import os                            # 文件系统操作（当前未直接用，被 pathlib 替代，保留备用）
import shutil                        # shutil.rmtree：递归删除目录（用于清理旧 checkpoint）
import time                          # time.strftime：格式化当前时间作为日志前缀
from pathlib import Path             # 面向对象的路径类，比字符串拼接更安全
from typing import Optional          # 类型标注：Optional[X] 表示"X 或 None"

import torch                         # 张量和模型操作
import torch.nn as nn                # nn.Module 类型标注用


# ─── 日志 ─────────────────────────────────────────────────────────────
class Logger:
    """简单的控制台日志，带时间戳，同时写入日志文件"""

    def __init__(self, log_file: Optional[str] = None):  # log_file 为 None 时只输出到控制台
        # 如果指定了日志文件路径，以追加模式打开；否则 log_file 为 None（只打印到控制台）
        # "a" 模式：追加写入，不会覆盖已有内容，续训时日志可以接着写
        self.log_file = open(log_file, "a", encoding="utf-8") if log_file else None

    def info(self, msg: str):              # 打印并写入一条带时间戳的日志消息
        # 给消息加上时间戳，格式如：[2024-01-15 08:30:00] 消息内容
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line)                  # 打印到终端（实时可见）
        if self.log_file:                  # 配置了日志文件才写入，否则只打印到控制台
            self.log_file.write(line + "\n")  # 写入日志文件
            self.log_file.flush()    # 立即刷新缓冲区，防止程序崩溃时丢失日志

    def close(self):                       # 训练结束时关闭日志文件
        # 训练结束时手动关闭文件，确保所有内容都写入磁盘
        if self.log_file:                  # 有日志文件才执行关闭操作
            self.log_file.close()          # 关闭文件句柄，将缓冲区内容写入磁盘


# ─── 学习率调度 ───────────────────────────────────────────────────────
def get_lr(
    step: int,           # 当前全局训练步数（从 1 开始，见 pretrain.py 的 Bug fix 说明）
    warmup_steps: int,   # 线性 Warmup 的步数
    max_steps: int,      # 总训练步数
    lr: float,           # 峰值学习率
    min_lr: float,       # 最小学习率（余弦退火的终点）
) -> float:
    """
    余弦退火学习率调度（含线性 Warmup）
    - step < warmup_steps：从 0 线性升温到 lr
    - warmup_steps <= step <= max_steps：余弦从 lr 降至 min_lr
    - step > max_steps：保持 min_lr

    为什么要 Warmup：训练刚开始时模型参数是随机的，如果学习率一步到最大值，
    可能会让参数"跑飞"。Warmup 让学习率从 0 慢慢爬到峰值，相当于"热身运动"。

    为什么要余弦退火：训练后期模型已经比较好了，需要小幅微调。
    余弦曲线（像钟摆一样从 1 摆到 0）比线性下降更平滑，避免训练末期震荡。

    注意：pretrain.py 已修复为先 global_step += 1 再调用本函数，
    因此 step 从 1 开始，Warmup 第一步即有非零 lr。
    """
    if step < warmup_steps:                # Warmup 阶段：学习率从 0 线性增长
        # 线性 Warmup：lr * (step / warmup_steps)
        # max(warmup_steps, 1) 防止 warmup_steps=0 时除以零
        return lr * step / max(warmup_steps, 1)
    if step > max_steps:                   # 超过最大步数：保持最小学习率不变
        # 超过总步数后保持最小学习率（训练已结束，这行通常不会触发）
        return min_lr
    # 余弦退火阶段：计算当前在 [warmup_steps, max_steps] 区间中的进度（0→1）
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    # 余弦系数：progress=0 时 coeff=1（峰值），progress=1 时 coeff=0（最小值）
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    # 线性插值：在 [min_lr, lr] 之间按 coeff 插值
    return min_lr + coeff * (lr - min_lr)


# ─── Checkpoint 管理 ──────────────────────────────────────────────────
class CheckpointManager:
    """
    HuggingFace 格式 checkpoint 管理器
    - 保存：模型权重（config.json + pytorch_model.bin）+ 分词器 + 优化器状态 + 训练进度
    - 自动删除旧 checkpoint，只保留最近 keep 个
    """

    def __init__(self, output_dir: str, keep: int = 3):  # keep 控制最多保留几个 checkpoint
        self.output_dir = Path(output_dir)   # 转为 Path 对象
        self.keep = keep                     # 最多保留多少个 checkpoint
        self.output_dir.mkdir(parents=True, exist_ok=True)  # 创建目录，已存在时不报错
        self._history: list[Path] = self._scan_existing()   # 扫描已有 checkpoint 并初始化历史列表

    def _scan_existing(self) -> list[Path]:  # 扫描目录下已有的 checkpoint 子目录
        # 扫描目录下所有名为 "checkpoint-{数字}" 的子目录，按步数排序
        dirs = sorted(
            [d for d in self.output_dir.iterdir()
             if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda d: int(d.name.split("-")[-1])  # 按步数（名称末尾的数字）升序排列
        )
        return dirs

    def save(
        self,
        step: int,                   # 当前全局步数，用于命名目录（checkpoint-{step}）
        model: nn.Module,            # SolmenModel 实例
        optimizer: torch.optim.Optimizer,  # AdamW 优化器，需要保存动量状态以便续训
        tokenizer,                   # HuggingFace tokenizer 实例
        data_progress: dict,         # 数据进度字典，由 dataset.get_progress() 获得
        extra: Optional[dict] = None,  # 其他需要保存的标量（如 loss 等）
        scaler: Optional[torch.cuda.amp.GradScaler] = None,  # fp16 训练时传入，保存 scale 状态
    ) -> Path:
        """
        保存当前 checkpoint
        step          : 当前全局步数
        model         : SolmenModel 实例
        optimizer     : AdamW 优化器
        tokenizer     : HuggingFace tokenizer 实例
        data_progress : 数据进度字典，由 dataset.get_progress() 获得
        extra         : 其他需要保存的标量（如 loss 等）
        scaler        : GradScaler 实例（fp16 训练时传入，bf16/fp32 可不传）
        """
        ckpt_dir = self.output_dir / f"checkpoint-{step}"  # 目录名格式：checkpoint-1000
        ckpt_dir.mkdir(parents=True, exist_ok=True)         # 创建 checkpoint 目录

        # 1. 保存模型权重（HuggingFace 格式）
        # state_dict()：返回模型所有参数的字典（key=参数名，value=tensor）
        torch.save(model.state_dict(), ckpt_dir / "pytorch_model.bin")

        # 2. 保存模型配置（HF 格式的 config.json）
        # 将 ModelConfig 的字段转换为 HuggingFace 标准字段名，方便后续用 HF 工具加载
        cfg = model.cfg
        hf_config = {
            "model_type": "solmen",
            "vocab_size": cfg.vocab_size,
            "hidden_size": cfg.hidden_size,
            "num_hidden_layers": cfg.num_layers,         # HF 用 num_hidden_layers，我们内部叫 num_layers
            "num_attention_heads": cfg.num_heads,
            "num_key_value_heads": cfg.num_kv_heads,
            "intermediate_size": cfg.intermediate_size,
            "max_position_embeddings": cfg.max_seq_len,  # HF 叫 max_position_embeddings
            "rms_norm_eps": cfg.rms_norm_eps,
            "rope_theta": cfg.rope_theta,
            "hidden_act": "silu",                        # 激活函数名称
            "tie_word_embeddings": True,                 # 标记权重共享
            "torch_dtype": "bfloat16",
            "architectures": ["SolmenModel"],            # 模型类名，用于 HF AutoModel 加载
        }
        with open(ckpt_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(hf_config, f, indent=2, ensure_ascii=False)  # indent=2 格式化缩进，便于阅读

        # 3. 保存分词器
        # Bug fix: 原代码忽略传入的 tokenizer 参数，硬编码从 tokenizer/ 目录复制文件。
        # 若该目录不存在（迁移环境 / 容器），checkpoint 里完全没有 tokenizer，
        # 续训时 PreTrainedTokenizerFast.from_pretrained(ckpt_dir) 直接报错。
        # 修复：使用 HuggingFace 标准接口，直接从 tokenizer 对象序列化保存。
        tokenizer.save_pretrained(str(ckpt_dir))  # 自动保存 tokenizer.json 和 tokenizer_config.json

        # 4. 保存训练状态（优化器动量、步数、数据进度）
        training_state = {
            "step": step,
            "optimizer_state_dict": optimizer.state_dict(),  # AdamW 的 m、v 动量，续训必须恢复
            "data_progress": data_progress,                  # 记录读到哪个文件哪一行
        }
        if extra:
            training_state.update(extra)  # 合并额外信息（如 loss）

        # Bug fix: 保存 GradScaler 状态，防止 fp16 续训时 scale 值从默认值重置，
        # 导致前几步出现 NaN → skip 的震荡现象。bf16 传入的 scaler 是 disabled
        # 状态，state_dict() 仍可正常序列化（enabled=False 时 state 为空字典）。
        if scaler is not None:
            training_state["scaler_state_dict"] = scaler.state_dict()

        # weights_only 不适用于 optimizer state（含 Python 对象），故不加此参数
        torch.save(training_state, ckpt_dir / "training_state.pt")  # 保存训练状态到 .pt 文件

        # 5. 更新历史记录，删除超出 keep 数量的旧 checkpoint
        self._history.append(ckpt_dir)
        while len(self._history) > self.keep:  # 超过保留数量上限，删除最旧的
            old = self._history.pop(0)    # 从历史列表头部取出最旧的
            if old.exists():               # 目录确实存在才删除（防御性检查）
                shutil.rmtree(old)        # 递归删除整个 checkpoint 目录

        return ckpt_dir                    # 返回保存的目录路径供日志使用

    def latest(self) -> Optional[Path]:
        """返回最新 checkpoint 目录，不存在则返回 None"""
        return self._history[-1] if self._history else None  # 列表为空时返回 None


def load_checkpoint(
    ckpt_dir: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,  # 不传则只加载模型权重（用于推理）
    device: Optional[torch.device] = None,              # 目标设备，None 默认 CPU
) -> dict:
    """
    从 checkpoint 目录加载模型权重和训练状态
    返回 training_state 字典（含 step、data_progress、scaler_state_dict 等）
    调用方根据需要自行恢复 scaler：
        if use_scaler and "scaler_state_dict" in state:
            scaler.load_state_dict(state["scaler_state_dict"])
    """
    weight_path = ckpt_dir / "pytorch_model.bin"
    # weights_only=True：只允许加载张量，禁止反序列化任意 Python 对象（防止恶意 checkpoint）
    state_dict = torch.load(weight_path, map_location=device or "cpu", weights_only=True)
    model.load_state_dict(state_dict)  # 将权重加载到模型

    state_path = ckpt_dir / "training_state.pt"
    # weights_only=False：训练状态包含非张量对象（如 step 整数、字典等），需关闭限制
    training_state = torch.load(state_path, map_location=device or "cpu", weights_only=False)

    if optimizer is not None:
        optimizer.load_state_dict(training_state["optimizer_state_dict"])
        # 加载完优化器状态后，需要手动将动量张量移到正确的设备
        # （optimizer.load_state_dict 默认将张量加载到 CPU，不会自动跟随模型设备）
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device or "cpu")

    return training_state            # 返回完整训练状态字典，供调用方提取 step、data_progress 等


# ─── 参数量统计 ───────────────────────────────────────────────────────
def count_parameters(model: nn.Module) -> int:
    """
    统计模型实际可训练参数量。
    注意：lm_head 与 embed_tokens 共享权重，nn.Module 会自动去重，
    因此本函数返回值比 ModelConfig.count_parameters() 小约 vocab_size × hidden_size。
    """
    # p.numel()：返回张量 p 中元素的总数
    # requires_grad=True：只统计需要更新的参数（排除 frozen 层）
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─── 梯度范数计算（用于监控训练稳定性）────────────────────────────────
def get_grad_norm(model: nn.Module) -> float:  # 返回所有参数梯度的总 L2 范数，用于监控训练稳定性
    # 计算所有参数梯度的 L2 范数（欧几里得范数）
    # 梯度范数突然变大通常意味着训练不稳定（梯度爆炸）
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:       # 没有参与本次前向传播的参数梯度为 None，跳过
            total_norm += p.grad.data.norm(2).item() ** 2  # 每个参数的 L2 范数平方后累加
    return total_norm ** 0.5         # 开平方得到总梯度范数