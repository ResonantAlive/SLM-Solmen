# config.py
# 用途：集中管理所有超参数，是新手唯一需要修改的文件
# 使用方法：修改下方 PRESET 变量选择模型规模，或直接调整 ModelConfig 中的具体参数

import math                          # Python 内置数学库（本文件未使用，保留供下游扩展）
from dataclasses import dataclass, field  # dataclass 让类自动生成 __init__，写超参数配置非常方便
from typing import Optional          # 类型标注用，Optional[X] 表示"X 或 None"

# ═══════════════════════════════════════════════════════════════════
# 新手入口：在这里选择模型规模
# 可选值："50M" | "80M" | "150M" | "200M" | "300M" | "500M" | "1.7B"
PRESET = "200M"
# ═══════════════════════════════════════════════════════════════════


@dataclass                           # 装饰器：自动生成 __init__，字段直接写成类变量即可
class ModelConfig:
    """模型结构超参数"""

    vocab_size: int = 53376          # 词表大小：分词器能识别多少个不同的 token
    hidden_size: int = 1024          # 隐藏层维度：每个 token 被表示为多少维的向量
    num_layers: int = 16             # Transformer 块的层数，越深模型表达能力越强
    num_heads: int = 16              # 多头注意力的"头"数，每头独立学习不同的注意力模式
    num_kv_heads: int = 8            # GQA 的 KV 头数，小于 num_heads 时多个 Q 头共享一组 KV，节省显存
    intermediate_size: int = 2752    # FFN 中间层维度，通常约为 hidden_size 的 2.7 倍（SwiGLU 经验值）
    max_seq_len: int = 1024          # 最长输入序列长度（token 数），超过会报错
    rms_norm_eps: float = 1e-5       # RMSNorm 的数值稳定项，防止除以零（1e-5 = 0.00001）
    rope_theta: float = 10000.0      # RoPE 位置编码的基频，控制位置感知的波长范围
    dropout: float = 0.0             # Dropout 概率，预训练阶段通常设 0（不随机丢弃神经元）

    def head_dim(self) -> int:
        # 每个注意力头处理的维度 = 总隐藏维度 / 头数
        # 例如 hidden_size=1024, num_heads=16 → head_dim=64
        assert self.hidden_size % self.num_heads == 0  # 必须能整除，否则无法均分
        return self.hidden_size // self.num_heads   # 整除得到每个注意力头的维度

    def count_parameters(self) -> int:
        """
        估算模型总参数量（含 embedding）。

        注意：本方法用于规模规划，计算时 embedding 与 lm_head 各算一份。
        实际训练中 lm_head 与 embed_tokens 共享权重（tie_word_embeddings=True），
        因此 utils.count_parameters(model) 返回的值会小约 vocab_size × hidden_size
        （200M 规模约少 55M）。两者差异属正常现象，不是 Bug。
        """
        d = self.hidden_size         # 简写，方便下面公式书写
        h = self.head_dim()          # 每头维度
        kv = self.num_kv_heads       # KV 头数
        n = self.num_heads           # Q 头数
        ffn = self.intermediate_size # FFN 中间维度

        # 注意力层参数量：Q 投影 + K 投影 + V 投影 + O 投影
        attn = d * (n * h + kv * h + kv * h + n * h)
        # FFN 参数量：gate_proj + up_proj + down_proj（SwiGLU 有两个升维矩阵）
        ffn_params = d * ffn * 2 + ffn * d
        # 每层有两个 RMSNorm，每个 RMSNorm 参数量 = hidden_size
        norm_params = d * 2
        # 单层 Transformer 块总参数
        per_layer = attn + ffn_params + norm_params

        # Embedding 层参数：词表大小 × 隐藏维度大小
        embed = self.vocab_size * d
        # 最后一层 RMSNorm（输出归一化）
        final_norm = d

        # 总参数 = embedding + 所有层 + 最终 norm
        total = embed + self.num_layers * per_layer + final_norm  # 各部分求和
        return total  # 返回估算的总参数量

    def print_summary(self):
        # 打印人类可读的模型规模摘要，方便核对配置
        total = self.count_parameters()  # 调用上面的参数量估算方法
        print(f"模型参数量（含 embedding，未去重）：{total / 1e6:.1f}M")  # 除以 1e6 转换为百万
        print(f"  hidden_size       = {self.hidden_size}")
        print(f"  num_layers        = {self.num_layers}")
        print(f"  num_heads         = {self.num_heads}")
        print(f"  num_kv_heads      = {self.num_kv_heads}  (GQA 比例 {self.num_heads // self.num_kv_heads}:1)")
        print(f"  intermediate_size = {self.intermediate_size}")
        print(f"  max_seq_len       = {self.max_seq_len}")
        print(f"  vocab_size        = {self.vocab_size}")
        shared = self.vocab_size * self.hidden_size  # 共享权重节省的参数量
        print(f"  （权重共享后实际参数量约 {(total - shared) / 1e6:.1f}M，"
              f"共享节省 {shared / 1e6:.1f}M）")


@dataclass                           # 装饰器：自动生成 __init__，让 TrainConfig 也成为 dataclass
class TrainConfig:
    """训练超参数"""

    learning_rate: float = 3e-4      # 峰值学习率（3e-4 = 0.0003），AdamW 的主要调节旋钮
    min_lr: float = 3e-5             # 余弦退火结束时的最小学习率，通常为 learning_rate 的 1/10
    warmup_steps: int = 2000         # 线性 Warmup 步数：前 2000 步学习率从 0 线性升到 learning_rate
    max_steps: int = 100000          # 总训练步数（梯度更新次数，不是 batch 数）
    batch_size: int = 4              # 每次喂给 GPU 的样本数（micro-batch），受显存限制
    gradient_accumulation_steps: int = 8  # 梯度累积步数：累积 8 个 micro-batch 再更新一次参数
    weight_decay: float = 0.1        # L2 正则化强度，防止权重过大（仅对 2D 以上参数生效）
    beta1: float = 0.9               # AdamW 一阶动量系数（梯度的指数移动平均）
    beta2: float = 0.95              # AdamW 二阶动量系数（梯度平方的指数移动平均）
    grad_clip: float = 1.0           # 梯度裁剪阈值：梯度范数超过 1.0 时按比例缩小，防止梯度爆炸
    dtype: str = "bf16"              # 训练精度："bf16"（推荐）| "fp16" | "fp32"
    save_steps: int = 1000           # 每隔多少步保存一次 checkpoint
    keep_checkpoints: int = 3        # 磁盘上最多保留多少个 checkpoint（自动删除旧的）
    output_dir: str = "./checkpoints"  # checkpoint 保存目录
    log_steps: int = 50              # 每隔多少步打印一次训练日志


# ─── 各规模预设配置 ──────────────────────────────────────────────────
# 字典：预设名 → {"model": {...}, "train": {...}}
# 这里集中了所有规模的超参数，get_config() 会从这里取
_PRESETS: dict[str, dict] = {
    "50M": dict(
        model=dict(
            hidden_size=512,         # 50M 规模用较小的隐藏维度
            num_layers=8,            # 层数少，训练快
            num_heads=8,
            num_kv_heads=4,          # GQA 2:1，每 2 个 Q 头共享 1 组 KV
            intermediate_size=1376,
            max_seq_len=512,         # 小模型序列也短一些
        ),
        train=dict(
            learning_rate=5e-4,      # 小模型可以用稍大的学习率
            min_lr=5e-5,
            warmup_steps=1000,       # 线性 Warmup 步数：前 1000 步学习率从 0 渐增到峰值
            max_steps=50000,
            batch_size=16,           # 小模型显存占用少，可以用大 batch
            gradient_accumulation_steps=2,
        ),
    ),
    "80M": dict(
        model=dict(
            hidden_size=640,
            num_layers=12,
            num_heads=10,
            num_kv_heads=5,
            intermediate_size=1728,
            max_seq_len=512,
        ),
        train=dict(
            learning_rate=4e-4,
            min_lr=4e-5,
            warmup_steps=1500,
            max_steps=80000,
            batch_size=12,
            gradient_accumulation_steps=4,
        ),
    ),
    "150M": dict(
        model=dict(
            hidden_size=768,
            num_layers=16,
            num_heads=12,
            num_kv_heads=4,
            intermediate_size=2048,
            max_seq_len=1024,
        ),
        train=dict(
            learning_rate=3e-4,
            min_lr=3e-5,
            warmup_steps=2000,
            max_steps=100000,
            batch_size=8,
            gradient_accumulation_steps=4,
        ),
    ),
    "200M": dict(
        model=dict(
            hidden_size=1024,
            num_layers=16,
            num_heads=16,
            num_kv_heads=8,          # GQA 2:1
            intermediate_size=2752,
            max_seq_len=1024,
        ),
        train=dict(
            learning_rate=3e-4,
            min_lr=3e-5,
            warmup_steps=2000,
            max_steps=100000,
            batch_size=4,
            gradient_accumulation_steps=8,  # 有效 batch = 4×8 = 32
        ),
    ),
    "300M": dict(
        model=dict(
            hidden_size=1024,        # 300M 与 200M 同宽，靠增加层数（24层）提升容量
            num_layers=24,
            num_heads=16,
            num_kv_heads=8,
            intermediate_size=2752,
            max_seq_len=1024,
        ),
        train=dict(
            learning_rate=2e-4,      # 模型越大，学习率通常要适当降低
            min_lr=2e-5,
            warmup_steps=2000,
            max_steps=150000,
            batch_size=4,
            gradient_accumulation_steps=8,
        ),
    ),
    "500M": dict(
        model=dict(
            hidden_size=1280,
            num_layers=24,
            num_heads=20,
            num_kv_heads=5,
            intermediate_size=3456,
            max_seq_len=2048,        # 更大的模型支持更长的上下文
        ),
        train=dict(
            learning_rate=2e-4,
            min_lr=2e-5,
            warmup_steps=3000,
            max_steps=200000,
            batch_size=2,            # 大模型显存紧张，batch 变小
            gradient_accumulation_steps=16,  # 靠更多累积步补回有效 batch
        ),
    ),
    "1.7B": dict(
        model=dict(
            hidden_size=2048,        # 1.7B 规模需要大幅增加宽度
            num_layers=24,
            num_heads=16,
            num_kv_heads=8,
            intermediate_size=5504,
            max_seq_len=2048,
        ),
        train=dict(
            learning_rate=1e-4,      # 大模型学习率要更保守
            min_lr=1e-5,
            warmup_steps=5000,       # Warmup 步数也相应增加
            max_steps=500000,
            batch_size=2,
            gradient_accumulation_steps=16,
        ),
    ),
}


def get_config(preset: str = PRESET) -> tuple[ModelConfig, TrainConfig]:
    """
    根据预设名称返回 (ModelConfig, TrainConfig)
    preset 可选："50M" | "80M" | "150M" | "200M" | "300M" | "500M" | "1.7B"
    """
    # 检查传入的预设名是否合法，不合法时给出可选列表提示
    assert preset in _PRESETS, (
        f"未知预设 '{preset}'，可选：{list(_PRESETS.keys())}"
    )
    p = _PRESETS[preset]                      # 取出对应规模的配置字典
    model_cfg = ModelConfig(**p["model"])     # ** 解包字典，等价于 ModelConfig(hidden_size=..., ...)
    train_cfg = TrainConfig(**p["train"])     # 同上
    return model_cfg, train_cfg         # 返回配置元组，调用方用解包接收


if __name__ == "__main__":
    # 直接运行此文件时，打印当前预设的完整配置摘要，便于核对
    m_cfg, t_cfg = get_config(PRESET)
    print(f"当前预设：{PRESET}")
    print("─" * 40)
    m_cfg.print_summary()
    print("─" * 40)
    print(f"学习率：{t_cfg.learning_rate}  Warmup：{t_cfg.warmup_steps} 步")
    print(f"batch_size：{t_cfg.batch_size}  梯度累积：{t_cfg.gradient_accumulation_steps}")
    print(f"有效 batch：{t_cfg.batch_size * t_cfg.gradient_accumulation_steps} 个样本/步")
    print(f"总步数：{t_cfg.max_steps}")
    print(f"混合精度：{t_cfg.dtype}")