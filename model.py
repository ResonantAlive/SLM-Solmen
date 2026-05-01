# model.py
# 用途：LLaMA Decoder-only 稠密模型定义
# 架构：RMSNorm（Pre-Norm）+ RoPE + SwiGLU + GQA
# 支持 HuggingFace 格式保存/加载

import math                          # 用于权重初始化时的 sqrt 计算
import torch                         # PyTorch 核心库，提供张量操作和自动求导
import torch.nn as nn                # 神经网络模块：Linear、Embedding、Module 等
import torch.nn.functional as F      # 函数式接口：softmax、silu、cross_entropy 等（无可学习参数）
from typing import Optional          # 类型标注：Optional[X] 表示"X 或 None"

from config import ModelConfig       # 导入我们自定义的模型配置类


# ─── RMSNorm ────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    """均方根归一化，比 LayerNorm 更快，LLaMA 系列标配"""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()           # 必须调用父类 __init__，初始化 nn.Module 内部状态
        self.eps = eps               # 数值稳定项，防止均方根为零时除以零
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习的缩放参数，初始化为全 1（即不缩放）

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 升到 fp32 计算范数，避免 fp16 下 pow(2) 溢出导致 NaN
        # x.float()：临时转为 fp32，不改变原始 x 的 dtype
        # .pow(2)：每个元素平方
        # .mean(-1, keepdim=True)：在最后一维（特征维）求均值，keepdim 保持维度方便广播
        # torch.rsqrt(...)：逐元素开方后取倒数，即 1/sqrt(...)
        norm = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return norm.to(x.dtype) * self.weight  # 转回原始 dtype，再乘以可学习缩放参数


# ─── RoPE 旋转位置编码 ────────────────────────────────────────────────
def precompute_rope_freqs(
    head_dim: int,       # 每个注意力头的维度（hidden_size // num_heads）
    max_seq_len: int,    # 支持的最大序列长度
    theta: float = 10000.0,          # 基频，控制不同维度的旋转频率范围
    device: Optional[torch.device] = None,  # 计算设备，None 时默认 CPU
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    预计算 RoPE 的 cos/sin 缓存

    RoPE 的核心思想：把 token 的位置信息编码为"旋转角度"。
    直觉类比：想象一个钟表，指针指向不同角度代表不同位置。
    位置 0 → 转 0°，位置 1 → 转 θ°，位置 2 → 转 2θ°...
    这样两个 token 的"相对位置"可以通过旋转向量的角度差来表示。

    返回形状：(max_seq_len, head_dim // 2)
    注意：buffer 以 FP32 存储，apply_rope 内会按需转换到 Q/K 的实际 dtype
    """
    # 生成频率向量：freqs[i] = 1 / (theta ^ (2i / head_dim))，i = 0,1,...,head_dim/2-1
    # torch.arange(0, head_dim, 2)：[0, 2, 4, ..., head_dim-2]，步长为 2 取偶数索引
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    # 位置索引：[0, 1, 2, ..., max_seq_len-1]
    positions = torch.arange(max_seq_len, device=device).float()
    # 外积：angles[pos, i] = positions[pos] * freqs[i]，形状 (max_seq_len, head_dim//2)
    angles = torch.outer(positions, freqs)
    cos = torch.cos(angles)          # 余弦值缓存，形状 (max_seq_len, head_dim//2)
    sin = torch.sin(angles)          # 正弦值缓存，形状 (max_seq_len, head_dim//2)
    return cos, sin


def apply_rope(
    q: torch.Tensor,    # Query，形状 (batch, num_heads, seq_len, head_dim)
    k: torch.Tensor,    # Key，形状 (batch, num_kv_heads, seq_len, head_dim)
    cos: torch.Tensor,  # 余弦缓存，形状 (seq_len, head_dim//2)
    sin: torch.Tensor,  # 正弦缓存，形状 (seq_len, head_dim//2)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    对 Q、K 施加旋转位置编码
    q/k 形状：(batch, num_heads, seq_len, head_dim)
    cos/sin 形状：(seq_len, head_dim // 2)

    Bug fix: cos/sin buffer 默认 FP32，但 bf16 训练下 Q/K 是 BF16。
    BF16 × FP32 = FP32，会导致 Q/K 变成 FP32，与 V（BF16）dtype 不一致，
    SDPA 要么报错要么 fallback 到 math backend（显存暴增 → OOM）。
    修复：在计算前将 cos/sin 强制转换到与 Q/K 相同的 dtype。
    """
    def rotate(x: torch.Tensor) -> torch.Tensor:
        # RoPE 旋转辅助函数：对单个张量执行旋转操作
        # 将 head_dim 均分为前半和后半
        x1 = x[..., : x.shape[-1] // 2]   # 前半部分，形状 (..., head_dim//2)
        x2 = x[..., x.shape[-1] // 2 :]   # 后半部分，形状 (..., head_dim//2)
        # 旋转操作：[-x2, x1] 相当于将向量旋转 90 度
        rotated = torch.cat([-x2, x1], dim=-1)
        # ↓ 关键修复：转换到 x 的实际 dtype（bf16 / fp16 / fp32）
        # unsqueeze(0).unsqueeze(0) 在 batch 和 head 维度各增加一维，方便广播
        c = cos.unsqueeze(0).unsqueeze(0).to(x.dtype)   # (1, 1, seq, hd/2)
        s = sin.unsqueeze(0).unsqueeze(0).to(x.dtype)
        # 将 cos/sin 从 head_dim//2 扩展到 head_dim（前后两半用同样的值）
        c = torch.cat([c, c], dim=-1)      # (1, 1, seq, head_dim)
        s = torch.cat([s, s], dim=-1)
        # RoPE 公式：x_rotated = x * cos + rotate(x) * sin
        return x * c + rotated * s

    return rotate(q), rotate(k)      # 对 Q 和 K 分别施加旋转位置编码


# ─── GQA 注意力 ──────────────────────────────────────────────────────
class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA)
    num_kv_heads < num_heads 时，多个 Q 头共享一组 KV，节省显存
    num_kv_heads == num_heads 时退化为标准 MHA
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_heads = cfg.num_heads        # Q 的头数
        self.num_kv_heads = cfg.num_kv_heads  # KV 的头数（≤ num_heads）
        self.head_dim = cfg.head_dim()        # 每个头的维度
        self.groups = cfg.num_heads // cfg.num_kv_heads  # 每组 KV 对应多少个 Q 头

        # GQA 要求 num_heads 必须能被 num_kv_heads 整除，否则无法均分
        assert cfg.num_heads % cfg.num_kv_heads == 0, (
            f"num_heads ({cfg.num_heads}) 必须能被 num_kv_heads ({cfg.num_kv_heads}) 整除"
        )

        # Q 投影：hidden_size → num_heads × head_dim
        # bias=False：LLaMA 架构不使用注意力层的偏置项，节省参数且效果相当
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_heads * self.head_dim, bias=False)
        # K 投影：hidden_size → num_kv_heads × head_dim（GQA 下 K 的头数更少）
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * self.head_dim, bias=False)
        # V 投影：同 K
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * self.head_dim, bias=False)
        # 输出投影：将所有头的输出拼接后映射回 hidden_size
        self.o_proj = nn.Linear(cfg.num_heads * self.head_dim, cfg.hidden_size, bias=False)

        self.dropout = cfg.dropout            # 注意力 dropout 概率

    def forward(
        self,
        x: torch.Tensor,             # 输入，形状 (batch, seq_len, hidden_size)
        cos: torch.Tensor,           # RoPE 余弦缓存
        sin: torch.Tensor,           # RoPE 正弦缓存
        mask: Optional[torch.Tensor] = None,  # 注意力掩码，None 时使用因果掩码（自动下三角）
    ) -> torch.Tensor:
        B, T, _ = x.shape            # B=batch大小，T=序列长度，_=hidden_size（不需要用）

        # 线性投影后 reshape 成多头格式，再 transpose 把 head 维提前
        # view(B, T, num_heads, head_dim)：重塑形状
        # transpose(1, 2)：交换 seq 和 head 维 → (B, num_heads, T, head_dim)
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 对 Q 和 K 施加旋转位置编码（V 不需要位置编码）
        # cos[:T], sin[:T]：只取当前序列长度对应的部分
        q, k = apply_rope(q, k, cos[:T], sin[:T])

        # Bug fix: 原写法 unsqueeze+expand+reshape 产生非连续 tensor，
        # reshape 会隐式触发内存拷贝，且 stride 不标准可能导致 Flash Attention 异常。
        # 改用 repeat_interleave：行为明确、内存连续、无歧义。
        if self.groups > 1:
            # 将 KV 头重复 groups 次，对齐 Q 的头数
            # 例如 num_heads=16, num_kv_heads=8, groups=2
            # k 从 (B,8,T,hd) 变为 (B,16,T,hd)，每个 KV 头被复制 2 份
            k = k.repeat_interleave(self.groups, dim=1)  # (B, num_heads, T, head_dim)
            v = v.repeat_interleave(self.groups, dim=1)

        # 训练时用 dropout，推理时关闭
        dropout_p = self.dropout if self.training else 0.0
        # PyTorch 内置的 Scaled Dot-Product Attention，自动选择 Flash Attention 等高效后端
        # is_causal=True（mask 为 None 时）：自动应用因果掩码（下三角），防止看到未来 token
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=dropout_p,
            is_causal=(mask is None),  # 没有显式 mask 时启用因果模式
        )

        # 把多头输出拼回去：(B, num_heads, T, head_dim) → (B, T, hidden_size)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        # contiguous()：确保内存连续，view() 要求内存连续
        return self.o_proj(out)       # 输出投影，映射回 hidden_size


# ─── SwiGLU 前馈网络 ──────────────────────────────────────────────────
class FeedForward(nn.Module):
    """
    SwiGLU FFN：FFN(x) = (Swish(gate(x)) ⊙ up(x)) · down
    LLaMA 使用此结构替代标准 FFN，参数利用率更高
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        # gate_proj 和 up_proj 都是升维到 intermediate_size
        # SwiGLU 用两个升维矩阵的乘积做门控，比标准 FFN 多一个矩阵但效果更好
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        # down_proj 将中间层降维回 hidden_size
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.silu：Sigmoid Linear Unit，即 x * sigmoid(x)，比 ReLU 更平滑
        # gate_proj 经过 SiLU 激活后与 up_proj 的输出逐元素相乘（门控机制）
        # 最后通过 down_proj 降维
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ─── Transformer 块 ───────────────────────────────────────────────────
class TransformerBlock(nn.Module):
    """Pre-Norm Transformer 块：Norm → Attn → 残差 → Norm → FFN → 残差"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)  # 注意力前的归一化
        self.attn = GroupedQueryAttention(cfg)                        # 注意力层
        self.ffn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)   # FFN 前的归一化
        self.ffn = FeedForward(cfg)                                   # 前馈网络

    def forward(
        self,
        x: torch.Tensor,             # 输入隐状态，形状 (batch, seq_len, hidden_size)
        cos: torch.Tensor,           # RoPE 余弦缓存
        sin: torch.Tensor,           # RoPE 正弦缓存
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-Norm：先归一化再进注意力，然后残差连接（x + ...）
        # 残差连接让梯度可以"跳过"该层直接传回，缓解深层网络的梯度消失问题
        x = x + self.attn(self.attn_norm(x), cos, sin, mask)
        # 同理：先归一化再进 FFN，然后残差连接
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ─── 完整模型 ─────────────────────────────────────────────────────────
class SolmenModel(nn.Module):
    """
    Solmen LLM：LLaMA 架构的 Decoder-only 语言模型
    支持规模：50M ~ 1.7B（由 ModelConfig 控制）
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg               # 保存配置，后续 forward 和 save 都会用到

        # Embedding 层：将 token id（整数）映射为 hidden_size 维的向量
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        # 堆叠 num_layers 个 Transformer 块，ModuleList 让 PyTorch 能正确追踪这些子模块
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        # 最终输出归一化，在 lm_head 之前做
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        # lm_head 与 embedding 共享权重，节省参数量
        # 直觉：token id → 向量（embedding）和 向量 → token id 概率（lm_head）
        # 本质是同一个查表过程的正反两面，共用一套"字典"是合理的
        # 这行创建线性层，下一行将其权重指向 embed_tokens.weight（同一块内存）
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight  # 权重共享：两个层用同一个权重矩阵

        # 预计算 RoPE 的 cos/sin 缓存，注册为 buffer（不是参数，不参与梯度更新）
        # register_buffer 的好处：.to(device) 时自动跟随模型移动，保存/加载时可选持久化
        cos, sin = precompute_rope_freqs(
            cfg.head_dim(), cfg.max_seq_len, cfg.rope_theta
        )
        # persistent=False：不保存到 checkpoint，因为可以随时从配置重新计算
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self._init_weights()         # 初始化所有层的权重

    def _init_weights(self):
        """遵循 LLaMA 初始化策略：正态分布，std 与层数相关"""
        std = 0.02                   # 基础标准差，来自 GPT-2 论文经验值
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=std)  # Embedding 用正态初始化
        for name, param in self.named_parameters():  # 遍历所有具名参数
            if "weight" in name and param.dim() == 2:  # 只处理 2D 权重矩阵（跳过 1D 的 norm.weight）
                if "o_proj" in name or "down_proj" in name:
                    # 残差分支末尾（o_proj、down_proj）用更小的 std
                    # 1/sqrt(2*num_layers) 确保堆叠多层后残差流的方差不爆炸
                    nn.init.normal_(param, mean=0.0, std=std / math.sqrt(2 * self.cfg.num_layers))
                elif "embed_tokens" not in name and "lm_head" not in name:
                    # Bug fix: lm_head.weight 与 embed_tokens.weight 是同一 tensor，
                    # 原代码 "embed_tokens" not in name 无法过滤 lm_head.weight，
                    # 导致该 tensor 被第二次覆盖初始化。现增加 "lm_head" not in name 保护。
                    nn.init.normal_(param, mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,     # 输入 token id，形状 (batch, seq_len)
        labels: Optional[torch.Tensor] = None,  # 标签（下一个 token），为 None 时只返回 logits
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        input_ids: (batch, seq_len)
        labels:    (batch, seq_len)，与 input_ids 错位一位（下一个 token 预测）
        返回：(logits, loss)
        """
        B, T = input_ids.shape       # 取出 batch 大小和序列长度
        # 检查序列长度不超过模型支持的最大值
        assert T <= self.cfg.max_seq_len, (
            f"输入序列长度 {T} 超过最大序列长度 {self.cfg.max_seq_len}"
        )

        # 将 token id 查表转为向量，x 形状变为 (batch, seq_len, hidden_size)
        x = self.embed_tokens(input_ids)

        # 逐层前向传播，rope_cos/rope_sin 是预计算好的位置编码缓存
        for layer in self.layers:
            x = layer(x, self.rope_cos, self.rope_sin)  # 每层输入 x，输出更新后的隐状态

        x = self.norm(x)             # 最终 RMSNorm 归一化
        logits = self.lm_head(x)     # 线性投影到词表维度，形状 (batch, seq_len, vocab_size)

        loss = None                    # 默认不计算 loss（推理时 labels 为 None）
        if labels is not None:         # 传入了标签则计算交叉熵损失
            # cross_entropy 计算交叉熵损失
            # view(-1, vocab_size)：把 batch 和 seq 两个维度合并成一个，方便计算
            # ignore_index=-100：标签为 -100 的位置不参与 loss 计算（padding 位置的标准做法）
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss            # 返回预测分布和损失（loss 可能为 None）

    def generate(
        self,
        input_ids: torch.Tensor,     # prompt 的 token ids，形状 (1, prompt_len)
        max_new_tokens: int = 128,   # 最多生成多少个新 token
        temperature: float = 1.0,   # 温度：越高越随机，越低越保守（→ 0 趋近于贪心解码）
        top_p: float = 0.9,         # 核采样阈值：只从累积概率前 90% 的 token 中采样
        eos_token_id: Optional[int] = None,  # 结束符 id，生成到此 token 时提前停止
    ) -> torch.Tensor:
        """
        自回归文本生成（推理阶段使用）
        input_ids: (1, prompt_len)
        返回：(1, prompt_len + max_new_tokens) 的 token ids
        """
        was_training = self.training  # 保存当前模式，生成结束后恢复
        self.eval()                  # 切换到推理模式（关闭 dropout 等训练专属行为）
        with torch.no_grad():        # 推理时不需要计算梯度，节省显存和计算量
            for _ in range(max_new_tokens):  # 逐个生成新 token，最多 max_new_tokens 个
                # 如果已生成序列超过 max_seq_len，只取最后 max_seq_len 个 token 作为上下文
                context = input_ids[:, -self.cfg.max_seq_len:]
                logits, _ = self.forward(context)   # 前向传播，只关心 logits，不需要 loss
                # 取最后一个位置的 logits（即"下一个 token"的预测分布）
                next_logits = logits[:, -1, :]      # 形状 (1, vocab_size)

                # 温度缩放：logits / temperature，让分布变尖锐（低温）或平坦（高温）
        # 类比：高温像"骰子"更随机，低温像"老师划重点"更确定
                if temperature != 1.0:
                    next_logits = next_logits / temperature

                # Top-p 核采样：只保留累积概率达到 top_p 的最高概率 token
                if top_p < 1.0:            # top_p=1.0 时不进行过滤，直接用完整分布采样
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    probs = F.softmax(sorted_logits, dim=-1)       # 转为概率
                    cumprobs = torch.cumsum(probs, dim=-1)         # 累积概率
                    # 标记累积概率超过 top_p 的位置（需要过滤掉）
                    # 减去 probs 是为了保留刚好超过阈值的那个 token
                    sorted_indices_to_remove = cumprobs - probs > top_p
                    # 将被过滤的 token 的 logit 设为 -inf（softmax 后概率为 0）
                    sorted_logits[sorted_indices_to_remove] = float("-inf")
                    # 将过滤后的 logits 按原始顺序写回
                    next_logits = torch.zeros_like(next_logits).scatter_(
                        1, sorted_indices, sorted_logits
                    )

                probs = F.softmax(next_logits, dim=-1)     # 转为概率分布
                # 按概率采样一个 token（比 argmax 更有多样性）
                next_token = torch.multinomial(probs, num_samples=1)  # 形状 (1, 1)
                # 将新生成的 token 拼接到序列末尾
                input_ids = torch.cat([input_ids, next_token], dim=1)

                # 遇到 EOS token 则提前停止
                if eos_token_id is not None and next_token.item() == eos_token_id:
                    break

        self.train(was_training)      # 恢复调用 generate() 之前的训练/推理模式
        return input_ids             # 返回完整序列（包含 prompt + 生成内容）


# ─── 快速验证 ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    from config import get_config      # 导入配置获取函数

    m_cfg, _ = get_config()            # 取模型配置，忽略训练配置
    model = SolmenModel(m_cfg)

    # numel()：返回张量中元素的总数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量：{total_params / 1e6:.1f}M")

    # 随机生成假数据做一次前向传播，验证模型结构没有 bug
    dummy_ids = torch.randint(0, m_cfg.vocab_size, (2, 64))    # batch=2, seq_len=64
    dummy_labels = torch.randint(0, m_cfg.vocab_size, (2, 64))  # 不复用 input_ids
    logits, loss = model(dummy_ids, dummy_labels)  # 执行一次前向传播，验证模型能跑通
    print(f"logits shape：{logits.shape}")   # 期望 (2, 64, vocab_size)
    print(f"loss：{loss.item():.4f}")        # 期望约 log(vocab_size) ≈ 11.0
    print("模型前向传播正常")