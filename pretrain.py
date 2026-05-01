# pretrain.py
# 用途：预训练主脚本
# 功能：bf16 混合精度、梯度累积、余弦学习率、断点续训、自动 checkpoint 轮换（保留最近 3 个）

import os                            # 环境变量访问（备用，当前逻辑主要用 pathlib）
import torch                         # PyTorch 核心：张量、设备、自动混合精度
from pathlib import Path             # 面向对象的路径操作
from torch.utils.data import DataLoader  # 将 Dataset 包装成可迭代的 batch 加载器
from transformers import PreTrainedTokenizerFast  # HuggingFace 快速分词器

from config import get_config, PRESET  # 获取模型和训练配置，PRESET 是当前选择的规模字符串
from model import SolmenModel          # 模型定义
from dataset import PretrainDataset, get_progress  # 数据集和进度获取函数
# 从 utils 导入训练所需的工具函数
# 注意：get_grad_norm 已不再需要，梯度范数由 clip_grad_norm_ 的返回值直接获取
from utils import Logger, CheckpointManager, load_checkpoint, get_lr, count_parameters

# ═══════════════════════════════════════════════════════════════════════
# 预训练数据文件列表（按训练顺序排列，修改此处换数据集）
# 格式要求：.jsonl，每行 {"text": "..."}
DATA_FILES = [
    "/path/to/your/pretrain_data_1.jsonl",  # 替换为实际路径
    "/path/to/your/pretrain_data_2.jsonl",  # 替换为实际路径
]
# ═══════════════════════════════════════════════════════════════════════


def main():                            # 预训练主函数，包含完整的训练循环
    # ── 加载配置 ───────────────────────────────────────────────────────
    model_cfg, train_cfg = get_config(PRESET)  # 根据 PRESET 获取模型和训练配置

    # ── 设备与精度 ────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    amp_dtype = dtype_map[train_cfg.dtype]  # 自动混合精度使用的 dtype

    # ── 日志与 Checkpoint 管理器 ──────────────────────────────────────
    output_dir = Path(train_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(log_file=str(output_dir / "train.log"))
    ckpt_manager = CheckpointManager(train_cfg.output_dir, keep=train_cfg.keep_checkpoints)

    logger.info(f"预设规模：{PRESET}，设备：{device}，精度：{train_cfg.dtype}")  # 打印启动信息，方便确认配置无误

    # ── 加载分词器 ────────────────────────────────────────────────────
    tokenizer_dir = Path(__file__).parent / "tokenizer"
    assert tokenizer_dir.exists(), (
        f"分词器目录不存在：{tokenizer_dir}\n请先运行 expand_tokenizer.py 生成分词器"
    )
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_dir))
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = "<|endoftext|>"  # 手动设置 EOS token（标记文档边界）
    logger.info(f"分词器词表大小：{len(tokenizer)}")

    # ── 初始化模型 ────────────────────────────────────────────────────
    model = SolmenModel(model_cfg).to(device)  # .to(device) 将模型参数移到 GPU/CPU
    logger.info(f"模型参数量：{count_parameters(model) / 1e6:.1f}M")

    # ── 初始化优化器 ──────────────────────────────────────────────────
    # 将参数分为两组：2D 以上加 weight_decay，1D 参数（norm.weight 等）不加
    decay_params = [p for n, p in model.named_parameters()
                    if p.dim() >= 2 and p.requires_grad]
    no_decay_params = [p for n, p in model.named_parameters()
                       if p.dim() < 2 and p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": train_cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},  # 1D 参数不作权重衰减
        ],
        lr=train_cfg.learning_rate,
        betas=(train_cfg.beta1, train_cfg.beta2),
        fused=True,  # 使用 CUDA fused 实现（更快），仅 CUDA 下有效
    )

    # ── 自动混合精度 Scaler ───────────────────────────────────────────
    # bf16 动态范围与 fp32 一致，不需要 GradScaler。fp16 需要 scaler 防下溢。
    use_scaler = (train_cfg.dtype == "fp16")
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    # ── 断点续训：从最新 checkpoint 恢复 ─────────────────────────────
    start_step = 0                     # 默认从第 0 步开始（会被 checkpoint 覆盖）
    resume_file_idx = 0                # 默认从第 0 个文件开始
    resume_line = 0                    # 默认从文件第 0 行开始

    latest_ckpt = ckpt_manager.latest()  # 查询最新的 checkpoint，None 表示从头训练
    if latest_ckpt is not None:
        logger.info(f"检测到 checkpoint，恢复训练：{latest_ckpt}")
        training_state = load_checkpoint(latest_ckpt, model, optimizer, device)
        start_step = training_state["step"]     # 从第几步继续
        data_prog = training_state.get("data_progress", {})
        resume_file_idx = data_prog.get("file_idx", 0)  # 从第几个文件继续
        resume_line = data_prog.get("line", 0)          # 从第几行继续

        # Bug fix: 恢复 GradScaler 状态，防止 fp16 续训时 scale 值重置导致短暂 NaN
        if use_scaler and "scaler_state_dict" in training_state:
            scaler.load_state_dict(training_state["scaler_state_dict"])
            logger.info("GradScaler 状态已恢复")

        logger.info(
            f"续训起点：step={start_step}，"
            f"数据文件[{resume_file_idx}] 第 {resume_line} 行"
        )
    else:
        logger.info("未发现 checkpoint，从头开始训练")

    # ── 数据集与 DataLoader ───────────────────────────────────────────
    dataset = PretrainDataset(
        file_list=DATA_FILES,
        tokenizer=tokenizer,
        seq_len=model_cfg.max_seq_len,       # 每个样本长度 = 模型最大序列长度
        resume_file_idx=resume_file_idx,
        resume_line=resume_line,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,     # micro-batch 大小
        num_workers=0,                       # 0 表示主进程加载（避免 Windows 多进程问题）
        # Bug fix: CPU 环境下 pin_memory=True 无意义且触发 warning
        pin_memory=(device.type == "cuda"),  # CUDA 下 pin_memory 加速 CPU→GPU 传输
    )

    # ── 训练主循环 ────────────────────────────────────────────────────
    model.train()                    # 切换到训练模式（启用 dropout 等）
    optimizer.zero_grad()            # 清空梯度缓存

    global_step = start_step         # 当前全局步数，从续训点开始
    accum_loss = 0.0                 # 累积 loss（用于日志汇报的平均值）
    accum_count = 0                  # 已累积的 micro-batch 数

    logger.info("开始训练...")          # 训练循环启动标记

    for batch in dataloader:           # 遍历数据加载器，每次 yield 一个 micro-batch
        if global_step >= train_cfg.max_steps:  # 达到预设步数上限，退出循环
            break                        # 跳出训练循环                    # 达到最大步数，停止训练

        input_ids = batch["input_ids"].to(device)  # 输入 token 移到 GPU
        labels    = batch["labels"].to(device)     # 标签也移到 GPU

        # autocast 上下文：前向传播用混合精度，反向传播自动处理
        # 混合精度的原理：大部分计算用 bf16（速度快、显存省），关键地方自动回退到 fp32（精度高）
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            _, loss = model(input_ids, labels)
            loss = loss / train_cfg.gradient_accumulation_steps  # 归一化到单步

        scaler.scale(loss).backward()  # 反向传播（scaler 自动缩放梯度防止 fp16 下溢）
        accum_loss += loss.item() * train_cfg.gradient_accumulation_steps  # 恢复原始 loss
        accum_count += 1

        # 梯度累积：攒够 accumulation_steps 个 micro-batch 再更新参数
        # 例如 batch_size=4, accumulation=8：每次只在 GPU 上跑 4 条，但累积 8 次后
        # 相当于一次性跑了 32 条的大 batch，显存只用了 4 条的量
        if accum_count % train_cfg.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)  # 梯度反缩放（还原真实梯度值）
            # 梯度裁剪：限制梯度范数最大值，防止单步更新过大
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_cfg.grad_clip
            )

            # Bug fix: 原代码先设置 lr 再 global_step += 1，导致 step=0 时
            # get_lr(0, warmup_steps=2000) = 0，第一次参数更新完全无效。
            # 修复：先 +1 再用新 step 计算 lr。
            global_step += 1
            current_lr = get_lr(
                global_step,
                train_cfg.warmup_steps,
                train_cfg.max_steps,
                train_cfg.learning_rate,
                train_cfg.min_lr,
            )
            for pg in optimizer.param_groups:  # 遍历优化器的参数组（weight_decay / no_decay 两组）
                pg["lr"] = current_lr       # 将计算好的学习率写入优化器

            scaler.step(optimizer)          # 执行参数更新
            scaler.update()                 # 更新 scaler 的缩放因子
            optimizer.zero_grad()           # 清空梯度，准备下一轮累积

            avg_loss = accum_loss / train_cfg.gradient_accumulation_steps
            accum_loss = 0.0

            # 按 log_steps 间隔打印训练日志
            if global_step % train_cfg.log_steps == 0:  # 每 log_steps 步打印一次训练日志
                logger.info(
                    f"step {global_step}/{train_cfg.max_steps} | "
                    f"loss {avg_loss:.4f} | "
                    f"lr {current_lr:.2e} | "          # 科学计数法显示学习率
                    f"grad_norm {grad_norm:.3f} | "
                    f"数据进度：文件[{dataset.current_file_idx}] 第{dataset.current_line}行"
                )

            # 按 save_steps 间隔保存 checkpoint
            if global_step % train_cfg.save_steps == 0:  # 每 save_steps 步保存一次 checkpoint
                data_progress = get_progress(dataset)
                ckpt_path = ckpt_manager.save(
                    step=global_step,
                    model=model,
                    optimizer=optimizer,
                    tokenizer=tokenizer,
                    data_progress=data_progress,
                    extra={"loss": avg_loss},
                    scaler=scaler,
                )
                logger.info(f"checkpoint 已保存：{ckpt_path}")

    # ── 训练循环结束后 flush 残余梯度 ────────────────────────────────
    # Bug fix: 若数据集在 max_steps 达到前耗尽，可能有 1~(G-1) 个
    # micro-batch 的梯度已 backward 但未 step，需处理避免丢失。
    remaining = accum_count % train_cfg.gradient_accumulation_steps  # 计算未 flush 的残余累积步数
    if remaining != 0:                 # 有残余梯度未更新，执行最后一次 flush
        logger.info(f"检测到末尾残余梯度（{remaining} 个 micro-batch），执行 flush step...")
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        global_step += 1
        current_lr = get_lr(
            global_step,
            train_cfg.warmup_steps,
            train_cfg.max_steps,
            train_cfg.learning_rate,
            train_cfg.min_lr,
        )
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        logger.info(f"flush 完成，global_step={global_step}，grad_norm={grad_norm:.3f}")

    # ── 保存最终 checkpoint ───────────────────────────────────────────
    logger.info("训练完成，保存最终 checkpoint...")  # 正常结束或数据耗尽时保存最终状态
    data_progress = get_progress(dataset)
    ckpt_path = ckpt_manager.save(
        step=global_step,
        model=model,
        optimizer=optimizer,
        tokenizer=tokenizer,
        data_progress=data_progress,
        scaler=scaler,
    )
    logger.info(f"最终 checkpoint：{ckpt_path}")
    logger.close()


if __name__ == "__main__":             # 直接运行此脚本时执行训练
    main()                             # 启动预训练主循环