# new-model-Solmen 开发文档

记录各文件的目的、使用方法、已修复 Bug 及相关文件。

---

## 文件说明

### expand_tokenizer.py
- **目的**：在 `New-Model-200M/checkpoint-13000/tokenizer.json` 基础上扩充词表，增加 LaTeX（主）、英语科技词汇（次）、化学符号（辅），将词表对齐到 128 的倍数（目标 53376）
- **使用**：`python expand_tokenizer.py`，输出到 `tokenizer/` 目录
- **注意**：只需运行一次；训练前必须先运行此脚本生成 `tokenizer/`

### config.py
- **目的**：集中管理所有超参数，是新手唯一需要修改的文件
- **使用**：修改顶部 `PRESET` 变量选择模型规模（"50M" / "80M" / "150M" / "200M" / "300M" / "500M" / "1.7B"），或直接修改对应预设字典内的参数
- **注意**：`vocab_size` 固定为 53376，与分词器一致；直接运行可打印当前规模摘要
- **参数量说明**：`ModelConfig.count_parameters()` 含 embedding 未去重，比 `utils.count_parameters(model)` 偏大约 `vocab_size × hidden_size`（200M 规模约多 55M），属正常现象（权重共享）
- **关键接口**：`get_config(preset)` 返回 `(ModelConfig, TrainConfig)`

### model.py
- **目的**：LLaMA Decoder-only 模型定义
- **架构**：RMSNorm（Pre-Norm）+ RoPE + SwiGLU + GQA + PyTorch 内置 Flash Attention
- **关键接口**：`SolmenModel(cfg)` 构造模型；`model(input_ids, labels)` 返回 `(logits, loss)`；`model.generate(...)` 做文本生成
- **注意**：lm_head 与 embedding 共享权重，节省参数；直接运行可做冒烟测试

### dataset.py
- **目的**：流式预训练数据集，支持多 .jsonl 文件顺序读取，记录当前训练到的文件/行号，支持断点续读
- **格式要求**：`.jsonl`，每行 `{"text": "..."}`
- **关键接口**：`PretrainDataset(file_list, tokenizer, seq_len, resume_file_idx, resume_line)`；`get_progress(dataset)` 返回进度字典

### utils.py
- **目的**：公共工具函数
- **功能**：
  - `Logger`：带时间戳的日志，同步写文件
  - `get_lr(step, ...)`：余弦退火 + 线性 Warmup 学习率
  - `CheckpointManager`：HF 格式 checkpoint 保存，自动轮换保留最近 N 个
  - `load_checkpoint(ckpt_dir, model, optimizer, device)`：恢复模型权重和优化器状态，返回含 `scaler_state_dict` 的训练状态字典

### pretrain.py
- **目的**：预训练主脚本
- **使用**：
  1. 修改顶部 `DATA_FILES` 列表填入实际数据路径
  2. 在 `config.py` 中设置 `PRESET` 选择模型规模
  3. 运行 `python pretrain.py`
- **功能**：bf16 混合精度、梯度累积、余弦学习率、自动断点续训、checkpoint 轮换（保留最近 3 个）、训练日志写入 `checkpoints/train.log`
- **断点续训**：自动检测 `checkpoints/` 下最新 checkpoint，无需手动指定

### tokenizer/（目录）
- **目的**：扩充后的分词器文件
- **生成方式**：运行 `expand_tokenizer.py` 自动生成
- **内容**：`tokenizer.json`（53376 词表）+ `tokenizer_config.json`

---

## Bug 记录

| 编号 | 日期       | 严重程度 | 涉及文件 | 描述 | 状态 |
|------|------------|----------|----------|------|------|
| B01  | 2026-04-18 | 🔴 极高   | `model.py` | `apply_rope` 中 cos/sin buffer 为 FP32，bf16 训练时 Q/K 被提升为 FP32，与 V（BF16）dtype 不一致，SDPA fallback 到 math backend 导致 OOM 或报错 | ✅ 已修复 |
| B02  | 2026-04-18 | 🔴 高     | `pretrain.py` | 数据集提前耗尽时，for 循环自然退出，1~(G-1) 个 micro-batch 的已累积梯度永不 step，且污染优化器动量状态影响续训 | ✅ 已修复 |
| B03  | 2026-04-18 | 🟠 中     | `model.py` | `_init_weights` 中 `"embed_tokens" not in name` 无法过滤 `lm_head.weight`，导致与 embedding 共享的权重 tensor 被二次覆盖初始化 | ✅ 已修复 |
| B04  | 2026-04-18 | 🟠 中     | `utils.py` | `CheckpointManager.save()` 忽略传入的 `tokenizer` 参数，硬编码从 `tokenizer/` 目录复制文件；目录不存在时 checkpoint 内无 tokenizer，续训直接报错 | ✅ 已修复 |
| B05  | 2026-04-18 | 🟠 中     | `expand_tokenizer.py` | 只更新 `added_tokens` 列表，未同步写入 `added_tokens_decoder` 字典；HuggingFace tokenizer decode 新 token 时依赖后者，缺失导致新 token 被跳过，实际词表小于 53376，可能触发 embedding 越界 | ✅ 已修复 |
| B06  | 2026-04-18 | 🟠 中     | `pretrain.py` | 学习率在 `global_step += 1` 之前计算，step 0 时 `get_lr(0, warmup=2000) = 0`，第一次参数更新完全无效（lr=0） | ✅ 已修复 |
| B07  | 2026-04-18 | 🟡 低     | `utils.py` / `pretrain.py` | fp16 训练时 GradScaler 状态（scale 值）未随 checkpoint 保存/恢复，续训后 scale 从默认值 65536 重置，可能导致前几步 NaN | ✅ 已修复 |
| B08  | 2026-04-18 | 🟡 低     | `model.py` | GQA 中 `unsqueeze+expand+reshape` 产生非连续 tensor，reshape 触发隐式内存拷贝，改用语义明确的 `repeat_interleave` | ✅ 已修复 |
| B09  | 2026-04-18 | 🟡 低     | `dataset.py` | `tokenizer.encode()` 默认 `add_special_tokens=True`，部分 tokenizer 配置自动追加 EOS，与代码手动 append 重复，导致双 EOS 污染序列边界 | ✅ 已修复 |
| B10  | 2026-04-18 | 🟢 极低   | `pretrain.py` | `get_grad_norm` 被导入但从未调用（梯度范数通过 `clip_grad_norm_` 返回值获取），为死代码 | ✅ 已修复 |
| B11  | 2026-04-18 | 🟢 极低   | `config.py` | `count_parameters` 注释写"不含 embedding"，但代码包含了；200M 规模下导致参数量显示比实际偏大约 53M，误导新手 | ✅ 已修复 |

---

## 变更历史

| 日期       | 内容                                                       |
|------------|------------------------------------------------------------|
| 2026-04-17 | 创建 dev_log.md，项目框架确立                              |
| 2026-04-17 | 完成阶段一全部文件：expand_tokenizer.py / config.py / model.py / dataset.py / utils.py / pretrain.py |
| 2026-04-18 | Bug 审查，修复全部 11 项问题（B01~B11），详见 Bug 记录表   |
