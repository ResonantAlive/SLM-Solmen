# Solmen LLM

> 中文版

一个从零实现的 LLaMA 架构语言模型预训练框架，支持 50M 到 1.7B 参数规模。

## 注意，如果训练的模型你希望可以输出数学式、化学式，你需要使用"expand_tokenizer.py"将缺失的Latex、ChatML token补回，如果你只是训练一个Demo，可以忽略，不过我们仍推荐你补全，以保证tensor的充分利用，提升MFU
\
tokenizer在中文数据集上训练，对于英语支持能力较弱，如果必须的话，你可以尝试 
* 1.训练自己的tokenizer(我们推荐 英文45% 中文40% 代码 15%） ；
* 2.使用LLama、Qwen的tokenizer，但是你可能会受到更大的tokenizer带来的副作用即Embedding巨大、Logist占用超18G


## 架构一览


```
输入 token → Embedding → [RMSNorm → GQA → 残差 → RMSNorm → SwiGLU FFN → 残差] × N → RMSNorm → LM Head → 输出 logits
```



核心技术栈：

| 技术 | 一句话说明 |
|------|-----------|
| RMSNorm（Pre-Norm） | 比 LayerNorm 更快的归一化，LLaMA 系列标配 |
| RoPE 旋转位置编码 | 通过旋转向量编码相对位置，支持外推到更长序列 |
| GQA（Grouped Query Attention） | 多个 Q 头共享一组 KV，节省显存 |
| SwiGLU FFN | 用门控机制替代 ReLU，参数利用率更高 |
| Flash Attention | PyTorch 内置高效注意力，自动选择最优后端 |
| 权重共享 | embedding 与 lm_head 共用同一套参数，节省约 50M 参数 |

## 支持规模

| 预设 | 参数量 | hidden_size | 层数 | heads | KV heads | FFN 维度 | 最大序列长度 |
|------|--------|-------------|------|-------|----------|----------|-------------|
| 50M  | ~50M   | 512         | 8    | 8     | 4        | 1376     | 512         |
| 80M  | ~80M   | 640         | 12   | 10    | 5        | 1728     | 512         |
| 150M | ~150M  | 768         | 16   | 12    | 4        | 2048     | 1024        |
| 200M | ~200M  | 1024        | 16   | 16    | 8        | 2752     | 1024        |
| 300M | ~300M  | 1024        | 24   | 16    | 8        | 2752     | 1024        |
| 500M | ~500M  | 1280        | 24   | 20    | 5        | 3456     | 2048        |
| 1.7B | ~1.7B  | 2048        | 24   | 16    | 8        | 5504     | 2048        |

**切换规模只需改一个地方**：`config.py` 第 12 行的 `PRESET` 变量。所有预设的 `vocab_size` 都固定为 53376（与分词器绑定），不需要额外改动。

## 项目结构

```
new-model-Solmen/
├── config.py               # 超参数配置（模型结构 + 训练参数 + 多规模预设）
├── model.py                # 模型定义（RMSNorm / RoPE / GQA / SwiGLU / 完整模型）
├── dataset.py              # 流式数据集（.jsonl 读取 / 滑窗切分 / 断点续读）
├── utils.py                # 工具函数（日志 / 学习率调度 / checkpoint 管理）
├── pretrain.py             # 训练主脚本（bf16 混合精度 / 梯度累积 / 自动续训）
├── expand_tokenizer.py     # 分词器扩充（LaTeX + 科技词汇 + 化学符号）
├── dev_log.md              # 开发文档 + Bug 记录
├── token count.py          # 数你的数据集包含的token数量
└── tokenizer/              # tokenizer
    ├── tokenizer.json
    └── tokenizer_config.json
```

---

## 手把手教程：从零开始训练

### 第一步：确认 Python 版本

本项目使用了 Python 3.10+ 的语法特性（如 `tuple[int, int]` 类型标注）。请确认你的 Python 版本：

```bash
python --version
```

如果版本低于 3.10，需要先升级 Python。推荐安装 [Python 3.10](https://www.python.org/downloads/) 或更高版本。

> 如果你的系统同时安装了 Python 2 和 Python 3，可能需要使用 `python3` 和 `pip3` 代替 `python` 和 `pip`。

### 第二步：安装依赖

本项目只需要两个第三方包：

| 包名 | 用途 | 版本要求 |
|------|------|---------|
| `torch` | 深度学习框架，提供张量运算、自动求导、GPU 加速 | >= 2.0 |
| `transformers` | HuggingFace 工具库，仅用于加载/保存分词器 | >= 4.30 |

其余所有 import（`json`, `math`, `os`, `shutil`, `time`, `pathlib`, `dataclasses`）都是 Python 自带的标准库，不需要额外安装。

**安装命令：**

```bash
pip install torch transformers
```

**如果需要 GPU（CUDA）加速**，请到 [PyTorch 官网](https://pytorch.org/get-started/locally/) 选择你的 CUDA 版本获取安装命令。例如 CUDA 12.1：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers
```

**我们建议你使用Flashattn2（在支持的设备上），以加速训练，此为预编译包**

```bash
https://github.com/mjun0812/flash-attention-prebuild-wheels/releases
```

**验证安装：**

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}，CUDA 可用：{torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers {transformers.__version__}')"
```

期望输出类似：

```
PyTorch 2.6.0+cu121，CUDA 可用：True
Transformers 4.48.0
```

### 第三步：分词器扩充（不重要，我已经扩充到53376，如果你有需要在修改这个脚本去扩充，尽量为128的倍数）

分词器（Tokenizer）负责将文本拆分为 token（模型能理解的数字序列）。本项目在上游分词器基础上扩充了 LaTeX 公式、科技英语词汇、化学符号，最终词表大小为 53376。

运行方式（非必要不用动）：

```bash
cd new-model-Solmen
python expand_tokenizer.py
```

### 第四步：准备训练数据

训练数据必须是 `.jsonl` 格式（JSON Lines），**每行一个 JSON 对象**，包含 `"text"` 字段：

**正确格式：**

```jsonl
{"text": "Transformer 是一种基于自注意力机制的神经网络架构，由 Vaswani 等人在 2017 年提出。"}
{"text": "The quadratic formula is x = (-b ± sqrt(b² - 4ac)) / (2a)."}
{"text": "水的化学式是 H₂O，由两个氢原子和一个氧原子组成。"}
```

**错误格式：**

```json
// 错误 1：整个文件是一个 JSON 数组（不是 .jsonl）
[
  {"text": "..."},
  {"text": "..."}
]

// 错误 2：缺少 "text" 字段
{"content": "..."}
{"message": "..."}
```

**数据量建议：**
Tips：仅供参考，实际上“T=N*20”是最好的，多了就开始边际效应递减了，不过仍有用，数据集质量越高需要的数据越少，数据集较多噪声可能就需要大于N*20，建议使用CCI（中文为主）、HuggingFace-EDU（英文为主），较为清洁
（N是参数量的意思，T是Token）

HuggingFaceFW/fineweb-edu：https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

CCI-3-HQ：https://www.modelscope.cn/datasets/BAAI/CCI3-HQ （十分推荐，我的很多模型就是用这个数据集，效果很好，不过需要注意Latex的文本支持度不高）

Wanjuan：https://github.com/opendatalab/WanJuan1.0

（这三个我只用过前两个，我们后续会开源Qwen3.5-35B-A3B蒸馏数据集，用于提升模型世界知识
（仅供参考，因为我也不知道你的数据集密度怎么样，可以使用token-count.py（新增）快速确认，建议为N*20个token，快速迭代并且效果较好

> 数据可以是多个文件。把所有文件的路径都填到第五步的 `DATA_FILES` 列表里即可，程序会按顺序读取。

### 第五步：修改配置（两处）

**第 1 处：选择模型规模**

打开 `config.py`，找到第 12 行：

```python
PRESET = "200M"  # ← 改这里
```

将 `"200M"` 改为你想要的规模。可选值：`"50M"` / `"80M"` / `"150M"` / `"200M"` / `"300M"` / `"500M"` / `"1.7B"`。

> **确认一下**：只改这一个变量就够了。所有预设共享同一个 `vocab_size=53376`，`PRESET` 会同时控制模型结构（层数、维度等）和训练参数（学习率、batch size 等）。

**第 2 处：填入数据路径**

打开 `pretrain.py`，找到第 21-24 行：

```python
DATA_FILES = [
    "/path/to/your/pretrain_data_1.jsonl",  # ← 替换为你的实际路径
    "/path/to/your/pretrain_data_2.jsonl",  # ← 替换为你的实际路径
]
```

把 `"/path/to/your/..."` 替换为你实际的数据文件路径。支持绝对路径和相对路径：

```python
# Windows 示例（注意用正斜杠 / 或双反斜杠 \\）
DATA_FILES = [
    "D:/data/wiki/train_part1.jsonl",
    "D:/data/wiki/train_part2.jsonl",
]

# Linux / macOS 示例
DATA_FILES = [
    "/home/user/data/pretrain_data.jsonl",
]

# 只有一个文件也没问题
DATA_FILES = [
    "data/my_corpus.jsonl",
]
```

### 第六步：开始训练

```bash
python pretrain.py
```

训练开始后，终端会显示类似：

```
[2026-04-18 10:00:00] 预设规模：200M，设备：cuda，精度：bf16
[2026-04-18 10:00:00] 分词器词表大小：53376
[2026-04-18 10:00:01] 模型参数量：198.2M
[2026-04-18 10:00:01] 未发现 checkpoint，从头开始训练
[2026-04-18 10:00:01] 开始训练...
[2026-04-18 10:05:23] step 50/100000 | loss 10.1234 | lr 7.50e-06 | grad_norm 1.234 | 数据进度：文件[0] 第1200行
[2026-04-18 10:10:45] step 100/100000 | loss 9.8765 | lr 1.50e-05 | grad_norm 0.987 | 数据进度：文件[0] 第2400行
```

训练过程中会自动：

- **每 50 步**（`log_steps`）打印一次训练日志
- **每 1000 步**（`save_steps`）保存一次 checkpoint 到 `checkpoints/checkpoint-{步数}/`
- **自动删除旧 checkpoint**，只保留最近 3 个

**训练产物：**

```
checkpoints/
├── checkpoint-1000/
│   ├── config.json              # 模型配置（HF 格式）
│   ├── pytorch_model.bin        # 模型权重
│   ├── tokenizer.json           # 分词器
│   ├── tokenizer_config.json    # 分词器配置
│   └── training_state.pt        # 训练状态（优化器 + 数据进度 + GradScaler）
├── checkpoint-2000/
│   └── ...
├── train.log                    # 完整训练日志
└── ...
```

### 第七步：断点续训

训练意外中断（停电、显存溢出、手动 Ctrl+C）后，**直接重新运行 `python pretrain.py` 即可自动续训**。

程序会自动检测 `checkpoints/` 下最新的 checkpoint，恢复：

- 模型权重
- 优化器状态（AdamW 的动量）
- 训练进度（当前步数、读到哪个数据文件的哪一行）
- GradScaler 状态（fp16 训练时）

日志会显示：

```
[2026-04-18 15:00:00] 检测到 checkpoint，恢复训练：checkpoints/checkpoint-2000
[2026-04-18 15:00:01] 续训起点：step=2000，数据文件[0] 第48000行
```

### 第八步：模型推理

训练完成后，可以用以下代码加载模型进行文本生成：

```python
from config import get_config
from model import SolmenModel
from transformers import PreTrainedTokenizerFast
import torch

# 1. 加载配置和模型（必须与训练时使用相同的 PRESET）
m_cfg, _ = get_config("200M")
model = SolmenModel(m_cfg)

# 2. 加载训练好的权重
model.load_state_dict(
    torch.load("checkpoints/checkpoint-10000/pytorch_model.bin", map_location="cpu")
)

# 3. 加载分词器
tokenizer = PreTrainedTokenizerFast.from_pretrained("tokenizer/")

# 4. 准备输入文本
input_ids = tokenizer.encode("机器学习是", return_tensors="pt")

# 5. 生成文本
output = model.generate(
    input_ids,
    max_new_tokens=100,    # 最多生成 100 个新 token
    temperature=0.8,       # 温度：越低越确定（推荐 0.7~1.0）
    top_p=0.9,             # 核采样：保留累积概率前 90% 的 token
)
print(tokenizer.decode(output[0]))
```

> **注意**：推理时的 `get_config("200M")` 必须与训练时的 `PRESET` 一致，否则模型结构不匹配会报错。

---

## 常见问题

### Q：改 PRESET 会影响分词器吗？

不会。所有 7 个预设共享同一个 `vocab_size=53376`，分词器不需要重新生成。

### Q：能不能用 CPU 训练？

可以，但极慢。200M 模型在 CPU 上训练一步可能需要几十秒到几分钟。推荐至少使用 8GB 显存的 GPU。

### Q：Windows 上有什么特别注意的？

- 路径使用正斜杠 `/` 或双反斜杠 `\\`：`"D:/data/train.jsonl"` 或 `"D:\\data\\train.jsonl"`
- `DataLoader` 的 `num_workers` 已固定为 0，不需要手动改
- 如果遇到 `UnicodeDecodeError`，检查数据文件是否为 UTF-8 编码

### Q：怎么判断训练效果好不好？

关注 `train.log` 中的 loss 趋势：

- **正常**：loss 从约 `log(vocab_size) ≈ 10.8`（随机水平）逐渐下降
- **异常**：loss 突然变 NaN → 可能是学习率太大或数据有问题
- **过拟合**：loss 下降后又上升 → 数据量不够，减小模型或增加数据

### Q：参数量显示不一致？

`config.py` 的 `count_parameters()` 显示的参数量比训练日志中的大约 55M（200M 规模），这是因为前者把 embedding 和 lm_head 各算了一份，而实际训练中它们共享权重。这是正常现象，不是 bug。

### Q：如何用训练好的模型对接 HuggingFace？

checkpoint 目录（`checkpoints/checkpoint-{步数}/`）已经是 HuggingFace 格式，包含 `config.json` + `pytorch_model.bin` + tokenizer 文件。你可以用标准的 HF 工具加载：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("checkpoints/checkpoint-10000")
tokenizer = AutoTokenizer.from_pretrained("checkpoints/checkpoint-10000")
```

> 注意：这需要你在 HuggingFace 上注册 `SolmenModel` 的自定义代码，或使用 `trust_remote_code=True`。

---
