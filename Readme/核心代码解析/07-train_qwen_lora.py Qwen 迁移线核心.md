---
type: project-note
project: MicroLM
section: core-code-analysis
priority: P1
file: scripts/train_qwen_lora.py (376行)
---

# train_qwen_lora.py — Qwen 迁移线核心

> 这是 Qwen 迁移线的核心实现文件。相比自研链路，它更像"工业生态适配能力"的证明。三个重点：HF AutoTokenizer/AutoModel 接入、ChatML 模板组织输入、**prefix 对比法**做 assistant-only loss mask。prefix 对比法是面试亮点——不复杂但很能体现对训练目标的理解。

---

## 与自研链路的关键差异

| 维度 | 自研 (`train_sft.py`) | Qwen (`train_qwen_lora.py`) |
|------|-----------------------|---------------------------|
| 基座模型 | 自研 TransformerLM (31.7M) | Qwen2.5-1.5B-Instruct (1.55B) |
| Tokenizer | 自研 BPETokenizer (vocab=6400) | HF AutoTokenizer (ChatML 模板) |
| LoRA 框架 | 自研 `lora.py`（LoRALinear） | PEFT 库（`get_peft_model`）|
| Chat 模板 | 自定义 role markers | ChatML（`<\|im_start\|>` / `<\|im_end\|>`）|
| Loss mask | `build_loss_labels()` 标记定位 | **prefix 对比法**定位 assistant 区间 |
| 精度 | float32 | FP16 |
| System prompt | 可选注入 (ratio=0.1) | 固定："严格遵循 schema 的信息抽取助手" |

复用的设计模式：配置文件组织、JSONL 日志格式、训练循环结构、报告模板。

---

## 逐段源码与解析

### 1. Dataset 与 ★ Prefix 对比法 Loss Mask（L33-110）

```python
class InstructIEDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512,
                 system_prompt=None, seed=42):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt
        self.samples = []

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                messages = record["messages"]
                processed = self._process_sample(messages)
                if processed is not None:
                    self.samples.append(processed)

    def _process_sample(self, messages):
        # 构建完整消息列表（含 system prompt）
        full_messages = []
        if self.system_prompt:
            full_messages.append({"role": "system", "content": self.system_prompt})
        for msg in messages:
            full_messages.append(msg)

        # 用 ChatML 模板 tokenize 完整对话
        full_ids = self.tokenizer.apply_chat_template(
            full_messages, tokenize=True, truncation=True, max_length=self.max_length)
        if len(full_ids) < 2:
            return None

        # ★ Prefix 对比法：构建"无回答"版本作为 prefix
        prefix_messages = full_messages[:-1]          # 去掉最后一条（assistant 回复）
        prefix_ids = self.tokenizer.apply_chat_template(
            prefix_messages, tokenize=True, add_generation_prompt=True)

        # 验证 prefix 匹配
        if full_ids[:len(prefix_ids)] != prefix_ids:
            prefix_len = len(prefix_ids)               # fallback
        else:
            prefix_len = len(prefix_ids)

        # 构建 labels：prefix 部分 -100，assistant 部分保留真实 ID
        labels = [-100] * prefix_len + full_ids[prefix_len:]

        # 截断到 max_length
        input_ids = full_ids[:self.max_length]
        labels = labels[:self.max_length]
        return {"input_ids": input_ids, "labels": labels}
```

#### Prefix 对比法详解

这是与自研 `build_loss_labels()` 思路一致但实现方式不同的 loss mask 构造方法。

**ChatML 格式的输入示例：**

```
<|im_start|>system
你是一个严格遵循 schema 的信息抽取助手<|im_end|>
<|im_start|>user
请从以下文本中抽取实体... Schema: {...} Input: ...<|im_end|>
<|im_start|>
{"entity": "华为技术有限公司", "location": "深圳"}
<|im_end|>
```

**Prefix 对比法的步骤：**

| 步骤 | 操作 | 结果 |
|------|------|------|
| 1 | 渲染完整对话（含 assistant 回复）| `full_ids` = [A, B, C, **D, E, F**, G] |
| 2 | 渲染到 assistant 标记为止（无回复内容）| `prefix_ids` = [A, B, C, D] |
| 3 | 对比长度差 | prefix_len = 4 |
| 4 | labels 前 4 位 = -100，后 3 位 = [E, F, G] | `[-100, -100, -100, -100, E, F, G]` |

**为什么叫"对比法"：** 通过比较"有回答"和"无回答"两个版本的 token 序列长度差异，精确定位 assistant 输出的起止位置。

**与自研 build_loss_labels() 的对比：**

| 维度 | 自研（标记定位）| Qwen（prefix 对比法）|
|------|-----------------|---------------------|
| 定位方式 | 在 token 序列中搜索 `assistant` 标记字符串 | 对比两版渲染结果的长度差 |
| 依赖 | ROLE_MARKERS 字符串匹配 | HF apply_chat_template 的确定性输出 |
| 适用场景 | 自定义 chat 模板 | ChatML 等标准模板 |
| 多轮支持 | while 循环找所有区间 | 只处理最后一个 assistant 区间（SFT 数据通常单轮）|

两者思路完全一致（都是 assistant-only masked loss），只是定位方式不同。

### 2. LoRA 配置与接入（L213-224）

```python
lora_cfg = cfg["lora"]
peft_config = LoraConfig(
    r=lora_cfg["r"],                          # rank = 8
    lora_alpha=lora_cfg["alpha"],              # alpha = 16 → scaling = 2
    target_modules=lora_cfg["targets"],         # ["q_proj", "k_proj", "v_proj", "o_proj"]
    lora_dropout=lora_cfg.get("dropout", 0.05),
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
print_trainable_params(model)
```

**PEFT vs 自研 lora.py 的对应关系：**

| 操作 | 自研 lora.py | PEFT |
|------|-------------|------|
| 创建 LoRA 层 | `LoRALinear(original, r, alpha)` | `LoraConfig(...)` + `get_peft_model()` |
| 冻结原参 | `requires_grad_(False)` | PEFT 自动处理 |
| 注入目标 | `_DEFAULT_TARGETS` 集合 | `target_modules` 列表 |
| 获取可训练参数 | `get_lora_params(model)` | `model.parameters()` 过滤 requires_grad |
| 保存/加载 | `get/load_lora_state_dict()` | `model.save_pretrained()` / `PeftModel.from_pretrained()` |
| Merge | `merge_lora()` | `model.merge_and_unload()` |

**Qwen LoRA 参数效率：**

| 指标 | 数值 |
|------|------|
| 总参数量 | 1,546,860,544 (~1.55B) |
| 可训练参数 | 2,179,072 |
| 可训练占比 | **0.14%** |
| Adaptor 存储 | **8.3 MB**（vs 基座 2,944 MB）|
| 存储节省 | **99.7%** |

比 MicroLM 的 0.83% 更高效——因为基座更大（1.55B vs 31.7M），同样的 r=8 注入的绝对参数量更大，但占总参数比例更低。

### 3. 训练循环（L289-350）

```python
for step in range(max_steps):
    optimizer.zero_grad()
    accumulated_loss = 0.0

    for micro_step in range(grad_accum):           # 梯度累积
        try:
            input_ids, labels, attention_mask = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)        # epoch 自动循环
            input_ids, labels, attention_mask = next(train_iter)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / grad_accum            # 损失除以累积步数
        loss.backward()
        accumulated_loss += loss.item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # 梯度裁剪
    optimizer.step()
    scheduler.step()

    # 定期评估
    if completed_step % eval_interval == 0 or completed_step == max_steps:
        val_loss = evaluate(model, valid_loader, device)
        # ... 日志记录 ...
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(out_dir / "best_adaptor")   # 保存最优 adaptor

    # 定期保存 checkpoint
    if completed_step % save_interval == 0 or completed_step == max_steps:
        model.save_pretrained(out_dir / f"ckpt_step_{completed_step}")
```

**Smoke → 正式训练的节奏控制：**

| 阶段 | 步数 | lr | batch_size | grad_accum | 目的 |
|------|------|----|-----------|------------|------|
| Smoke test | 50 | 3e-5 | 2 | 4 (eff=8) | 验证全链路：loss 能否下降 |
| 正式训练 | 2000 | 2e-5 | 4 | 4 (eff=16) | 充分训练 ~6.5 小时 |

**Loss 曲线结果：**

| Step | train_loss | val_loss | 阶段特征 |
|------|-----------|----------|----------|
| 100 | 0.3722 | 0.3999 | 起点 |
| 500 | 0.2538 | 0.2028 | 快速下降 |
| 1000 | 0.1520 | 0.1753 | 稳定收敛 |
| 2000 | 0.1857 | **0.1534** | 趋缓但未完全收敛 |

val_loss 从 0.3999 降至 **0.1534**，降幅 **61.6%**。全程无震荡、无过拟合。

### 4. evaluate 函数（L145-176）

```python
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for input_ids, labels, attention_mask in loader:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            loss_mask = (shift_labels != -100).float()

            per_token_loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1), reduction="none",
            ).reshape(shift_labels.shape)

            n_tokens = loss_mask.sum().item()
            if n_tokens > 0:
                total_loss += (per_token_loss * loss_mask).sum().item()
                total_tokens += n_tokens

    model.train()
    return total_loss / total_tokens if total_tokens > 0 else float("nan")
```

注意这里用 PyTorch 内置的 `F.cross_entropy` 配合手动 mask（而非自研的 `masked_cross_entropy`）。shift 操作（logits[:, :-1], labels[:, 1:]）实现标准的 next-token prediction 对齐。

---

## 面试高频追问清单

| 问题 | 回答要点 |
|------|----------|
| 怎么用 HF 接入 Qwen？ | AutoModelForCausalLM.from_pretrained + AutoTokenizer.from_pretrained |
| ChatML 模板长什么样？ | `<\|im_start\|>role\ncontent<\|im_end\|>` |
| prefix 对比法怎么工作的？ | 渲染完整对话和"无回答"版本，对比 token 数量差异定位 assistant 区间 |
| 和自研的 build_loss_labels 有什么区别？ | 思路一致（都是 assistant-only loss），定位方式不同（标记搜索 vs 长度对比）|
| LoRA 配置参数怎么选的？ | r=8, alpha=16 (scaling=2), 目标层 q/k/v/o, dropout=0.05 |
| 为什么先 smoke 再正式？ | 50 步验证全链路完整性（数据加载、loss 下降、checkpoint 保存），避免跑 10 小时才发现问题 |
| FP16 安全吗？ | 1.55B 模型在 RTX 5070 Ti 上安全使用混合精度，加速训练且不影响收敛 |

---

## 相关记录

- [[01-transformer.py 模型主干]] — 自研模型的训练对照
- [[02-lora.py LoRA 参数高效微调]] — 自研 LoRA 实现（本文件的 PEFT 版本对应物）
- [[03-sft.py SFT 数据协议]] — 自研 SFT 数据协议（prefix 对比法的参照系）
