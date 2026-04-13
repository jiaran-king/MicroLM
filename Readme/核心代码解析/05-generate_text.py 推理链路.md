---
type: project-note
project: MicroLM
section: core-code-analysis
priority: P1
file: scripts/generate_text.py (351行)
---

# generate_text.py — 推理链路

> 这个文件把模型从"checkpoint 文件"串成"能生成文本的系统"。面试中如果被问到"从 checkpoint 到输出文本的全流程"，decode 循环（prefill → temperature → top-p → sampling → EOS）就是最直接的答案载体。

---

## 完整调用链

```
CLI 参数解析
    │
    ▼ load_model_config()          — JSON config 或 CLI 覆盖
    │
    ▼ BPETokenizer.from_files()    — 加载 BPE tokenizer（vocab.json + merge.txt）
    │
    ▼ TransformerLM(...)           — 创建模型结构
    │
    ▼ load_state_dict()            — 加载 checkpoint 权重
    │   └─ normalize_state_dict_keys() — 处理 _orig_mod. 前缀（LoRA 兼容）
    │
    ▼ resolve_generation_prompt()  — 组装 prompt（纯文本 / 对话模式）
    │
    ▼ tokenizer.encode()           — prompt → token IDs
    │
    ▼ sample_greedy_or_temperature() — ★ 核心推理入口
    │   ├─ temperature=0 → greedy decode（无 KV Cache）
    │   └─ temperature>0 → model.generate()（KV Cache 加速）
    │
    ▼ tokenizer.decode()           — token IDs → 文本输出
```

---

## 逐段源码与解析

### 1. Prompt 组装（L312-316）

```python
generation_prompt = resolve_generation_prompt(
    prompt=args.prompt,
    conversations_json=args.conversations_json,
    conversations_path=args.conversations_path,
)
```

支持两种输入模式的统一入口：

| 模式 | 参数 | 适用场景 |
|------|------|----------|
| 纯文本续写 | `--prompt "..."` | 观察 pretrain 的语言建模能力 |
| 对话生成 | `--conversations-json '...'` | SFT/LoRA 的指令跟随能力 |

`resolve_generation_prompt()` 内部调用 `sft.py` 的 `build_generation_prompt()`，保证推理格式与训练格式严格一致。

### 2. 模型加载与权重恢复（L289-303）

```python
model = TransformerLM(
    vocab_size=int(config["vocab_size"]),
    context_length=int(config["context_length"]),
    d_model=int(config["d_model"]),
    num_layers=int(config["num_layers"]),
    num_heads=int(config["num_heads"]),
    d_ff=int(config["d_ff"]),
    rope_theta=float(config["rope_theta"]),
    device=device, dtype=dtype,
).to(device)

state_dict = load_state_dict(args.checkpoint_path, device)
model.load_state_dict(state_dict)
model.eval()
```

**normalize_state_dict_keys（L222-228）：**

```python
def normalize_state_dict_keys(state_dict):
    normalized = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod."):]     # 剥离 LoRA 前缀
        normalized[key] = value
    return normalized
```

处理 LoRA merge 后保存的 checkpoint：merge 后 `Linear` 变成 `LoRALinear`，其内部原始权重的 key 会带上 `_orig_mod.` 前缀。加载时需要剥离才能匹配原始模型结构。

### 3. ★ sample_greedy_or_temperature（L243-271）— 双模式推理

```python
def sample_greedy_or_temperature(model, prompt_ids, max_new_tokens,
                                eos_token_id, temperature, top_p):
    if temperature == 0.0:
        # ── Greedy 模式（无 KV Cache）──
        model.eval()
        generated = prompt_ids.clone()
        for _ in range(max_new_tokens):
            idx_cond = generated[:, -model.context_length:]   # 截断到上下文长度
            logits = model(idx_cond)[:, -1, :]                # 取最后位置
            if top_p < 1.0:
                logits = model._top_p_filter(logits, top_p)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)  # 取最大值
            generated = torch.cat((generated, next_token), dim=1)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        return generated

    # ── Sampling 模式（KV Cache 加速）──
    return model.generate(
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        temperature=temperature,
        top_p=top_p,
    )
```

**两种模式的对比：**

| 维度 | Greedy (temp=0) | Sampling (temp>0) |
|------|-----------------|-------------------|
| 采样方式 | argmax（确定性） | multinomial（随机性） |
| KV Cache | 不使用 | 使用 |
| 计算效率 | 低（每步重算完整序列） | 高（增量解码） |
| 输出特点 | 确定性、可能重复 | 多样性更好 |
| 适用场景 | 调试、格式要求严格的任务 | 创意生成、对话 |

### 4. EOS 处理（L306-310）

```python
eos_token_id = None
if args.eos_token is not None:
    eos_token_bytes = args.eos_token.encode("utf-8")
    if eos_token_bytes not in tokenizer.vocab_to_id:
        raise ValueError(f"EOS token {eos_token_id!r} is not in vocab")
    eos_token_id = tokenizer.vocab_to_id[eos_token_bytes]
```

EOS token 需要通过 UTF-8 编码后在词表中查找。如果不在词表中直接报错，避免运行时静默失败。

### 5. 输出处理（L334-346）

```python
full_ids = generated[0].tolist()
new_ids = full_ids[len(prompt_token_ids):]       # 只取新生成的部分
full_text = tokenizer.decode(full_ids)             # 全文（含 prompt）
new_text = tokenizer.decode(new_ids)               # 仅新生成部分

if args.show_token_ids:
    print(f"prompt_token_ids={prompt_token_ids}")
    print(f"generated_token_ids={new_ids}")

if args.print_new_text_only:
    print(new_text)                                # 只打印新文本
else:
    print(full_text)                               # 打印完整序列
```

---

## 推理全流程图

```
用户输入: "什么是深度学习？"
    │
    ▼ build_generation_prompt()
prompt_text: "system\n...\n\nuser\n什么是深度学习？\n\nassistant\n"
    │
    ▼ tokenizer.encode()
prompt_ids: [12, 45, 230, 67, 89, ..., 5]      (长度 N)
    │
    ▼ [Prefill] model.forward(prompt_ids, use_cache=True, start_pos=0)
logits[:, -1, :]                                  (shape: [1, vocab_size])
    │
    ▼ [Decode Loop] × max_new_tokens
    ① logits /= temperature                        (控制随机性)
    ② top_p filter                                 (截断低概率 tail)
    ③ softmax + multinomial → next_token           (采样)
    ④ EOS? → 终止                                 (可选停止条件)
    ⑤ forward([next_token], kv_cache, start_pos=cur_pos)
       logits[:, -1, :]                            (只取最后位置)
    │
    ▼
new_ids: [102, 45, 890, 33, ...]                 (新生成的 token 序列)
    │
    ▼ tokenizer.decode()
output: "深度学习是机器学习的一个子领域..."
```

---

## 面试高频追问清单

| 问题 | 回答要点 |
|------|----------|
| 从 checkpoint 到生成文本经过哪些步骤？ | 加载 config → 创建模型 → 加载权重 → 组装 prompt → encode → prefill → decode loop → decode → 输出 |
| prefill 和 decode 有什么区别？ | prefill 送完整 prompt，decode 每次只送 1 个新 token；prefill 缓存 K/V，decode 复用缓存 |
| temperature 怎么影响输出？ | >1 更多样（高温），<1 更确定（低温），=0 是 greedy |
| top-p nucleus sampling 怎么工作？ | 按概率降序排列，累积概率超过 p 的截断，保留最小候选集使概率和 ≥ p |
| 为什么推理时要在末尾补 assistant 标记？ | 与训练时 render_chat_prompt 格式一致，让模型从正确的上下文开始生成 |
| greedy 和 sampling 模式选哪个？ | 调试用 greedy（确定性好），对话/创意用 sampling（多样性好）|

---

## 相关记录

- [[01-transformer.py 模型主干]] — 本文件调用的 model.generate() 和 _top_p_filter()
- [[03-sft.py SFT 数据协议]] — 本文件调用的 build_generation_prompt()
- [[06-chat.py 多轮对话系统]] — 本文件的单轮版本扩展为多轮交互
