---
type: project-note
project: MicroLM
section: core-code-analysis
priority: P2
file: scripts/chat.py (604行)
---

# chat.py — 多轮对话系统

> 从"单次生成脚本"升级为可交互的多轮对话系统。工程价值很高——它解决的问题不是"模型能不能生成文本"，而是"用户能不能直观感受和对比不同模型版本"。这里修复的 3 个关键 bug 都集中在累积型状态管理上，面试中讲清楚这些 bug 会很加分。

---

## 系统架构

```
CLI (argparse)
    → load_chat_model()     加载 tokenizer + TransformerLM + state_dict [+ LoRA merge]
    → ChatSession
        │
        ├─ conversations 列表（多轮历史）
        ├─ _render_prompt()   ROLE_MARKERS 格式化 + EOS 处理
        ├─ chat()             encode → clip → generate → decode → sanitize → 存储
        │
        └─ repl()             输入分发 / 命令处理 / 计时 / 日志 / 异常捕获
```

**REPL 命令：**

| 命令 | 功能 |
|------|------|
| `/temp <value>` | 运行时切换 temperature |
| `/topp <value>` | 运行时切换 top-p |
| `/system <prompt>` | 设置或更换 system prompt |
| `/clear` | 清空对话历史，重新开始 |
| `/history` | 查看当前对话历史（轮次统计）|
| `/save [path]` | 保存会话日志到 JSONL 文件 |
| `/quit` | 退出 |

---

## 逐段源码与解析

### 1. 模型加载（L398-444）

```python
def load_chat_model(checkpoint_path, config_path, vocab_path, merges_path,
                    eos_token, lora_path=None, dtype="float32", device="auto"):
    device = resolve_device(device)
    torch_dtype = resolve_model_dtype(dtype, device)

    tokenizer = BPETokenizer.from_files(str(vocab_path), str(merges_path),
                                       special_tokens=[eos_token])
    config = load_model_config(config_path, vocab_size=len(tokenizer.id_to_vocab))

    model = TransformerLM(...).to(device)
    state_dict = load_state_dict(checkpoint_path, device)
    model.load_state_dict(state_dict)

    # LoRA 分支：注入 → 加载 adaptor → merge → 推理时无额外开销
    if lora_path is not None:
        apply_lora_to_model(model)
        lora_state = torch.load(lora_path, map_location=device, weights_only=False)
        load_lora_state_dict(model, lora_state)
        merge_lora(model)          # ← 推理前合并，对上层透明

    model.eval()
    return model, tokenizer, config
```

三种模型状态共享同一套加载逻辑：

| 加载方式 | 行为 |
|----------|------|
| Pretrain checkpoint | 基础续写模式 |
| SFT baseline checkpoint | 全参微调后的对话助手 |
| SFT LoRA checkpoint + adaptor | 先 merge_lora 再推理 |

LoRA 加载对上层完全透明——调用方不需要知道内部是否做了 merge。

### 2. ★ _render_prompt（L212-233）— 必须匹配训练格式

```python
def _render_prompt(self, convs):
    parts = []
    for message in convs:
        role = message["role"]
        content = message["content"]
        parts.append(ROLE_MARKERS[role])
        parts.append(content)
        parts.append("\n")
        if role == "assistant" and self.eos_token_id is not None:
            parts.append(self.eos_token)      # assistant 后追加 EOS
            parts.append("\n")
    # 末尾补 assistant 标记触发生成
    parts.append(ROLE_MARKERS["assistant"])
    return "".join(parts)
```

**关键约束：必须与 sft.py 的 render_chat_prompt 格式一致。** 这里是 Bug 2 的修复位置——原始版本缺少 EOS 标记。

**EOS 条件判断 `self.eos_token_id is not None`：** 这是 Bug 1 的防御性修复。tokenizer 词表 (6401) 可能大于模型 embedding (6400)，此时 EOS token ID 越界，不能安全使用。

### 3. 多轮对话核心流程 chat()（L235-309）

```python
def chat(self, user_input):
    self.conversations.append({"role": "user", "content": user_input})   # ① 追加用户输入

    convs = self._build_prompt_conversations()
    prompt_text = self._render_prompt(convs)                              # ② 渲染 prompt
    prompt_ids = self.tokenizer.encode(prompt_text)

    # ③ 安全裁剪：过滤越界的 token ID（EOS 可能越界）
    model_vocab_size = self.model.token_embeddings.weight.shape[0]
    prompt_ids = [tid for tid in prompt_ids if tid < model_vocab_size]

    # ④ 长度检查与截断
    budget = self.context_length - self.max_new_tokens - 16              # 16 token 余量
    if len(prompt_ids) > budget:
        convs = self._truncate_conversations(prompt_ids)                  # 裁剪最早的历史
        prompt_text = self._render_prompt(convs)
        prompt_ids = self.tokenizer.encode(prompt_text)

    # ⑤ 生成
    with torch.no_grad():
        output = self.model.generate(
            prompt_ids=prompt_tensor,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.eos_token_id,
            temperature=self.temperature,
            top_p=self.top_p,
        )

    new_ids = full_ids[len(prompt_ids):]
    reply = self.tokenizer.decode(new_ids).strip()

    # ⑥ 清理 surrogate 字符（Bug 3 修复）
    reply = _remove_surrogates(reply)

    # ⑦ 清除尾部 EOS 残留
    if self.eos_token and reply.endswith(self.eos_token):
        reply = reply[: -len(self.eos_token)].strip()

    self.conversations.append({"role": "assistant", "content": reply})     # ⑧ 存入历史
    return reply
```

**上下文窗口自动裁剪（`_truncate_conversations`，L179-210）：**

```python
def _truncate_conversations(self, prompt_ids):
    budget = self.context_length - self.max_new_tokens - 16
    if len(prompt_ids) <= budget:
        return self._build_prompt_conversations()

    convs = self._build_prompt_conversations()
    has_system = convs and convs[0]["role"] == "system"
    system_part = [convs[0]] if has_system else []
    dialogue = convs[1:] if has_system else convs

    while dialogue and len(prompt_ids) > budget:
        dialogue.pop(0)                          # 从最早的轮次开始删除
        # 重新编码检查长度
        trial = system_part + dialogue
        if dialogue:
            rendered = self._render_prompt(trial)
            prompt_ids = self.tokenizer.encode(rendered)
        else:
            break

    return system_part + dialogue
```

保留 system prompt，从最早的对话轮次开始裁剪。没有这个机制，多轮对话在第 4-5 轮后会因超出 context_length 而 crash 或退化。

### 4. Bug 修复记录

#### Bug 1 — EOS 标记缺失（已修复于 L227-230）

**现象：** SFT checkpoint 生成质量明显低于预期，尤其是第一条回复经常跑题。

**根因：** `_render_prompt()` 在 assistant 消息后缺少 EOS 标记。训练数据中每条 assistant 回复都以 EOS 结尾，推理时如果漏掉这个标记，模型的分布起点就和训练时不一致。

**修复：** 在 assistant 渲染后统一追加 EOS + 换行。并增加条件判断：仅当 `eos_token_id` 有效（< vocab_size）时才追加。

> [!warning] 这类 bug 最危险
> 不会报错，只会静默降低生成质量。loss mask 保证梯度只流向 assistant 区间，但 **assistant 区间的边界定义** 必须训练和推理完全一致。

#### Bug 2 — Token ID 越界（已修复于 L247-248）

**现象：** 多轮对话第二轮崩溃 `IndexError: index out of range`。

**根因：** tokenizer 注册 special token 后词表变为 6401，EOS 的 ID=6400。但模型 embedding 层的 vocab_size 仍是 6400。

**修复：** 两层防御：
1. `ChatSession.__init__` 中检查 `eid < model_vocab_size`
2. `chat()` 中编码后裁剪所有 ≥ vocab_size 的 token ID

#### Bug 3 — Unicode Surrogate 字符（L42-52）

**现象：** 多轮对话第二轮 `UnicodeEncodeError`。

**根因：** 小模型生成的某些 token 序列解码后产生 Unicode surrogate 字符（U+D800–U+DFFF）。存入对话历史后下一轮 `.encode("utf-8")` 时崩溃。

**修复：** `_remove_surrogates()` 每轮回复存入历史前自动清理。

```python
def _remove_surrogates(text: str) -> str:
    import regex as re
    return re.sub(r"[\ud800-\udfff]", "", text)
```

> [!tip] 累积型 bug 的特征
> 这三个 bug 有共同模式：**单次运行正常，问题只在多轮/累积后才暴露**。对任何涉及"历史""缓存""累积"的代码都要特别警惕。

---

## 面试高频追问清单

| 问题 | 回答要点 |
|------|----------|
| 怎么把单轮推理扩成多轮？ | conversations 列表维护历史；每次渲染完整历史为 prompt；超长则裁剪最早轮次 |
| 上下文怎么维护？ | conversations 列表追加用户输入和模型回复；_render_prompt 统一渲染 |
| 超长历史怎么裁剪？ | 保留 system prompt，从最早轮次开始逐个删除，重新编码检查长度 |
| LoRA 模型怎么加载？ | apply_lora → load_state_dict → merge_lora，三步完成，对上层透明 |
| 为什么第二轮会崩？ | 三个可能原因：EOS 缺失导致格式偏移、token ID 越界、surrogate 字符编码失败 |
| /temp 和 /topp 为什么不需要重启会话？ | 直接修改 ChatSession 实例属性，下一次生成立即生效 |

---

## 相关记录

- [[01-transformer.py 模型主干]] — 本文件调用的 model.generate()
- [[02-lora.py LoRA 参数高效微调]] — 本文件调用的 apply_lora/merge_lora
- [[03-sft.py SFT 数据协议]] — 本文件的 _render_prompt 必须与其格式一致
- [[05-generate_text.py 推理链路]] — 本文件的单轮版本
