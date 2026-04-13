---
type: project-note
project: MicroLM
section: core-code-analysis
priority: P2
file: scripts/01~06.py 数据 pipeline（六步处理）
---

# 数据 Pipeline — 六步处理

> 不太像"手撕算法题"，但适合项目面试中的工程追问。不需要每行都看懂，但要把每步在做什么讲顺。最值得抓的是背后的方法——尤其 `04_derive_tasks.py` 的四类任务派生设计和 `schema_repair` 的扰动构造。

---

## Pipeline 全景

```
InstructIE 原始数据 (171,471 条)
  │
  ▼ [Step 1] 01_normalize.py — 字段标准化 (171,471)
     text→input, relation 对齐, cate 归一化, 新增 source 字段
  │
  ▼ [Step 2] 02_filter.py — 两层过滤 (163,629)
     ├─ 硬过滤（剔除 3,585 条）: 空关系/跨集泄漏/超长等
     └─ 软过滤（per-topic P99 分位数, 剔除 4,257 条）
  │
  ▼ [Step 3] 03_quality_tier.py — 质量三档分层 (156,275 条 high)
     high: 匹配率100% + 理想区间 (95.5%)
  │
  ▼ [Step 4] 04_derive_tasks.py — 四类任务派生 (623,650 条) ★
     从每个原始样本派生 4 个 SFT 训练样本
  │
  ▼ [Step 5] 05_stratified_sample.py — 分层采样 (30,000 条)
     按 task_type(50/25/15/10) + topic(12均衡) + quality(high) + 复杂度
  │
  ▼ [Step 6] 06_to_chat_jsonl.py — 格式转写 + valid 切分
     统一为 instruction + schema + input → JSON output 三段式 chat-style JSONL
  │
  ▼ 最终产出
  data/sft_candidate/train.jsonl (28,500 条)
  data/sft_candidate/valid.jsonl (1,500 条)
```

**阈值集中配置：** 所有阈值在 `conf.py` 中统一管理，每步产出独立的 JSON 统计报告。

---

## Step 1：字段标准化（01_normalize.py）

### 源码核心：normalize_item()

```python
def normalize_item(item, source):
    # 统一输入字段名
    text = item.get("text") or item.get("input", "")   # train 用 text，valid/test 用 input

    # 统一 cate 命名
    cate = item.get("cate", "")
    cate = CATE_MAP.get(cate, cate)                     # "建筑结构" → "建筑"

    # 统一 relation 结构
    relations = []
    for r in raw_relations:
        rel = {
            "head": r.get("head", "").strip(),
            "relation": r.get("relation", "").strip(),
            "tail": r.get("tail", "").strip(),
        }
        relations.append(rel)

    result = {
        "id": str(item.get("id", "")),
        "cate": cate,
        "input": text,              # ← 统一为 "input"
        "relation": relations,
        "source": source,           # ← 新增来源标记
    }
    return result
```

**解决的数据集问题：**

| 问题 | 详情 | 处理方式 |
|------|------|----------|
| 字段不统一 | train 用 `text`，valid/test 用 `input` | `text or input` 合并 |
| relation 结构差异 | train 含 `head_type`/`tail_type`，valid/test 不含 | 保留为可选字段 |
| cate 命名漂移 | "建筑结构" vs "建筑" | `CATE_MAP` 归一化映射 |

**验证断言（L115-122）：** 标准化后用 assert 校验所有样本的字段一致性和 cate 命名正确性。这是"拿到新数据集第一件事先 profiling"的体现。

---

## Step 2：两层过滤（02_filter.py）

| 过滤类型 | 规则 | 剔除数量 |
|----------|------|----------|
| **硬过滤** | 空关系 / 跨集泄漏 / 关系数>25 / 输入过长>800 / head或tail过长>100 / 输入过短<15 | **3,585** |
| **软过滤** | per-topic P99 分位数：输入长度超 P99 / head/tail 长度超 P99 / 关系数超 P99 / 输出 JSON 长度超 P99 | **4,257** |

**硬过滤 vs 软过滤的设计逻辑：**
- 硬过滤：明确坏样本，无条件剔除（空关系、数据泄漏、极端异常值）
- 软过滤：按每个 topic 的 P99 分位数控制极端值，保留大部分数据的同时裁剪长尾

---

## Step 3：质量三分层（03_quality_tier.py）

```python
# 分层标准（示意）:
# high:  head/tail 匹配率 100% + 关系数和长度在理想区间
# medium: 匹配率较高但某些指标偏离
# low:    存在明显质量问题
```

最终采样只取 **high 质量**（占比 95.5%）。对于 30K 规模的数据集，优先保证质量比追求数量更有价值。

---

## ★ Step 4：四类任务派生（04_derive_tasks.py）— 最有设计价值的步骤

### 源码核心：四类任务构造函数

#### 任务 A：ie_extraction（50%）— 标准信息抽取

```python
def make_ie_extraction(item, schema_fields):
    schema_str = json.dumps(schema_fields, ensure_ascii=False)
    prompt = (
        "你是一个信息抽取助手。请根据给定的 schema 从文本中抽取信息，并以 JSON 格式输出。\n"
        f"Schema: {schema_str}\n"
        f"文本: {item['input']}"
    )
    output = relations_to_output_json(item["relation"])   # 正确的 gold output
    return {"task_type": "ie_extraction", "prompt": prompt, "output": output, ...}
```

核心主轴：给定 schema + 文本 → 抽取实体/属性/事件 → JSON 输出。

#### 任务 B：text_to_json（25%）— 结构化格式转换

```python
def make_text_to_json(item, schema_fields):
    prompt = (
        "请将以下文本中的信息按照指定 schema 转换为结构化 JSON 对象。\n"
        f"Schema 字段: {schema_str}\n"
        f"要求: 输出合法 JSON，字段名为 schema 中定义的关系类型...按实体分组。\n"
        f"文本: {item['input']}"
    )
    return {"task_type": "text_to_json", ...}
```

更强调输出格式的稳定性训练。prompt 中明确要求"合法 JSON""按实体分组"。

#### 任务 C：format_following（15%）— 零容忍格式约束

```python
def make_format_following(item, schema_fields):
    constraints = [
        "只输出 JSON，不要附加任何解释文字。",
        "只输出 JSON 格式的结果，不要包含任何额外说明。",
        "严格按照 JSON 格式输出，不要在 JSON 前后添加任何文字。",
        "仅输出结构化 JSON 数据，禁止附加解释、标注或格式化标记。",
    ]
    constraint = random.choice(constraints)            # 随机选一种约束表达

    prompt = f"{constraint}\nSchema: {schema_str}\n从文本中抽取信息并输出 JSON: {item['input']}"
    return {"task_type": "format_following", ...}
```

行为约束训练：随机选择不同的"零解释"约束表述，让模型学会"只输出 JSON 不废话"。这是 LoRA 擅长的"格式塑形"能力的关键训练信号。

#### 任务 D：schema_repair（10%）— 可控扰动纠错 ★

```python
def perturb_output(output_str, schema_fields, relations):
    """对正确的 JSON 输出做可控扰动"""
    output_obj = json.loads(output_str)

    # 扰动1: 字段名拼写错误（替换一个字符）
    if used_in_schema:
        target_field = random.choice(list(used_in_schema))
        field_chars = list(target_field)
        idx = random.randint(0, len(field_chars) - 1)
        field_chars[idx] = chr(ord(field_chars[idx]) + random.choice([1, -1, 2]))
        wrong_field = "".join(field_chars)
        perturbed[entity][wrong_field] = perturbed[entity].pop(target_field)
        # → 返回 ("字段名 'location' 被错误写成了 'locatuon'", perturbed_json)

    # 扰动2: 缺失一个字段
    # → 删除某个存在的字段

    # 扰动3: 添加幻觉字段（schema 中不存在的关系）
    # → 在某 entity 下添加 fake_field

    # 扰动4: 类型错误（字符串变列表）
    # → 将字符串值包装为 ["原值", "多余值"]

def make_schema_repair(item, schema_fields):
    output_str = relations_to_output_json(item["relation"])
    perturbed_output, perturbation_desc = perturb_output(output_str, schema_fields, ...)
    prompt = (
        "以下信息抽取结果存在错误，请根据 schema 和原文找出并修正错误。\n"
        f"Schema: {schema_str}\n原文: {item['input']}\n"
        f"有错误的抽取结果: {perturbed_output}\n错误类型: {perturbation_desc}\n"
        f"请输出修正后的正确 JSON。"
    )
    return {"task_type": "schema_repair", "output": output_str,  # 正确答案
            "perturbation": perturbation_desc, ...}
```

**这是整个 pipeline 最有创意的设计。** 原始 InstructIE 数据集中不存在"纠错"任务——纯靠数据增强引入。四种扰动类型覆盖了结构化输出中最常见的错误模式：

| 扰动类型 | 触发条件 | 训练模型的能力 |
|----------|----------|---------------|
| 字段名拼写错误 | 有使用中的字段 | 字段名精确匹配 |
| 缺失字段 | 有 ≥2 个使用中的字段 | 完整性检查 |
| 幻觉字段 | 存在未使用的 schema 字段 | 不编造不存在的信息 |
| 类型错误 | 有字符串值的字段 | 值类型遵循 |

**质量分层与派生的配合：**

| 原始质量 | 派生任务 |
|----------|----------|
| high | 全部 4 类 |
| medium | ie_extraction + text_to_json |
| low | 仅 ie_extraction |

这样自然形成配比框架，后续 Step 5 再精确控制到 50/25/15/10。

---

## Step 5：分层采样（05_stratified_sample.py）

### 源码核心：三维分层采样

```python
def classify_complexity(item):
    """按关系数和输入长度分复杂度桶"""
    n_rel = item["n_relations"]
    input_len = item["input_len"]
    if n_rel <= 3 and input_len < 100:
        return "simple"
    elif n_rel <= 6 and input_len < 250:
        return "medium"
    else:
        return "complex"

# 采样优先级：
# 1. 按 (quality_tier, topic, complexity) 三维分桶
# 2. quality: high > medium > low
# 3. complexity: medium > simple > complex（中等复杂度最能代表典型场景）
# 4. 每个 topic 等量分配（各 ~2,500 条）
```

**采样结果（精确匹配目标）：**

| 维度 | 目标 | 实际 |
|------|------|------|
| 总规模 | 30,000 | **30,000** |
| ie_extraction | 50% | **50.0%** (15,000) |
| text_to_json | 25% | **25.0%** (7,500) |
| format_following | 15% | **15.0%** (4,500) |
| schema_repair | 10% | **10.0%** (3,000) |
| 12 topic 均衡 | 各 ~2,500 | 各 **2,500**（精确均衡）|
| high 质量 | 优先 | **100%** |

---

## Step 6：格式转写（06_to_chat_jsonl.py）

将派生任务的 `(instruction, schema, input, output)` 四元组统一转为 HF-style 的 chat messages 格式：

```json
{
  "messages": [
    {"role": "system", "content": "你是一个严格遵循 schema 的信息抽取助手"},
    {"role": "user", "content": "Schema: [...]\nInput: ..."},
    {"role": "assistant", "content": "{...JSON output...}"}
  ]
}
```

从 train 中独立切分 5%（1,500 条）作为 valid 集。全量 JSON 合法性校验 **100% 通过**。

---

## 配比设计逻辑总结

| 任务 | 占比 | 为什么是这个比例 |
|------|------|-----------------|
| ie_extraction | 50% | 核心主轴，InstructIE 最大价值所在 |
| text_to_json | 25% | 输出格式稳定性训练，25% 足够 |
| format_following | 15% | 行为约束不需要大量样本 |
| schema_repair | 10% | 增强型任务，少量即可 |

**为什么 topic 要强制均衡：** 前 7 个 topic 占约 85%，自然科学(2.5%)和医学(1.9%)明显偏少。不均衡会导致模型过度拟合高频 topic 的模式。

---

## 面试高频追问清单

| 问题 | 回答要点 |
|------|----------|
| 六步 pipeline 每步做什么？ | 标准化 → 过滤(硬+软) → 质量分层 → 任务派生 → 分层采样 → 格式转写 |
| 为什么需要标准化？ | InstructIE 的 train/valid/test 字段命名不统一（text vs input, cate 漂移）|
| 硬过滤和软过滤的区别？ | 硬过滤剔除明确坏样本；软过滤按 per-topic P99 控制长尾 |
| 四类任务怎么设计的？ | ie_extraction(核心) / text_to_json(格式) / format_following(约束) / schema_repair(纠错)|
| schema_repair 怎么构造的？ | 对正确的 gold output 做可控扰动（拼写错误/缺失字段/幻觉字段/类型错误）|
| 为什么 50/25/15/10？ | IE 是核心占半壁；格式辅助不需多；约束少量即够；纠错增强型少量 |
| 为什么要 topic 均衡？ | 高频 topic 占 85%，不均衡导致过拟合；强制均衡确保低频 topic 也有足够信号 |
| 最终数据集多大？ | 28.5K train + 1.5K valid，JSON 合法性 100% |

---

## 相关记录

- [[07-train_qwen_lora.py Qwen 迁移线核心]] — 消费本 pipeline 产出的 SFT 数据集
- [[03-Qwen 迁移与结构化输出主线]] — pipeline 在项目全景图中的位置
