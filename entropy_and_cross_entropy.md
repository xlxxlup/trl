# 信息熵与交叉熵

## 信息熵 (Information Entropy)

### 定义
信息熵用于衡量一个概率分布的不确定性或随机性。熵值越大，分布越均匀、不确定；熵值越小，分布越集中、确定。

### 数学公式
对于离散概率分布 P = (p₁, p₂, ..., pₙ)，信息熵定义为：

```
H(P) = -∑ pᵢ log(pᵢ)
```

其中：
- `pᵢ` 是第 i 个类别的概率
- `log` 通常以自然对数(ln)或以2为底
- 熵的单位取决于对数底（nat 或 bit）

### 性质
- 熵总是非负的：H(P) ≥ 0
- 当分布均匀时熵最大（最不确定）
- 当分布集中于某个值时熵最小（最确定）

### 代码实现
```python
import torch
import torch.nn.functional as F

def entropy(logits):
    """
    计算信息熵

    Args:
        logits: 形状为 (..., num_classes) 的张量

    Returns:
        熵值，形状为 (...)
    """
    logps = F.log_softmax(logits, dim=-1)
    probs = torch.exp(logps)
    ent = -(probs * logps).sum(-1)
    return ent
```

### TRL 中的实现
在 [trl/trainer/utils.py:597-604](trl/trainer/utils.py#L597-L604)：

```python
flat_logits = logits.reshape(-1, num_classes)

entropies = []
for chunk in flat_logits.split(chunk_size, dim=0):
    logps = F.log_softmax(chunk, dim=-1)
    chunk_entropy = -(torch.exp(logps) * logps).sum(-1)
    entropies.append(chunk_entropy)
```

这里使用分块处理来节省内存，每步计算：
1. `log_softmax` 得到 log(pᵢ)
2. `torch.exp(logps)` 得到 pᵢ
3. `-(pᵢ * log(pᵢ)).sum()` 得到熵值

---

## 交叉熵 (Cross Entropy)

### 定义
交叉熵用于衡量两个概率分布之间的差异。在机器学习中，常用于计算损失函数，衡量预测分布与真实分布的距离。

### 数学公式
对于两个离散概率分布 P（真实分布）和 Q（预测分布）：

```
H(P, Q) = -∑ pᵢ log(qᵢ)
```

其中：
- `pᵢ` 是真实概率（通常为 one-hot 编码）
- `qᵢ` 是预测概率
- `log(qᵢ)` 是预测概率的对数

### 与信息熵的关系
交叉熵可以分解为：

```
H(P, Q) = H(P) + D_KL(P || Q)
```

其中：
- `H(P)` 是真实分布的信息熵（常数）
- `D_KL(P || Q)` 是 KL 散度，衡量两个分布的差异

因此，最小化交叉熵等价于最小化 KL 散度。

### 代码实现
```python
import torch
import torch.nn.functional as F

def cross_entropy(logits, targets):
    """
    计算交叉熵

    Args:
        logits: 形状为 (batch_size, num_classes) 的原始输出
        targets: 形状为 (batch_size,) 的目标类别索引

    Returns:
        标量损失值
    """
    # 方法1：使用内置函数
    loss = F.cross_entropy(logits, targets)

    # 方法2：手动实现
    # log_probs = F.log_softmax(logits, dim=-1)
    # loss = -log_probs[range(len(targets)), targets].mean()

    return loss
```

---

## 两者对比

| 特性 | 信息熵 | 交叉熵 |
|------|------------------------|-------------------------|
| **公式** | `-∑ pᵢ log(pᵢ)` | `-∑ pᵢ log(qᵢ)` |
| **涉及分布** | 单个分布 P | 两个分布 P 和 Q |
| **用途** | 衡量分布的不确定性 | 衡量两个分布的差异 |
| **应用场景** | 探索策略、多样性正则化 | 分类损失、强化学习策略优化 |

---

## 应用示例

### 1. 信息熵在强化学习中的应用
```python
# 策略网络的熵正则化，鼓励探索
policy_loss = -log_prob * advantage  # 策略梯度
entropy_bonus = entropy(logits) * entropy_coef  # 熵系数
total_loss = policy_loss - entropy_bonus
```

### 2. 交叉熵在监督学习中的应用
```python
# 分类任务的标准损失函数
loss = F.cross_entropy(predictions, targets)
```

### 3. 交叉熵在强化学习中的应用
```python
# PPO/TRPO 中的策略损失
# ratio = p_new / p_old
# surrogate = ratio * advantage
# 使用交叉熵近似 KL 散度约束
```

---

## 参考资源

- [TRL 熵计算实现](https://github.com/huggingface/trl/blob/main/trl/trainer/utils.py)
- PyTorch `F.cross_entropy` 文档
- Information Theory (Cover & Thomas)
