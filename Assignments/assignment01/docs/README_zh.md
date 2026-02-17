# CS6493：自然语言处理 - 作业 1（中文翻译）

## 说明
1. 截止时间：2026 年 3 月 11 日晚上 10 点；
2. 你可以提交以下任一形式：
   - 一个 PDF 文件（包含答案）+ 代码包；或
   - 一个 Jupyter Notebook（同时包含答案和代码）；
3. 对于编程题，除代码外，鼓励补充代码设计与工作流程说明；也建议对实验结果做详细分析；
4. 总分：100 分；
5. 如有问题，请在 Canvas-Discussion 论坛发帖，或联系助教 Mr. Junyi ZHOU（junyizhou8-c@my.cityu.edu.hk）。

---

## 问题 1（11 分）
在语言模型中，我们可以用 one-hot 编码而不是 ASCII 表示来表示词。
即用如下向量表示词 `w`：

`[0, 0, ..., 1, ..., 0, 0]`

其中 `V` 为词表，`|V|` 为词表大小。设词表 `V = {cat, dog, run, runs}`。

1.  表示构造（2 分）
- 按字典序排列单词，为 `V` 中每个词写出标准 one-hot 向量表示。

2.  内积与相似性（3 分）
- 计算任意两个不同词的 one-hot 向量点积；
- 该结果对 one-hot 编码下的语义相似性意味着什么？
- 为什么这对建模语言关系是个问题？

3.  参数效率（3 分）
- 假设这些 one-hot 向量输入到一个线性层，权重矩阵 `W ∈ R^(4×d)`，其中 `d = 128`；
- 表示整个 `V` 的 embedding table 需要多少参数？
- 若词表扩大到 `|V| = 50,000`，需要多少参数？
- 列出该扩展带来的两个实际问题。

4.  无法捕获形态信息（2 分）
- `run` 和 `runs` 存在明显形态关系（动词屈折）；
- 这种关系能否从 one-hot 表示中自动识别？为什么？
- 这对泛化到未见过的屈折形式（如 `jump -> jumps`）带来什么挑战？

---

## 问题 2（24 分）
在 NLP 中，对文本序列 `S = {w1, w2, ..., wT}`，统计语言模型将联合概率写为条件概率乘积：

`P(S) = P(w1, w2, ..., wT) = Π_{t=1..T} p(w_t | w1, w2, ..., w_{t-1})`

但该形式在实践中计算代价很高，因此常用简化的 `n-gram` 模型：

`p(w_t | w1, w2, ..., w_{t-1}) ≈ p(w_t | w_{t-n+1}, ..., w_{t-1})`

通常使用 bi-gram（`n = 2`）或 tri-gram（`n = 3`）。考虑句子：

“I am taking CS6493 this semester and studying NLP is really fascinating”

假设按空格分词，并忽略标点。

1. n-gram 上下文提取与稀疏性分析（8 分）
- a. 列出所有第二个词为 `CS6493` 或 `NLP` 的 bigram；
- b. 列出所有第三个词为 `CS6493` 或 `NLP` 的 trigram。

2. n-gram 模型的批判性分析（6 分）
- a. 解释为什么 n-gram 不能泛化到由已知词组成但未见过的新序列；
- b. 讨论 Markov 假设如何限制长程依赖建模，并给出一个具体失败例子；
- c. 为什么 n-gram 需要大语料才能表现好？请结合“维度灾难”说明。

3. 神经语言模型设计与参数敏感性（10 分）
给定如下 PyTorch 代码：

```python
class LanguageModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, hidden_size = 128):
        super(LanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim = 1)
        return log_probs
```

目标：在给定句子上训练一个 four-gram 模型（由前三个词预测下一个词），训练 10 个 epoch。

- a. 说明如何预处理句子，构造训练样本 `(w_{t-3}, w_{t-2}, w_{t-1}) -> w_t`；
- b. 写出一个最小训练循环（伪代码或 Python），用于在生成数据上训练 1 个 epoch；
- c. 比较 embedding 维度为 32、64、128 时的训练动态（如 loss 收敛），并讨论容量、泛化、计算开销之间的权衡。

---

## 问题 3（45 分）
训练词向量有多种方式，其中常见两类是 Word2Vec 和 GloVe（Global Vectors for Word Representation）。

- Word2Vec：通过“由目标词预测上下文词（Skip-gram）”或“由上下文预测目标词（CBOW）”学习词向量；
- GloVe：通过分解全局词共现矩阵学习词向量。

### Word2Vec Skip-gram
Word2Vec Skip-gram 通过最大化“给定目标词观测到上下文词”的概率来学习词向量。实践中常用负采样近似 softmax 目标。

给定目标词 `w_i` 与上下文词 `w_o`，负采样目标为：

`L = log σ(u_{w_o}^T v_{w_i}) + Σ_{k=1..K} E_{w_k ~ P_n(w)} [log σ(-u_{w_k}^T v_{w_i})]`

其中：
- `v_w` 是输入（目标）词向量；
- `u_w` 是输出（上下文）词向量；
- `K` 是负样本数；
- `σ(·)` 是 sigmoid 函数；
- `P_n(w)` 是噪声分布。

### GloVe
GloVe 在全局词-词共现矩阵上最小化加权最小二乘目标：

`J = Σ_{i,j=1..V} f(X_ij) (w_i^T \\tilde{w}_j + b_i + \\tilde{b}_j - log X_ij)^2`

其中：
- `X_ij` 为词 `j` 在词 `i` 上下文中出现的次数；
- `f(·)` 为权重函数，用于防止高频共现主导训练；
- `w_i, \\tilde{w}_j` 为词向量，`b_i, \\tilde{b}_j` 为偏置项。

### 任务
1. 在提供的 `wiki_corpus.txt` 上训练带负采样的 Word2Vec Skip-gram 模型，使用合适超参数（如 embedding 维度、窗口大小、负采样率）。（15 分）
2. 在同一 `wiki_corpus.txt` 上训练 GloVe 模型。可自行实现，或使用现有 GloVe 实现/库，并保持与 Skip-gram 可比的超参数（相同 embedding 维度与上下文窗口）。（15 分）
3. 比较并评估两个模型质量：
   - 对选定词（如 “Australia”, “YMCA”, “South”, “building”）做近邻词定性分析；
   - 至少研究一个超参数的影响（如 embedding 维度：50/100/200，或窗口大小：2/5/10）。（15 分）

---

## 问题 4（20 分）
把文本切分为更小块并不简单，有多种方法。

1. 简单空格分词在某些文本上有局限。请从下列示例中至少选择两类，解释为什么空格分词在每类场景会出问题。（5 分）
- 复合词（如 `state-of-the-art`）
- 缩写/ contractions（如 `don't`, `we're`）
- URL/邮箱（如 `user@email.com`）
- 话题标签（如 `#MachineLearning`）
- 带单位数字（如 `3.14km`）

2. Transformer 常用一种介于词级和字符级之间的子词分词。BPE（Byte-Pair Encoding）是 Sennrich 等（2015）提出的子词方法。BPE 依赖一个 pre-tokenizer 将训练数据先切成“词”，该过程可简单采用空格分词。设预分词后得到如下词及频次：

`(old,10), (older,5), (oldest,8), (hug,8), (pug,4), (hugs,5)`

基础词表为：

`o, l, d, e, r, s, t, h, u, g, p`

将所有词拆成基础符号后得到：

`(o,l,d,10), (o,l,d,e,r,5), (o,l,d,e,s,t,8), (h,u,g,8), (p,u,g,4), (h,u,g,s,5)`

BPE 会统计所有相邻符号对频次，并选频次最高者合并。在上例中，`o` 后接 `l` 出现 `10 + 5 + 8 = 23` 次，因此第一条 merge 规则为 `o + l -> ol`，并把 `ol` 加入词表。此时词集合变为：

`(ol,d,10), (ol,d,e,r,5), (ol,d,e,s,t,8), (h,u,g,8), (p,u,g,4), (h,u,g,s,5)`

该过程迭代执行。词表大小（基础词表大小 + merge 次数）是可选超参数。学到的 merge 规则可用于新词（前提是新词不含基础词表外符号）；否则用 `[unk]` 表示。

请实现该 BPE tokenizer，设置词表大小为 16，并完成训练迭代过程。然后用训练后的 tokenizer 对以下词进行分词：（15 分）

`{hold, oldest, older, pug, mug, huggingface}`

---

