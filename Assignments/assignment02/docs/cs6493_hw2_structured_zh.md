# CS6493：自然语言处理 - 作业 2（中文整理版）

## 作业说明

1. 截止时间：**2026 年 4 月 29 日晚 10:00**。
2. 提交形式：
   - 单个 PDF（附代码包），或
   - 单个 Jupyter Notebook（同时包含答案与代码）。
3. 对于编程题，除代码外，建议补充：
   - 代码设计与流程说明，
   - 实验结果的详细分析。
4. 总分：**100 分**。
5. 如有问题：
   - 在 Canvas Discussion forum 提问，或
   - 联系助教 **Junyi ZHOU**（`junyizhou8-c@my.cityu.edu.hk`）。

---

## Question 1（20 分）

机器翻译是 NLP 中将一种语言的文本/语音自动翻译为另一种语言的任务。常见方法包括：基于规则的方法、统计模型、神经机器翻译（NMT）模型。

### 1）批处理中的索引化与 padding

模型使用矩阵加速，但训练时按 batch 输入；同一 batch 内句长需一致，因此要进行 padding（用 `0` 补齐）。

句子：
- Sentence 1: `I like cats and dogs`
- Sentence 2: `She enjoys reading books and playing video games`
- Sentence 3: `They play football every weekend`

词表（`token -> index`）：

| Token | Index |
|---|---:|
| I | 4 |
| She | 5 |
| They | 6 |
| like | 7 |
| enjoys | 8 |
| play | 9 |
| cats | 10 |
| dogs | 11 |
| reading | 12 |
| books | 13 |
| football | 14 |
| every | 15 |
| weekend | 16 |
| and | 17 |
| playing | 18 |
| video | 19 |
| games | 20 |

(a) 将每个句子分词后，按上述词表转为索引序列。**（2 分）**  
(b) 在右侧用 `0` 进行 padding，使所有序列长度与最长句一致。**（2 分）**

### 2）词嵌入（Word Embedding）

在神经 NLP 模型中，词通常先被映射为向量再作为输入。

(a) 什么是词嵌入？**（2 分）**  
(b) 相比 one-hot 编码，词嵌入的一个优势是什么？**（2 分）**

### 3）Greedy Search 与 Beam Search

在机器翻译/文本生成等序列生成任务中，greedy search 与 beam search 是常见解码算法。

`Figure 1`：解码树（每条边表示生成下一个 token 的概率）。

根据 Figure 1 回答：

(a) 使用 greedy search，给出生成序列并计算其概率。**（2 分）**  
(b) 使用 beam search（beam width `k = 2`），找出最可能序列并展示中间步骤。**（2 分）**  
(c) 简述为什么 beam search 往往比 greedy search 结果更好。**（2 分）**

### 4）BLEU 计算

BLEU（Bilingual Evaluation Understudy）是机器翻译中常见自动评测指标。

BLEU 公式：

\[
\mathrm{BLEU} = \mathrm{BP} \cdot \exp\left(\sum_{n=1}^{4} w_n \ln P_n\right)
\]

候选句（Candidate）：
- `A small cat is sitting on the wooden table.`

参考句 1（Reference 1）：
- `A little cat is sitting on the leather chair.`

参考句 2（Reference 2）：
- `A small cat is sitting on a wooden chair.`

设权重 \(w_n\) 均匀，计算 \(N = 4\) 时的 BLEU，并给出计算步骤。**（6 分）**

---

## Question 2（20 分）

传统 seq2seq（RNN/LSTM 编码器-解码器）受限于顺序计算，并且较难有效建模长距离依赖。Vaswani 等人 [3] 提出的 Transformer 以注意力机制为核心，去除了循环和卷积。

其核心操作之一为 Scaled Dot-Product Attention：

\[
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
\]

其中 \(d_k\) 为 key/query 维度。

`Figure 2`：Transformer 模型结构图。

### 1）Self-Attention 参数约束分析

通常：

\[
Q = XW^Q,\quad K = XW^K,\quad V = XW^V,\quad X \in \mathbb{R}^{n \times d_{\text{model}}}
\]

假设有人简化参数，令 \(W^Q = W^K\)，并进一步令 \(V = K\)。

在仅有 \(W^Q = W^K\) 约束时：
- 评分矩阵 \(S = QK^\top\) 必须满足什么性质？
- 这种性质会如何限制模型表示方向性/非对称关系的能力？

若进一步强制 \(V = K\)：
- 简要说明将 key 与 value 合并后损失了哪些表达能力。

**（8 分）**

### 2）缩放系数变化下的注意力分布

单查询注意力设定：
- 查询向量 \(q \in \mathbb{R}^{d_k}\)；
- 键向量集合 \(\{k_i\}_{i=1}^{n},\ k_i \in \mathbb{R}^{d_k}\)；
- 值向量集合 \(\{v_i\}_{i=1}^{n}\)（维度兼容）。

定义：

\[
\alpha_i = \frac{\exp(\beta q^\top k_i)}{\sum_{j=1}^{n}\exp(\beta q^\top k_j)},\quad
c = \sum_{i=1}^{n}\alpha_i v_i
\]

其中 \(\beta > 0\) 为 softmax 前 logit 缩放系数；当 \(\beta = 1/\sqrt{d_k}\) 时，对应 Transformer 原始 scaled dot-product attention [3]。

说明：
- 当 \(\beta \to 0\) 时，\(\{\alpha_i\}\) 趋于什么分布，对 \(c\) 有何影响；
- 当 \(\beta \to \infty\) 时会发生什么。

**（7 分）**

### 3）解码器因果掩码（Causal Mask）

在 masked self-attention 中，设 \(Q, K, V \in \mathbb{R}^{n \times d_k}\)：

\[
A = \mathrm{softmax}\left(\frac{QK^\top + M}{\sqrt{d_k}}\right),\quad O = AV
\]

其中 \(M \in \mathbb{R}^{n \times n}\) 为因果掩码，保证自回归约束（位置 \(t\) 不能关注未来位置 \(j > t\)）。

说明：
- 为什么掩码必须在 softmax **之前** 加入（而不是 softmax 后再将部分权重置 0）；
- 当 \(n = 4\) 时，写出由 `0` 与 `-∞` 组成的显式 \(4 \times 4\) 掩码矩阵 \(M\)。

**（5 分）**

---

## Question 3（30 分）

基于课程中对理解任务（NLU）与生成任务（NLG）的内容：

### 1）NLU 与 NLG 的差异

用你自己的话说明 NLU 与 NLG 的关键区别，并各给出 **两个** 具体任务示例。**（4 分）**

### 2）分类任务类型判断

对于以下文本分类应用，判断最合适的分类类型（如二分类、单标签多分类等）：

(a) 判断邮件是垃圾邮件还是正常邮件。**（2 分）**  
(b) 将新闻分到 `{HomeNews, International, Entertainment, Lifestyle, Sports}` 中且只能选一个。**（2 分）**  
(c) 给技术论文打上分类体系中所有适用标签。**（2 分）**  
(d) 预测评论等级，标签有序：`{Disastrous, Poor, Mediocre, Good, Excellent}`。**（2 分）**

### 3）阅读理解与 SQuAD

阅读理解（RC）可表述为“给定篇章回答问题”，答案可能是抽取式或生成式。

(a) 用一句话说明在 SQuAD 风格 [2] 下，什么是抽取式 QA。**（2 分）**  
(b) 给出两点理由说明 SQuAD 相比一些早期 QA 数据集更好。**（4 分）**  
(c) SQuAD 2.0 [1] 含大量不可回答问题；系统要在该数据集上表现好，还需具备什么额外能力？**（2 分）**

### 4）解码策略与 BLEU

对于序列生成（如机器翻译）：

(a) 什么是 greedy decoding？其主要缺点是什么？**（2 分）**  
(b) 为什么穷举搜索解码在实践中不可行？**（2 分）**  
(c) 简述 beam search、beam size `k` 控制什么，以及为何常用长度归一化。**（2 分）**  
(d) BLEU 的 n-gram 精确率与 brevity penalty 分别试图抑制什么问题。**（2 分）**

### 5）启发式自动评测指标

人工评测常常昂贵且耗时，因此实际系统会采用启发式自动指标评估生成质量。

请写出两个此类指标，并说明它们大致想衡量什么。**（2 分）**

---

## Question 4（30 分）

大语言模型与问答系统已广泛用于教育、客服和组织内部知识管理等真实场景。

### 1）校园助手系统设计（15 分）

某大学希望部署 AI 助手，用于回答学生关于以下内容的问题：
- 选课注册、
- 截止日期、
- 评分政策、
- 毕业要求。

信息来源包括 PDF、网页和政策文档，且每学期更新。

系统要求：
- 答案必须基于官方文档；
- 需降低错误或过时信息风险。

(a) 给出系统设计方案，清晰描述核心组件（如文档存储、检索、语言模型、重排等）。**（8 分）**  
(b) 描述从学生提问到系统生成答案的完整步骤流程。**（7 分）**

### 2）不确定或缺失信息处理（15 分）

系统上线后发现，学生问题有时：
- 文档中没有覆盖，或
- 表述含糊/不清晰。

当前系统仍强行作答，偶尔产生错误回复。

(a) 解释为什么现代语言模型在不确定时仍可能给出答案。**（6 分）**  
(b) 提出两种可落地的策略提升系统在此类场景下的安全性与可靠性，并说明各自作用。**（9 分）**

---

## 参考文献

1. P. Rajpurkar, R. Jia, and P. Liang. *Know What You Don’t Know: Unanswerable Questions for SQuAD*. arXiv:1806.03822, 2018.
2. P. Rajpurkar, J. Zhang, K. Lopyrev, and P. Liang. *SQuAD: 100,000+ Questions for Machine Comprehension of Text*. arXiv:1606.05250, 2016.
3. A. Vaswani et al. *Attention Is All You Need*. NeurIPS 2017.
