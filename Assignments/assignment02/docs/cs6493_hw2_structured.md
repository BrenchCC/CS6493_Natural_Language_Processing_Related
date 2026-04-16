# CS6493: Natural Language Processing - Assignment 2

## Instructions

1. Due at **10:00 PM, April 29, 2026**.
2. You can submit your answers as:
   - a single PDF with the code package, or
   - a single Jupyter notebook containing both answers and code.
3. For coding questions, besides code, you are encouraged to provide:
   - a brief description of code design and workflow,
   - detailed analysis of experimental results.
4. Total marks: **100**.
5. Questions:
   - Post on Canvas Discussion forum, or
   - Contact TA: **Mr. Junyi ZHOU** (`junyizhou8-c@my.cityu.edu.hk`).

---

## Question 1 (20 marks)

Machine translation in NLP is the task of automatically translating text/speech between languages using computational methods. Approaches include rule-based methods, statistical models, and neural machine translation (NMT).

### 1) Padding and indexing in batched training

The model uses matrix operations and batch training. Since sentence lengths differ, padding is required (fill with `0` to equalize lengths).

Sentences:
- Sentence 1: `I like cats and dogs`
- Sentence 2: `She enjoys reading books and playing video games`
- Sentence 3: `They play football every weekend`

Vocabulary (`token -> index`):

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

(a) Tokenize each sentence and convert to index sequence using the vocabulary above. **(2 marks)**  
(b) Pad all sequences on the right with `0` to the same length as the longest sentence. **(2 marks)**

### 2) Word embeddings

In neural NLP models, words are often converted to vectors before being used as input.

(a) What is a word embedding? **(2 marks)**  
(b) Explain one advantage of word embeddings compared with one-hot encoding. **(2 marks)**

### 3) Greedy search vs beam search

Beam search and greedy search are common decoding algorithms for sequence generation tasks (e.g., machine translation, text generation).

`Figure 1`: Decoding tree (each edge indicates probability of generating the next token).

Answer based on Figure 1:

(a) Using greedy search, find the generated sequence and compute its probability. **(2 marks)**  
(b) Using beam search with beam width `k = 2`, find the most probable sequence and show intermediate steps. **(2 marks)**  
(c) Briefly explain why beam search can produce better results than greedy search. **(2 marks)**

### 4) BLEU calculation

BLEU (Bilingual Evaluation Understudy) is a common metric for machine translation quality.

BLEU formula:

\[
\mathrm{BLEU} = \mathrm{BP} \cdot \exp\left(\sum_{n=1}^{4} w_n \ln P_n\right)
\]

Candidate sentence:
- `A small cat is sitting on the wooden table.`

Reference sentence 1:
- `A little cat is sitting on the leather chair.`

Reference sentence 2:
- `A small cat is sitting on a wooden chair.`

Assume uniform weights \(w_n\). Compute BLEU with \(N = 4\), and show calculation steps. **(6 marks)**

---

## Question 2 (20 marks)

Traditional seq2seq models (RNN/LSTM encoder-decoder) are limited by sequential computation and long-range dependency modeling. Vaswani et al. [3] proposed the Transformer, based entirely on attention, removing recurrence and convolution from core sequence modeling.

A core operation is Scaled Dot-Product Attention:

\[
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
\]

where \(d_k\) is key/query dimensionality.

`Figure 2`: Transformer model architecture.

### 1) Constraint analysis in self-attention

In self-attention, typically:

\[
Q = XW^Q,\quad K = XW^K,\quad V = XW^V,\quad X \in \mathbb{R}^{n \times d_{\text{model}}}
\]

Suppose someone simplifies parameters by setting \(W^Q = W^K\), and even further sets \(V = K\).

Under only \(W^Q = W^K\):
- What property must the score matrix \(S = QK^\top\) satisfy?
- How does that property limit representation of directional/asymmetric relations?

If additionally \(V = K\):
- Briefly explain what expressivity is lost by conflating keys and values.

**(8 marks)**

### 2) Behavior under different scaling coefficients

Single-query attention with:
- query \(q \in \mathbb{R}^{d_k}\),
- keys \(\{k_i\}_{i=1}^{n},\ k_i \in \mathbb{R}^{d_k}\),
- values \(\{v_i\}_{i=1}^{n}\) (compatible dimensionality).

Define:

\[
\alpha_i = \frac{\exp(\beta q^\top k_i)}{\sum_{j=1}^{n}\exp(\beta q^\top k_j)},\quad
c = \sum_{i=1}^{n}\alpha_i v_i
\]

where \(\beta > 0\) scales dot-product logits before softmax; \(\beta = 1/\sqrt{d_k}\) recovers the Transformer scaling [3].

Describe:
- what distribution \(\{\alpha_i\}\) approaches as \(\beta \to 0\), and effect on \(c\),
- what happens as \(\beta \to \infty\).

**(7 marks)**

### 3) Causal mask in decoder self-attention

In masked self-attention, let \(Q, K, V \in \mathbb{R}^{n \times d_k}\):

\[
A = \mathrm{softmax}\left(\frac{QK^\top + M}{\sqrt{d_k}}\right),\quad O = AV
\]

where \(M \in \mathbb{R}^{n \times n}\) is the causal mask enforcing autoregression (position \(t\) cannot attend to future position \(j > t\)).

Explain:
- why the mask must be added **before** softmax (instead of zeroing weights after softmax),
- for \(n = 4\), write the explicit \(4 \times 4\) mask \(M\) using entries `0` and `-∞`.

**(5 marks)**

---

## Question 3 (30 marks)

Based on lecture materials on understanding tasks (NLU) and generation tasks (NLG):

### 1) NLU vs NLG

State the key difference between NLU and NLG in your own words, and give **two concrete task examples** for each. **(4 marks)**

### 2) Classification type identification

For each application, identify the most appropriate classification type (e.g., binary, single-label multi-class, etc.):

(a) Decide whether an email is spam or legitimate. **(2 marks)**  
(b) Assign a news article to exactly one section from `{HomeNews, International, Entertainment, Lifestyle, Sports}`. **(2 marks)**  
(c) Tag a technical paper with all applicable category labels in a taxonomy. **(2 marks)**  
(d) Predict a review rating from ordered labels `{Disastrous, Poor, Mediocre, Good, Excellent}`. **(2 marks)**

### 3) Reading comprehension and SQuAD

Reading comprehension can be framed as answering questions based on a passage, with extractive or abstractive answers.

(a) In one sentence, explain extractive QA in SQuAD-style RC [2]. **(2 marks)**  
(b) Give two reasons why SQuAD was considered better than some earlier QA datasets. **(4 marks)**  
(c) SQuAD 2.0 [1] includes many unanswerable questions; what additional capability is needed to perform well? **(2 marks)**

### 4) Decoding and BLEU

For sequence generation (e.g., machine translation):

(a) What is greedy decoding, and what is its main drawback? **(2 marks)**  
(b) Why is exhaustive search decoding impractical? **(2 marks)**  
(c) Briefly explain beam search, what beam size `k` controls, and why length normalization is commonly used. **(2 marks)**  
(d) BLEU uses n-gram precision and brevity penalty; explain what each component is intended to discourage. **(2 marks)**

### 5) Heuristic automatic metrics

Human evaluation is often expensive/slow, so practical systems use heuristic automatic metrics for generation quality.

Name two heuristic metrics and state what each is roughly intended to capture. **(2 marks)**

---

## Question 4 (30 marks)

Large language models and question answering systems are increasingly used in education, customer service, and internal knowledge management.

### 1) Designing a campus assistant system (15 marks)

A university plans to deploy an AI assistant to answer student questions on:
- course registration,
- deadlines,
- grading policies,
- graduation requirements.

Information sources: PDFs, web pages, policy documents (updated every semester).

Requirements:
- Answers must be grounded in official documents.
- The system should reduce risk of incorrect/outdated information.

(a) Propose a system design. Clearly describe main components (e.g., document storage, retrieval, language model, ranking). **(8 marks)**  
(b) Describe the step-by-step workflow from student query to final system answer. **(7 marks)**

### 2) Handling uncertain or missing information (15 marks)

After deployment, some student queries are:
- not covered by documents, or
- ambiguous/poorly phrased.

Current system still attempts answers and sometimes responds incorrectly.

(a) Explain why modern language models may still produce answers even when uncertain. **(6 marks)**  
(b) Propose two practical strategies to improve safety and reliability in these cases, and explain how each helps. **(9 marks)**

---

## References

1. P. Rajpurkar, R. Jia, and P. Liang. *Know What You Don’t Know: Unanswerable Questions for SQuAD*. arXiv:1806.03822, 2018.
2. P. Rajpurkar, J. Zhang, K. Lopyrev, and P. Liang. *SQuAD: 100,000+ Questions for Machine Comprehension of Text*. arXiv:1606.05250, 2016.
3. A. Vaswani et al. *Attention Is All You Need*. NeurIPS 2017.
