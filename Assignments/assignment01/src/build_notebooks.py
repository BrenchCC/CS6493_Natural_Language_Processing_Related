import json
from pathlib import Path

import nbformat as nbf


def load_results(results_dir):
    """Load all result json files.

    Parameters:
        results_dir (Path): Results directory.

    Returns:
        tuple[dict, dict, dict, dict]: Q1, Q2, Q3, Q4 results.
    """
    q1 = json.loads((results_dir / "q1_results.json").read_text(encoding = "utf-8"))
    q2 = json.loads((results_dir / "q2_results.json").read_text(encoding = "utf-8"))
    q3 = json.loads((results_dir / "q3_summary.json").read_text(encoding = "utf-8"))
    q4 = json.loads((results_dir / "q4_bpe_results.json").read_text(encoding = "utf-8"))
    return q1, q2, q3, q4


def md_cell(text):
    """Create markdown cell.

    Parameters:
        text (str): Markdown content.

    Returns:
        NotebookNode: Markdown cell.
    """
    return nbf.v4.new_markdown_cell(text)


def code_cell(text):
    """Create code cell.

    Parameters:
        text (str): Python code content.

    Returns:
        NotebookNode: Code cell.
    """
    return nbf.v4.new_code_cell(text)


def build_en_notebook(base_dir, q1, q2, q3, q4):
    """Build English notebook.

    Parameters:
        base_dir (Path): Submission directory.
        q1 (dict): Q1 results.
        q2 (dict): Q2 results.
        q3 (dict): Q3 results.
        q4 (dict): Q4 results.

    Returns:
        NotebookNode: Notebook object.
    """
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(md_cell("# CS6493 NLP - Assignment 1 (English Version)\n\nThis notebook contains my answers for Q1-Q4 with runnable code and experiment outputs."))

    cells.append(code_cell(
        "from pathlib import Path\n"
        "import json\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n\n"
        "BASE_DIR = Path('Assignments/assignment01/submission')\n"
        "RESULTS_DIR = BASE_DIR / 'results'\n"
        "SRC_DIR = BASE_DIR / 'src'\n\n"
        "q1 = json.loads((RESULTS_DIR / 'q1_results.json').read_text(encoding = 'utf-8'))\n"
        "q2 = json.loads((RESULTS_DIR / 'q2_results.json').read_text(encoding = 'utf-8'))\n"
        "q3 = json.loads((RESULTS_DIR / 'q3_summary.json').read_text(encoding = 'utf-8'))\n"
        "q4 = json.loads((RESULTS_DIR / 'q4_bpe_results.json').read_text(encoding = 'utf-8'))\n"
        "print('Loaded results:', sorted([p.name for p in RESULTS_DIR.glob('*')]))"
    ))

    cells.append(md_cell(
        "## Question 1 (11 marks)\n\n"
        "### 1) One-hot vectors (lexicographic order)\n"
        f"Vocabulary: `{q1['vocab']}`\n\n"
        "- cat -> [1, 0, 0, 0]\n"
        "- dog -> [0, 1, 0, 0]\n"
        "- run -> [0, 0, 1, 0]\n"
        "- runs -> [0, 0, 0, 1]\n\n"
        "### 2) Dot products and similarity\n"
        "For any two distinct words, the dot product is **0**.\n"
        "This means one-hot vectors treat all different words as equally unrelated.\n"
        "That is bad for NLP because semantic or morphological similarity is not represented.\n\n"
        "### 3) Parameter efficiency\n"
        "- `|V| = 4, d = 128` -> `4 x 128 = 512` parameters\n"
        "- `|V| = 50,000, d = 128` -> `6,400,000` parameters\n"
        "Two practical issues: large memory/computation and no parameter sharing.\n\n"
        "### 4) Morphology\n"
        "`run` and `runs` cannot be recognized as related from one-hot vectors alone.\n"
        "So it is hard to generalize from seen forms to unseen inflections like `jump -> jumps`.\n\n"
        "**Short reflection:** this question shows why distributed embeddings are necessary."
    ))

    cells.append(md_cell(
        "## Question 2 (24 marks)\n\n"
        "Sentence: `I am taking CS6493 this semester and studying NLP is really fascinating`\n\n"
        "### 1) n-gram extraction\n"
        "- Bigrams with second word in {CS6493, NLP}:\n"
        "  - (taking, CS6493)\n"
        "  - (studying, NLP)\n"
        "- Trigrams with third word in {CS6493, NLP}:\n"
        "  - (am, taking, CS6493)\n"
        "  - (and, studying, NLP)\n\n"
        "### 2) Critical discussion\n"
        "a. n-gram counts seen patterns only; unseen but valid combinations get near-zero support.\n\n"
        "b. Markov assumption ignores long context. Example: in `The key to the cabinets ... was/were`,"
        " agreement can depend on a far-away head word.\n\n"
        "c. n-gram dimensionality grows fast with n and vocabulary size, so large corpora are needed to reduce sparsity.\n\n"
        "### 3) Four-gram neural LM experiment\n"
        "I trained a small two-layer model for 10 epochs with embedding sizes 32/64/128."
    ))

    cells.append(code_cell(
        "import matplotlib.image as mpimg\n"
        "img = mpimg.imread(RESULTS_DIR / 'q2_loss_curve.png')\n"
        "plt.figure(figsize = (8, 5))\n"
        "plt.imshow(img)\n"
        "plt.axis('off')\n"
        "plt.show()\n\n"
        "pd.DataFrame(q2['losses'])"
    ))

    cells.append(md_cell(
        "**Observation:** all three dimensions converge smoothly on this tiny dataset.\n"
        "Higher dimensions do not show large gains here because the training set is very small.\n"
        "Cost still grows with embedding size, so dim=64 is a reasonable middle choice for this case."
    ))

    cells.append(md_cell(
        "## Question 3 (45 marks)\n\n"
        "I trained two models on `wiki_corpus.txt`:\n"
        "- SGNS (Skip-gram + negative sampling, implemented from scratch)\n"
        "- GloVe (implemented from scratch with weighted least squares + AdaGrad)\n\n"
        f"Corpus used: {q3['num_sentences']} sentences\n\n"
        f"Vocabulary size (min_count = 5): {q3['vocab_size']}\n\n"
        f"Co-occurrence entries: {q3['cooc_size']}\n\n"
        "Hyperparameter study: embedding dimensions = 50, 100, 200"
    ))

    cells.append(code_cell(
        "img = mpimg.imread(RESULTS_DIR / 'q3_loss_curve.png')\n"
        "plt.figure(figsize = (10, 6))\n"
        "plt.imshow(img)\n"
        "plt.axis('off')\n"
        "plt.show()\n\n"
        "pd.read_csv(RESULTS_DIR / 'q3_loss_table.csv').head(12)"
    ))

    cells.append(code_cell(
        "dim = '100'\n"
        "rows = []\n"
        "for word, item in q3['dims'][dim]['neighbors'].items():\n"
        "    rows.append({'word': word, 'model': 'sgns', 'top5': ', '.join([w for w, _ in item['sgns'][:5]])})\n"
        "    rows.append({'word': word, 'model': 'glove', 'top5': ', '.join([w for w, _ in item['glove'][:5]])})\n"
        "pd.DataFrame(rows)"
    ))

    cells.append(md_cell(
        "**Observation:** SGNS and GloVe both learn distributional patterns, but neighbors are still noisy.\n"
        "This corpus is medium-sized and mixed-domain, so frequent function-like words can dominate local similarity.\n"
        "Increasing dimension from 50 to 200 improves fit, but the semantic gain is not always proportional."
    ))

    cells.append(md_cell(
        "## Question 4 (20 marks)\n\n"
        "### 1) Why space tokenization fails (3 cases)\n"
        "- Contractions (`don't`, `we're`): meaning and morphology are fused in one token.\n"
        "- URLs/emails (`user@email.com`): punctuation is structural, not simple separators.\n"
        "- Numbers with units (`3.14km`): numeric value and unit should often be split.\n\n"
        "### 2) BPE implementation (vocab size = 16)\n"
        "I started from base symbols `{o,l,d,e,r,s,t,h,u,g,p}` and performed 5 merges."
    ))

    cells.append(code_cell(
        "pd.DataFrame(q4['merge_history'])"
    ))

    cells.append(code_cell(
        "pd.DataFrame([{'word': k, 'tokens': ' '.join(v)} for k, v in q4['tokens'].items()])"
    ))

    cells.append(md_cell(
        "`mug` and `huggingface` contain characters outside the base vocabulary (`m`, `i`, `n`, `f`, `a`, `c`...),\n"
        "so they become `[unk]` under this toy setup.\n\n"
        "## Final Notes\n"
        "- All outputs in this notebook come from runnable scripts in `submission/src`.\n"
        "- I kept the implementation small and readable to match an assignment setting."
    ))

    nb["cells"] = cells
    return nb


def build_zh_notebook(base_dir, q1, q2, q3, q4):
    """Build Chinese notebook.

    Parameters:
        base_dir (Path): Submission directory.
        q1 (dict): Q1 results.
        q2 (dict): Q2 results.
        q3 (dict): Q3 results.
        q4 (dict): Q4 results.

    Returns:
        NotebookNode: Notebook object.
    """
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(md_cell("# CS6493 自然语言处理 - 作业 1（中文版本）\n\n本 notebook 按 Q1-Q4 给出答案、代码与实验结果。"))

    cells.append(code_cell(
        "from pathlib import Path\n"
        "import json\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n\n"
        "BASE_DIR = Path('Assignments/assignment01/submission')\n"
        "RESULTS_DIR = BASE_DIR / 'results'\n"
        "SRC_DIR = BASE_DIR / 'src'\n\n"
        "q1 = json.loads((RESULTS_DIR / 'q1_results.json').read_text(encoding = 'utf-8'))\n"
        "q2 = json.loads((RESULTS_DIR / 'q2_results.json').read_text(encoding = 'utf-8'))\n"
        "q3 = json.loads((RESULTS_DIR / 'q3_summary.json').read_text(encoding = 'utf-8'))\n"
        "q4 = json.loads((RESULTS_DIR / 'q4_bpe_results.json').read_text(encoding = 'utf-8'))\n"
        "print('已加载结果文件数量:', len(list(RESULTS_DIR.glob('*'))))"
    ))

    cells.append(md_cell(
        "## 问题 1（11 分）\n\n"
        "### 1) one-hot 向量（按字典序）\n"
        f"词表：`{q1['vocab']}`\n\n"
        "- cat -> [1, 0, 0, 0]\n"
        "- dog -> [0, 1, 0, 0]\n"
        "- run -> [0, 0, 1, 0]\n"
        "- runs -> [0, 0, 0, 1]\n\n"
        "### 2) 点积与相似性\n"
        "任意两个不同词的点积都是 **0**。\n"
        "这表示 one-hot 只能区分“是不是同一个词”，不能表示“有多相近”。\n"
        "因此语义相似和词形关系都丢失。\n\n"
        "### 3) 参数量\n"
        "- `|V| = 4, d = 128` -> `4 x 128 = 512`\n"
        "- `|V| = 50,000, d = 128` -> `6,400,000`\n"
        "问题：内存/计算开销大；相关词之间不能共享信息。\n\n"
        "### 4) 形态关系\n"
        "`run` 与 `runs` 的关系无法从 one-hot 自动识别。\n"
        "所以对 `jump -> jumps` 这类未见过的屈折形式泛化会很弱。\n\n"
        "**简短反思：** 这也是后来词向量方法必要的核心动机。"
    ))

    cells.append(md_cell(
        "## 问题 2（24 分）\n\n"
        "句子：`I am taking CS6493 this semester and studying NLP is really fascinating`\n\n"
        "### 1) n-gram 提取\n"
        "- 第二词为 CS6493 或 NLP 的 bigram：\n"
        "  - (taking, CS6493)\n"
        "  - (studying, NLP)\n"
        "- 第三词为 CS6493 或 NLP 的 trigram：\n"
        "  - (am, taking, CS6493)\n"
        "  - (and, studying, NLP)\n\n"
        "### 2) n-gram 局限\n"
        "a. 没见过的组合就很难估计，已知词也不能保证可泛化。\n\n"
        "b. 马尔可夫假设只看短上下文，长距离依赖容易丢。\n"
        "例如主谓一致可能依赖很前面的中心词。\n\n"
        "c. n 越大维度越爆炸，数据稀疏更严重，所以要更大语料才稳定。\n\n"
        "### 3) 四元语言模型实验\n"
        "我用两层小网络训练 10 个 epoch，对比 embedding 维度 32/64/128。"
    ))

    cells.append(code_cell(
        "import matplotlib.image as mpimg\n"
        "img = mpimg.imread(RESULTS_DIR / 'q2_loss_curve.png')\n"
        "plt.figure(figsize = (8, 5))\n"
        "plt.imshow(img)\n"
        "plt.axis('off')\n"
        "plt.show()\n\n"
        "pd.DataFrame(q2['losses'])"
    ))

    cells.append(md_cell(
        "**观察：** 这个数据集非常小，三种维度都能平稳下降，差异不大。\n"
        "维度更大不一定带来明显收益，但训练和参数成本会上升。\n"
        "这里 dim=64 是比较均衡的选择。"
    ))

    cells.append(md_cell(
        "## 问题 3（45 分）\n\n"
        "我在 `wiki_corpus.txt` 上训练了两种词向量模型：\n"
        "- SGNS（Skip-gram + 负采样，自实现）\n"
        "- GloVe（加权最小二乘 + AdaGrad，自实现）\n\n"
        f"使用语料句数：{q3['num_sentences']}\n\n"
        f"词表大小（min_count = 5）：{q3['vocab_size']}\n\n"
        f"共现对数量：{q3['cooc_size']}\n\n"
        "超参数分析：embedding 维度 = 50 / 100 / 200"
    ))

    cells.append(code_cell(
        "img = mpimg.imread(RESULTS_DIR / 'q3_loss_curve.png')\n"
        "plt.figure(figsize = (10, 6))\n"
        "plt.imshow(img)\n"
        "plt.axis('off')\n"
        "plt.show()\n\n"
        "pd.read_csv(RESULTS_DIR / 'q3_loss_table.csv').head(12)"
    ))

    cells.append(code_cell(
        "dim = '100'\n"
        "rows = []\n"
        "for word, item in q3['dims'][dim]['neighbors'].items():\n"
        "    rows.append({'词': word, '模型': 'sgns', 'Top5近邻': ', '.join([w for w, _ in item['sgns'][:5]])})\n"
        "    rows.append({'词': word, '模型': 'glove', 'Top5近邻': ', '.join([w for w, _ in item['glove'][:5]])})\n"
        "pd.DataFrame(rows)"
    ))

    cells.append(md_cell(
        "**观察：** 两个模型都学到了分布统计信息，但近邻仍有噪声。\n"
        "语料是中等规模且主题混合，频繁词会干扰语义近邻。\n"
        "维度从 50 提到 200 后拟合更强，但语义质量提升不是线性的。"
    ))

    cells.append(md_cell(
        "## 问题 4（20 分）\n\n"
        "### 1) 空格分词的问题（3 类）\n"
        "- contractions（`don't`, `we're`）：词形和语义粘连。\n"
        "- URL/邮箱（`user@email.com`）：符号不是普通分隔符。\n"
        "- 数字+单位（`3.14km`）：通常要拆成数值和单位。\n\n"
        "### 2) BPE 实现（词表大小 16）\n"
        "从 `{o,l,d,e,r,s,t,h,u,g,p}` 出发，执行 5 次 merge。"
    ))

    cells.append(code_cell("pd.DataFrame(q4['merge_history'])"))

    cells.append(code_cell(
        "pd.DataFrame([{'词': k, '分词结果': ' '.join(v)} for k, v in q4['tokens'].items()])"
    ))

    cells.append(md_cell(
        "`mug`、`huggingface` 包含基础词表外字符（如 `m`、`i`、`n`、`f`、`a`、`c`），\n"
        "因此在这个 toy BPE 设定中会输出 `[unk]`。\n\n"
        "## 最后说明\n"
        "- 结果由 `submission/src` 下脚本实际运行得到。\n"
        "- 我尽量保持代码简洁、可读，符合课程作业场景。"
    ))

    nb["cells"] = cells
    return nb


def main():
    """Generate both notebooks.

    Parameters:
        None.

    Returns:
        None
    """
    submission_dir = Path("Assignments/assignment01/submission")
    results_dir = submission_dir / "results"
    q1, q2, q3, q4 = load_results(results_dir)

    nb_en = build_en_notebook(submission_dir, q1, q2, q3, q4)
    nb_zh = build_zh_notebook(submission_dir, q1, q2, q3, q4)

    en_path = submission_dir / "HW1_submission_en.ipynb"
    zh_path = submission_dir / "HW1_submission_zh.ipynb"

    nbf.write(nb_en, en_path)
    nbf.write(nb_zh, zh_path)

    print(f"Saved: {en_path}")
    print(f"Saved: {zh_path}")


if __name__ == "__main__":
    main()
