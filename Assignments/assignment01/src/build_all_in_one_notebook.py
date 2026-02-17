from pathlib import Path

import nbformat as nbf


OUTPUT_PATH = Path("Assignments/assignment01/HW1_submission_all_in_one.ipynb")


def md_cell(text):
    """Create markdown cell.

    Parameters:
        text (str): Markdown text.

    Returns:
        NotebookNode: Markdown cell.
    """
    return nbf.v4.new_markdown_cell(text)


def code_cell(text):
    """Create code cell.

    Parameters:
        text (str): Python code text.

    Returns:
        NotebookNode: Code cell.
    """
    return nbf.v4.new_code_cell(text)


def build_notebook():
    """Build all-in-one assignment notebook.

    Parameters:
        None.

    Returns:
        NotebookNode: Notebook object.
    """
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(md_cell(
        "# CS6493 NLP Assignment 1 - All-in-One Notebook\n\n"
        "This file includes explanations and code in one place.\n"
        "Training code is included, but if cached results already exist, the notebook reads them directly."
    ))

    cells.append(code_cell(
        "import json\n"
        "import random\n"
        "import re\n"
        "from collections import Counter, defaultdict\n"
        "from pathlib import Path\n\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import torch\n"
        "import torch.nn as nn\n"
        "import torch.nn.functional as F\n"
        "from tqdm.auto import tqdm\n\n"
        "random.seed(42)\n"
        "np.random.seed(42)\n\n"
        "RESULTS_DIR = Path('Assignments/assignment01/results')\n"
        "RESULTS_DIR.mkdir(parents = True, exist_ok = True)\n"
        "print('Results directory:', RESULTS_DIR)"
    ))

    cells.append(md_cell(
        "## Question 1 (11 marks)\n\n"
        "Given vocabulary `V = {cat, dog, run, runs}` in lexicographic order."
    ))

    cells.append(code_cell(
        "vocab = ['cat', 'dog', 'run', 'runs']\n"
        "one_hot = {}\n"
        "for i, word in enumerate(vocab):\n"
        "    vec = [0] * len(vocab)\n"
        "    vec[i] = 1\n"
        "    one_hot[word] = vec\n\n"
        "dot_distinct = 0\n"
        "params_small = 4 * 128\n"
        "params_large = 50000 * 128\n\n"
        "display(pd.DataFrame({'word': vocab, 'one_hot': [one_hot[w] for w in vocab]}))\n"
        "print('Dot product between distinct one-hot vectors:', dot_distinct)\n"
        "print('Params for |V|=4, d=128:', params_small)\n"
        "print('Params for |V|=50000, d=128:', params_large)"
    ))

    cells.append(md_cell(
        "**Q1 conclusion**\n"
        "- Distinct one-hot vectors are orthogonal (dot product = 0), so `cat` is as unrelated to `dog` as to `runs` in this space.\n"
        "- Parameter count jumps from **512** (`4 x 128`) to **6,400,000** (`50,000 x 128`), which quickly increases memory and optimization cost.\n"
        "- Morphological relation is missing: `run` and `runs` are independent one-hot ids, so the model cannot transfer this pattern to unseen pairs like `jump/jumps`."
    ))

    cells.append(md_cell(
        "## Question 2 (24 marks)\n\n"
        "Sentence: `I am taking CS6493 this semester and studying NLP is really fascinating`\n"
        "- bigrams with second word in {CS6493, NLP}\n"
        "- trigrams with third word in {CS6493, NLP}"
    ))

    cells.append(code_cell(
        "tokens = 'I am taking CS6493 this semester and studying NLP is really fascinating'.split()\n"
        "bigrams = [(tokens[i - 1], tokens[i]) for i in range(1, len(tokens)) if tokens[i] in {'CS6493', 'NLP'}]\n"
        "trigrams = [(tokens[i - 2], tokens[i - 1], tokens[i]) for i in range(2, len(tokens)) if tokens[i] in {'CS6493', 'NLP'}]\n\n"
        "print('bigrams:', bigrams)\n"
        "print('trigrams:', trigrams)"
    ))

    cells.append(code_cell(
        "class FourGramLM(nn.Module):\n"
        "    def __init__(self, vocab_size, embedding_dim, hidden_size = 128):\n"
        "        super().__init__()\n"
        "        self.embeddings = nn.Embedding(vocab_size, embedding_dim)\n"
        "        self.linear1 = nn.Linear(3 * embedding_dim, hidden_size)\n"
        "        self.linear2 = nn.Linear(hidden_size, vocab_size)\n\n"
        "    def forward(self, context_ids):\n"
        "        embeds = self.embeddings(context_ids).view(1, -1)\n"
        "        hidden = F.relu(self.linear1(embeds))\n"
        "        logits = self.linear2(hidden)\n"
        "        return F.log_softmax(logits, dim = 1)\n\n"
        "def train_fourgram_torch(tokens, embedding_dim = 64, hidden_size = 128, epochs = 10, lr = 0.08, seed = 42):\n"
        "    torch.manual_seed(seed + embedding_dim)\n"
        "    vocab = sorted(set(tokens))\n"
        "    w2i = {w: i for i, w in enumerate(vocab)}\n"
        "    examples = []\n"
        "    for i in range(3, len(tokens)):\n"
        "        examples.append(([tokens[i - 3], tokens[i - 2], tokens[i - 1]], tokens[i]))\n\n"
        "    model = FourGramLM(len(vocab), embedding_dim, hidden_size = hidden_size)\n"
        "    optimizer = torch.optim.SGD(model.parameters(), lr = lr)\n"
        "    criterion = nn.NLLLoss()\n\n"
        "    losses = []\n"
        "    for _ in tqdm(range(epochs), desc = f'Q2 dim={embedding_dim}', leave = False):\n"
        "        total = 0.0\n"
        "        for context, target in examples:\n"
        "            context_ids = torch.tensor([w2i[w] for w in context], dtype = torch.long)\n"
        "            target_id = torch.tensor([w2i[target]], dtype = torch.long)\n"
        "            optimizer.zero_grad()\n"
        "            log_probs = model(context_ids)\n"
        "            loss = criterion(log_probs, target_id)\n"
        "            loss.backward()\n"
        "            optimizer.step()\n"
        "            total += loss.item()\n"
        "        losses.append(total / len(examples))\n"
        "    return losses\n\n"
        "q2_path = RESULTS_DIR / 'q2_results.json'\n"
        "if q2_path.exists():\n"
        "    q2_payload = json.loads(q2_path.read_text(encoding = 'utf-8'))\n"
        "    q2_losses = {int(k): v for k, v in q2_payload['losses'].items()}\n"
        "    print('Loaded cached Q2 results from', q2_path)\n"
        "else:\n"
        "    q2_losses = {}\n"
        "    for dim in [32, 64, 128]:\n"
        "        q2_losses[dim] = train_fourgram_torch(tokens, embedding_dim = dim)\n"
        "    q2_payload = {\n"
        "        'losses': {str(k): v for k, v in q2_losses.items()},\n"
        "        'sentence_tokens': tokens\n"
        "    }\n"
        "    q2_path.write_text(json.dumps(q2_payload, ensure_ascii = False, indent = 2), encoding = 'utf-8')\n"
        "    print('Trained Q2 and saved results to', q2_path)\n\n"
        "plt.figure(figsize = (8, 5))\n"
        "for dim in [32, 64, 128]:\n"
        "    plt.plot(range(1, 11), q2_losses[dim], marker = 'o', label = f'dim={dim}')\n"
        "plt.xlabel('Epoch')\n"
        "plt.ylabel('Average NLL Loss')\n"
        "plt.title('Q2 Four-gram LM Loss')\n"
        "plt.legend()\n"
        "plt.show()\n\n"
        "display(pd.DataFrame({str(k): v for k, v in q2_losses.items()}))"
    ))

    cells.append(md_cell(
        "**Q2 conclusion**\n"
        "- n-gram limitation: it only memorizes observed local patterns. Even known words can form unseen sequences with weak probability estimates.\n"
        "- Markov limitation: long-range relations are truncated. A classic failure is long-distance agreement where the true trigger word appears far before the prediction point.\n"
        "- In this tiny sentence-level experiment, all three embedding settings (32/64/128) converge with close loss curves, so extra capacity gives limited gain under very small data."
    ))

    cells.append(md_cell(
        "## Question 3 (45 marks)\n\n"
        "The following cell includes full preprocessing + SGNS + GloVe implementations.\n"
        "Execution logic: **read cached results first**, train only if cache is missing.\n\n"
        "Concrete setup used in the code:\n"
        "- Tokenization: regex-based, case preserved (so `Australia` and `YMCA` stay queryable).\n"
        "- Vocabulary: `min_count = 5` with `[unk]` fallback.\n"
        "- SGNS: negative sampling with unigram^0.75, window size 5.\n"
        "- GloVe: weighted least squares with `x_max = 100`, `alpha = 0.75`, AdaGrad-style updates.\n"
        "- Hyperparameter study: embedding dimensions = 50, 100, 200."
    ))

    cells.append(code_cell(
        "def tokenize_line(line):\n"
        "    return re.findall(r\"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?\", line)\n\n"
        "def read_corpus(path, max_lines = -1):\n"
        "    tokenized = []\n"
        "    with open(path, 'r', encoding = 'utf-8', errors = 'ignore') as f:\n"
        "        for idx, line in enumerate(f):\n"
        "            if max_lines > 0 and idx >= max_lines:\n"
        "                break\n"
        "            tokens = tokenize_line(line.strip())\n"
        "            if tokens:\n"
        "                tokenized.append(tokens)\n"
        "    return tokenized\n\n"
        "def build_vocab(tokenized_corpus, min_count = 5):\n"
        "    counter = Counter()\n"
        "    for sent in tokenized_corpus:\n"
        "        counter.update(sent)\n"
        "    items = [(w, c) for w, c in counter.items() if c >= min_count]\n"
        "    items.sort(key = lambda x: (-x[1], x[0]))\n"
        "    idx_to_token = ['[unk]'] + [w for w, _ in items]\n"
        "    token_to_idx = {w: i for i, w in enumerate(idx_to_token)}\n"
        "    counts = [1] + [counter[w] for w, _ in items]\n"
        "    return token_to_idx, idx_to_token, counts\n\n"
        "def corpus_to_ids(tokenized_corpus, token_to_idx):\n"
        "    unk = token_to_idx['[unk]']\n"
        "    return [[token_to_idx.get(t, unk) for t in sent] for sent in tokenized_corpus]\n\n"
        "def sgns_build_pairs(corpus_ids, window_size = 5, max_pairs = 100000):\n"
        "    pairs = []\n"
        "    for sent in corpus_ids:\n"
        "        n = len(sent)\n"
        "        for i, center in enumerate(sent):\n"
        "            left = max(0, i - window_size)\n"
        "            right = min(n, i + window_size + 1)\n"
        "            for j in range(left, right):\n"
        "                if i == j:\n"
        "                    continue\n"
        "                pairs.append((center, sent[j]))\n"
        "                if len(pairs) >= max_pairs:\n"
        "                    return pairs\n"
        "    return pairs\n\n"
        "def sgns_train_numpy(corpus_ids, counts, vocab_size, embedding_dim = 100, window_size = 5, negative_samples = 5, epochs = 3, lr = 0.02, max_pairs = 100000, seed = 42):\n"
        "    rng = np.random.default_rng(seed)\n"
        "    pairs = sgns_build_pairs(corpus_ids, window_size = window_size, max_pairs = max_pairs)\n"
        "    random.shuffle(pairs)\n"
        "    dist = np.array(counts, dtype = np.float64) ** 0.75\n"
        "    dist /= dist.sum()\n"
        "    cdf = np.cumsum(dist)\n\n"
        "    scale = 0.5 / embedding_dim\n"
        "    W_in = rng.uniform(-scale, scale, size = (vocab_size, embedding_dim)).astype(np.float32)\n"
        "    W_out = np.zeros((vocab_size, embedding_dim), dtype = np.float32)\n"
        "    losses = []\n\n"
        "    def sigmoid(x):\n"
        "        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))\n\n"
        "    for _ in tqdm(range(epochs), desc = 'SGNS epochs', leave = False):\n"
        "        random.shuffle(pairs)\n"
        "        epoch_loss = 0.0\n"
        "        for center, positive in pairs:\n"
        "            v_c = W_in[center]\n"
        "            v_p = W_out[positive]\n"
        "            pos_score = np.dot(v_c, v_p)\n"
        "            pos_sig = sigmoid(pos_score)\n"
        "            pos_grad = pos_sig - 1.0\n"
        "            grad_center = pos_grad * v_p\n"
        "            grad_pos = pos_grad * v_c\n"
        "            loss = -np.log(pos_sig + 1e-12)\n\n"
        "            neg_ids = np.searchsorted(cdf, np.random.random(negative_samples))\n"
        "            for neg in neg_ids:\n"
        "                v_n = W_out[neg]\n"
        "                neg_score = np.dot(v_c, v_n)\n"
        "                neg_sig = sigmoid(neg_score)\n"
        "                grad_center += neg_sig * v_n\n"
        "                grad_neg = neg_sig * v_c\n"
        "                W_out[neg] -= lr * grad_neg\n"
        "                loss += -np.log(1.0 - neg_sig + 1e-12)\n\n"
        "            W_in[center] -= lr * grad_center\n"
        "            W_out[positive] -= lr * grad_pos\n"
        "            epoch_loss += float(loss)\n\n"
        "        losses.append(epoch_loss / max(len(pairs), 1))\n\n"
        "    embeddings = W_in + W_out\n"
        "    return {'losses': losses, 'embeddings': embeddings}\n\n"
        "def glove_build_cooccurrence(corpus_ids, window_size = 5, max_cooc = 250000):\n"
        "    cooc = defaultdict(float)\n"
        "    for sent in corpus_ids:\n"
        "        n = len(sent)\n"
        "        for i, center in enumerate(sent):\n"
        "            left = max(0, i - window_size)\n"
        "            right = min(n, i + window_size + 1)\n"
        "            for j in range(left, right):\n"
        "                if i == j:\n"
        "                    continue\n"
        "                d = abs(i - j)\n"
        "                cooc[(center, sent[j])] += 1.0 / d\n"
        "        if len(cooc) >= max_cooc:\n"
        "            break\n"
        "    return dict(cooc)\n\n"
        "def glove_train_numpy(cooc, vocab_size, embedding_dim = 100, epochs = 10, lr = 0.05, x_max = 100.0, alpha = 0.75, seed = 42):\n"
        "    rng = np.random.default_rng(seed)\n"
        "    W = (rng.random((vocab_size, embedding_dim)) - 0.5) / embedding_dim\n"
        "    WT = (rng.random((vocab_size, embedding_dim)) - 0.5) / embedding_dim\n"
        "    b = np.zeros(vocab_size)\n"
        "    bt = np.zeros(vocab_size)\n"
        "    gW = np.ones_like(W)\n"
        "    gWT = np.ones_like(WT)\n"
        "    gb = np.ones_like(b)\n"
        "    gbt = np.ones_like(bt)\n"
        "    items = list(cooc.items())\n"
        "    losses = []\n\n"
        "    def weight_fn(x):\n"
        "        if x < x_max:\n"
        "            return (x / x_max) ** alpha\n"
        "        return 1.0\n\n"
        "    for _ in tqdm(range(epochs), desc = 'GloVe epochs', leave = False):\n"
        "        random.shuffle(items)\n"
        "        total = 0.0\n"
        "        for (i, j), xij in items:\n"
        "            w = weight_fn(xij)\n"
        "            diff = (W[i] @ WT[j] + b[i] + bt[j] - np.log(max(xij, 1e-10)))\n"
        "            fd = w * diff\n"
        "            total += 0.5 * fd * diff\n"
        "            grad_wi = fd * WT[j]\n"
        "            grad_wj = fd * W[i]\n"
        "            grad_bi = fd\n"
        "            grad_bj = fd\n"
        "            W[i] -= (lr / np.sqrt(gW[i])) * grad_wi\n"
        "            WT[j] -= (lr / np.sqrt(gWT[j])) * grad_wj\n"
        "            b[i] -= (lr / np.sqrt(gb[i])) * grad_bi\n"
        "            bt[j] -= (lr / np.sqrt(gbt[j])) * grad_bj\n"
        "            gW[i] += grad_wi ** 2\n"
        "            gWT[j] += grad_wj ** 2\n"
        "            gb[i] += grad_bi ** 2\n"
        "            gbt[j] += grad_bj ** 2\n"
        "        losses.append(total / max(len(items), 1))\n\n"
        "    embeddings = W + WT\n"
        "    return {'losses': losses, 'embeddings': embeddings}\n\n"
        "def nearest_neighbors(word, k, token_to_idx, idx_to_token, embeddings):\n"
        "    if word not in token_to_idx:\n"
        "        return []\n"
        "    wid = token_to_idx[word]\n"
        "    vec = embeddings[wid]\n"
        "    norms = np.linalg.norm(embeddings, axis = 1) + 1e-12\n"
        "    sims = embeddings @ vec / (norms * (np.linalg.norm(vec) + 1e-12))\n"
        "    sims[wid] = -1.0\n"
        "    top = np.argsort(-sims)[:k]\n"
        "    return [(idx_to_token[i], float(sims[i])) for i in top]"
    ))

    cells.append(code_cell(
        "q3_path = RESULTS_DIR / 'q3_summary.json'\n"
        "q3_loss_table_path = RESULTS_DIR / 'q3_loss_table.csv'\n"
        "if q3_path.exists():\n"
        "    q3_summary = json.loads(q3_path.read_text(encoding = 'utf-8'))\n"
        "    print('Loaded cached Q3 results from', q3_path)\n"
        "else:\n"
        "    corpus_path = Path('Assignments/assignment01/data/wiki_corpus.txt')\n"
        "    tokenized = read_corpus(str(corpus_path), max_lines = 7000)\n"
        "    token_to_idx, idx_to_token, counts = build_vocab(tokenized, min_count = 5)\n"
        "    corpus_ids = corpus_to_ids(tokenized, token_to_idx)\n"
        "    cooc = glove_build_cooccurrence(corpus_ids, window_size = 5, max_cooc = 250000)\n"
        "    query_words = ['Australia', 'YMCA', 'South', 'building']\n"
        "    dims = [50, 100, 200]\n"
        "    q3_summary = {\n"
        "        'num_sentences': len(tokenized),\n"
        "        'vocab_size': len(idx_to_token),\n"
        "        'cooc_size': len(cooc),\n"
        "        'dims': {}\n"
        "    }\n"
        "    rows = []\n"
        "    for dim in tqdm(dims, desc = 'Q3 dimensions'):\n"
        "        sgns_out = sgns_train_numpy(corpus_ids, counts, len(idx_to_token), embedding_dim = dim, window_size = 5, negative_samples = 5, epochs = 3, lr = 0.02, max_pairs = 100000, seed = 42)\n"
        "        glove_out = glove_train_numpy(cooc, len(idx_to_token), embedding_dim = dim, epochs = 10, lr = 0.05, x_max = 100.0, alpha = 0.75, seed = 42)\n"
        "        q3_summary['dims'][str(dim)] = {\n"
        "            'sgns_losses': sgns_out['losses'],\n"
        "            'glove_losses': glove_out['losses'],\n"
        "            'neighbors': {}\n"
        "        }\n"
        "        for w in tqdm(query_words, desc = f'neighbors@d{dim}', leave = False):\n"
        "            s_nb = nearest_neighbors(w, 8, token_to_idx, idx_to_token, sgns_out['embeddings'])\n"
        "            g_nb = nearest_neighbors(w, 8, token_to_idx, idx_to_token, glove_out['embeddings'])\n"
        "            q3_summary['dims'][str(dim)]['neighbors'][w] = {'sgns': s_nb, 'glove': g_nb}\n"
        "        for ep, value in enumerate(sgns_out['losses'], start = 1):\n"
        "            rows.append({'model': 'sgns', 'dim': dim, 'epoch': ep, 'loss': value})\n"
        "        for ep, value in enumerate(glove_out['losses'], start = 1):\n"
        "            rows.append({'model': 'glove', 'dim': dim, 'epoch': ep, 'loss': value})\n"
        "    q3_path.write_text(json.dumps(q3_summary, ensure_ascii = False, indent = 2), encoding = 'utf-8')\n"
        "    pd.DataFrame(rows).to_csv(q3_loss_table_path, index = False)\n"
        "    print('Trained Q3 and saved results to', q3_path)\n\n"
        "dim = '100'\n"
        "neighbor_rows = []\n"
        "for w, item in q3_summary['dims'][dim]['neighbors'].items():\n"
        "    neighbor_rows.append({'word': w, 'model': 'sgns', 'top5': ', '.join([x[0] for x in item['sgns'][:5]])})\n"
        "    neighbor_rows.append({'word': w, 'model': 'glove', 'top5': ', '.join([x[0] for x in item['glove'][:5]])})\n"
        "display(pd.DataFrame(neighbor_rows))\n\n"
        "plt.figure(figsize = (10, 5))\n"
        "for model in ['sgns', 'glove']:\n"
        "    for dim in ['50', '100', '200']:\n"
        "        ys = q3_summary['dims'][dim][f'{model}_losses']\n"
        "        xs = list(range(1, len(ys) + 1))\n"
        "        plt.plot(xs, ys, marker = 'o', label = f'{model}-d{dim}')\n"
        "plt.xlabel('Epoch')\n"
        "plt.ylabel('Loss')\n"
        "plt.title('Q3 Training Loss')\n"
        "plt.legend(ncol = 2)\n"
        "plt.show()"
    ))

    cells.append(md_cell(
        "**Q3 conclusion**\n"
        "- Both SGNS and GloVe reduce training loss, which indicates they capture corpus-level distributional signals.\n"
        "- Neighbor inspection for `Australia`, `YMCA`, `South`, and `building` shows partial semantic relevance, but also frequent high-frequency or topical noise.\n"
        "- Increasing dimension from 50 to 200 improves optimization fit, but nearest-neighbor quality does not improve proportionally, which matches the usual capacity-vs-noise tradeoff."
    ))

    cells.append(md_cell(
        "## Question 4 (20 marks)\n\n"
        "Space tokenization is problematic for contractions, URLs/emails, and numbers with units.\n"
        "- `don't` mixes negation with verb form.\n"
        "- `user@email.com` contains separators that are part of the token itself.\n"
        "- `3.14km` packs numeric value and unit into one surface form.\n"
        "The next cells implement BPE and then apply load-or-train behavior for results."
    ))

    cells.append(code_cell(
        "class BPETokenizer:\n"
        "    def __init__(self):\n"
        "        self.base_vocab = set()\n"
        "        self.vocab = set()\n"
        "        self.merges = []\n\n"
        "    def _pair_counts(self, word_symbols_freq):\n"
        "        pair_counts = defaultdict(int)\n"
        "        for symbols, freq in word_symbols_freq:\n"
        "            for i in range(len(symbols) - 1):\n"
        "                pair_counts[(symbols[i], symbols[i + 1])] += freq\n"
        "        return pair_counts\n\n"
        "    def _merge_once(self, symbols, pair):\n"
        "        merged = []\n"
        "        i = 0\n"
        "        while i < len(symbols):\n"
        "            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:\n"
        "                merged.append(symbols[i] + symbols[i + 1])\n"
        "                i += 2\n"
        "            else:\n"
        "                merged.append(symbols[i])\n"
        "                i += 1\n"
        "        return merged\n\n"
        "    def fit(self, word_freq_dict, vocab_size = 16):\n"
        "        self.base_vocab = set()\n"
        "        for w in word_freq_dict:\n"
        "            self.base_vocab.update(list(w))\n"
        "        self.vocab = set(self.base_vocab)\n"
        "        self.merges = []\n"
        "        ws = [(list(w), f) for w, f in word_freq_dict.items()]\n"
        "        history = []\n"
        "        while len(self.vocab) < vocab_size:\n"
        "            pair_counts = self._pair_counts(ws)\n"
        "            if not pair_counts:\n"
        "                break\n"
        "            pair, cnt = max(pair_counts.items(), key = lambda x: x[1])\n"
        "            new_token = pair[0] + pair[1]\n"
        "            self.merges.append(pair)\n"
        "            self.vocab.add(new_token)\n"
        "            ws = [(self._merge_once(s, pair), f) for s, f in ws]\n"
        "            history.append({'merge': f'{pair[0]} + {pair[1]} -> {new_token}', 'count': cnt, 'vocab_size': len(self.vocab)})\n"
        "        return history\n\n"
        "    def tokenize_word(self, word):\n"
        "        if any(ch not in self.base_vocab for ch in word):\n"
        "            return ['[unk]']\n"
        "        symbols = list(word)\n"
        "        for pair in self.merges:\n"
        "            symbols = self._merge_once(symbols, pair)\n"
        "        return symbols"
    ))

    cells.append(code_cell(
        "q4_path = RESULTS_DIR / 'q4_bpe_results.json'\n"
        "if q4_path.exists():\n"
        "    q4_results = json.loads(q4_path.read_text(encoding = 'utf-8'))\n"
        "    print('Loaded cached Q4 results from', q4_path)\n"
        "else:\n"
        "    wf = {'old': 10, 'older': 5, 'oldest': 8, 'hug': 8, 'pug': 4, 'hugs': 5}\n"
        "    bpe = BPETokenizer()\n"
        "    history = bpe.fit(wf, vocab_size = 16)\n"
        "    targets = ['hold', 'oldest', 'older', 'pug', 'mug', 'huggingface']\n"
        "    tokens = {w: bpe.tokenize_word(w) for w in targets}\n"
        "    q4_results = {\n"
        "        'base_vocab_size': len(bpe.base_vocab),\n"
        "        'target_vocab_size': 16,\n"
        "        'final_vocab_size': len(bpe.vocab),\n"
        "        'merge_history': history,\n"
        "        'tokens': tokens\n"
        "    }\n"
        "    q4_path.write_text(json.dumps(q4_results, ensure_ascii = False, indent = 2), encoding = 'utf-8')\n"
        "    print('Trained Q4 BPE and saved results to', q4_path)\n\n"
        "display(pd.DataFrame(q4_results['merge_history']))\n"
        "display(pd.DataFrame([{'word': k, 'tokens': ' '.join(v)} for k, v in q4_results['tokens'].items()]))"
    ))

    cells.append(md_cell(
        "`mug` and `huggingface` are `[unk]` in this toy setup because they include characters outside the base vocabulary.\n\n"
        "## Final Note\n"
        "This notebook keeps full training code for reproducibility, but defaults to loading cached outputs from `results` when those files already exist."
    ))

    nb["cells"] = cells
    return nb


def main():
    """Build and save all-in-one notebook.

    Parameters:
        None.

    Returns:
        None
    """
    nb = build_notebook()
    OUTPUT_PATH.parent.mkdir(parents = True, exist_ok = True)
    nbf.write(nb, OUTPUT_PATH)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
