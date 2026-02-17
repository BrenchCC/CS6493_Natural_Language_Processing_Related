import argparse
import random
from collections import defaultdict

import numpy as np
from tqdm import tqdm


def parse_args():
    """Parse command line arguments for GloVe training.

    Parameters:
        None.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description = "Train GloVe model")
    parser.add_argument("--embedding_dim", type = int, default = 100)
    parser.add_argument("--window_size", type = int, default = 5)
    parser.add_argument("--epochs", type = int, default = 12)
    parser.add_argument("--learning_rate", type = float, default = 0.05)
    parser.add_argument("--x_max", type = float, default = 100.0)
    parser.add_argument("--alpha", type = float, default = 0.75)
    parser.add_argument("--max_cooc", type = int, default = 300000)
    return parser.parse_args()


def build_cooccurrence(corpus_ids, window_size, max_cooc = 300000):
    """Build weighted co-occurrence dictionary.

    Parameters:
        corpus_ids (list[list[int]]): Corpus token ids.
        window_size (int): Context window size.
        max_cooc (int): Cap of co-occurrence entries.

    Returns:
        dict[tuple[int, int], float]: Co-occurrence map.
    """
    cooc = defaultdict(float)
    for sent in corpus_ids:
        n = len(sent)
        for i, center in enumerate(sent):
            left = max(0, i - window_size)
            right = min(n, i + window_size + 1)
            for j in range(left, right):
                if i == j:
                    continue
                context = sent[j]
                distance = abs(i - j)
                if distance == 0:
                    continue
                cooc[(center, context)] += 1.0 / distance
        if len(cooc) >= max_cooc:
            break
    return dict(cooc)


def _weight_fn(x, x_max, alpha):
    """Weighting function in GloVe objective.

    Parameters:
        x (float): Co-occurrence count.
        x_max (float): Saturation threshold.
        alpha (float): Exponent.

    Returns:
        float: Weight.
    """
    if x < x_max:
        return (x / x_max) ** alpha
    return 1.0


def train_glove(
    cooc_dict,
    vocab_size,
    embedding_dim = 100,
    epochs = 12,
    learning_rate = 0.05,
    x_max = 100.0,
    alpha = 0.75,
    seed = 42
):
    """Train GloVe with AdaGrad and return losses and embeddings.

    Parameters:
        cooc_dict (dict[tuple[int, int], float]): Co-occurrence map.
        vocab_size (int): Vocabulary size.
        embedding_dim (int): Embedding dimension.
        epochs (int): Number of epochs.
        learning_rate (float): Learning rate.
        x_max (float): Weighting threshold.
        alpha (float): Weighting exponent.
        seed (int): Random seed.

    Returns:
        dict: Training outputs.
    """
    random.seed(seed)
    np.random.seed(seed)

    W = (np.random.rand(vocab_size, embedding_dim) - 0.5) / embedding_dim
    W_tilde = (np.random.rand(vocab_size, embedding_dim) - 0.5) / embedding_dim
    b = np.zeros(vocab_size)
    b_tilde = np.zeros(vocab_size)

    grad_sq_W = np.ones_like(W)
    grad_sq_W_tilde = np.ones_like(W_tilde)
    grad_sq_b = np.ones_like(b)
    grad_sq_b_tilde = np.ones_like(b_tilde)

    entries = list(cooc_dict.items())
    losses = []

    for _ in range(epochs):
        random.shuffle(entries)
        total_loss = 0.0

        for (i, j), x_ij in tqdm(entries, leave = False):
            weight = _weight_fn(x_ij, x_max, alpha)
            if weight == 0.0:
                continue

            dot = np.dot(W[i], W_tilde[j]) + b[i] + b_tilde[j]
            diff = dot - np.log(max(x_ij, 1e-10))
            fdiff = weight * diff

            total_loss += 0.5 * fdiff * diff

            grad_wi = fdiff * W_tilde[j]
            grad_wj = fdiff * W[i]
            grad_bi = fdiff
            grad_bj = fdiff

            W[i] -= (learning_rate / np.sqrt(grad_sq_W[i])) * grad_wi
            W_tilde[j] -= (learning_rate / np.sqrt(grad_sq_W_tilde[j])) * grad_wj
            b[i] -= (learning_rate / np.sqrt(grad_sq_b[i])) * grad_bi
            b_tilde[j] -= (learning_rate / np.sqrt(grad_sq_b_tilde[j])) * grad_bj

            grad_sq_W[i] += grad_wi ** 2
            grad_sq_W_tilde[j] += grad_wj ** 2
            grad_sq_b[i] += grad_bi ** 2
            grad_sq_b_tilde[j] += grad_bj ** 2

        losses.append(total_loss / max(len(entries), 1))

    embeddings = W + W_tilde
    return {
        "losses": losses,
        "embeddings": embeddings,
        "cooc_size": len(entries)
    }


def nearest_neighbors(word, k, token_to_idx, idx_to_token, embeddings):
    """Find top-k nearest neighbors by cosine similarity.

    Parameters:
        word (str): Query token.
        k (int): Number of neighbors.
        token_to_idx (dict[str, int]): Token to id mapping.
        idx_to_token (list[str]): Id to token mapping.
        embeddings (np.ndarray): Embedding matrix [V, D].

    Returns:
        list[tuple[str, float]]: Neighbor token and similarity score.
    """
    if word not in token_to_idx:
        return []

    w_id = token_to_idx[word]
    vec = embeddings[w_id]

    norms = np.linalg.norm(embeddings, axis = 1) + 1e-12
    sims = embeddings @ vec / (norms * (np.linalg.norm(vec) + 1e-12))
    sims[w_id] = -1.0

    top_ids = np.argsort(-sims)[:k]
    return [(idx_to_token[i], float(sims[i])) for i in top_ids]
