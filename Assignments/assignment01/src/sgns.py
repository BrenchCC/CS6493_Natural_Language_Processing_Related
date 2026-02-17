import argparse
import random

import numpy as np
from tqdm import tqdm


def parse_args():
    """Parse command line arguments for SGNS training.

    Parameters:
        None.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description = "Train SGNS model")
    parser.add_argument("--embedding_dim", type = int, default = 100)
    parser.add_argument("--window_size", type = int, default = 5)
    parser.add_argument("--negative_samples", type = int, default = 5)
    parser.add_argument("--epochs", type = int, default = 3)
    parser.add_argument("--learning_rate", type = float, default = 0.025)
    parser.add_argument("--max_pairs", type = int, default = 120000)
    return parser.parse_args()


def _sigmoid(x):
    """Compute stable sigmoid.

    Parameters:
        x (np.ndarray | float): Input value.

    Returns:
        np.ndarray | float: Sigmoid output.
    """
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


class SGNSModel:
    """Skip-gram with negative sampling model implemented in NumPy."""

    def __init__(self, vocab_size, embedding_dim, seed = 42):
        """Initialize SGNS model.

        Parameters:
            vocab_size (int): Vocabulary size.
            embedding_dim (int): Embedding dimension.
            seed (int): Random seed.
        """
        rng = np.random.default_rng(seed)
        scale = 0.5 / embedding_dim
        self.input_embeddings = rng.uniform(-scale, scale, size = (vocab_size, embedding_dim)).astype(np.float32)
        self.output_embeddings = np.zeros((vocab_size, embedding_dim), dtype = np.float32)


def build_training_pairs(corpus_ids, window_size, max_pairs = 120000):
    """Build center-context pairs.

    Parameters:
        corpus_ids (list[list[int]]): Corpus token ids.
        window_size (int): Context window size.
        max_pairs (int): Cap of generated pairs.

    Returns:
        list[tuple[int, int]]: (center, context) pairs.
    """
    pairs = []
    for sent in corpus_ids:
        n = len(sent)
        for i, center in enumerate(sent):
            left = max(0, i - window_size)
            right = min(n, i + window_size + 1)
            for j in range(left, right):
                if j == i:
                    continue
                pairs.append((center, sent[j]))
                if len(pairs) >= max_pairs:
                    return pairs
    return pairs


def _build_negative_cdf(counts, power = 0.75):
    """Build CDF for negative sampling.

    Parameters:
        counts (list[int]): Token counts.
        power (float): Exponent for unigram smoothing.

    Returns:
        np.ndarray: Cumulative distribution.
    """
    dist = np.array(counts, dtype = np.float64) ** power
    dist /= dist.sum()
    return np.cumsum(dist)


def _sample_negative(cdf, n):
    """Sample token ids from CDF.

    Parameters:
        cdf (np.ndarray): CDF array.
        n (int): Number of samples.

    Returns:
        np.ndarray: Sampled ids.
    """
    r = np.random.random(n)
    return np.searchsorted(cdf, r)


def train_sgns(
    corpus_ids,
    counts,
    vocab_size,
    embedding_dim = 100,
    window_size = 5,
    negative_samples = 5,
    epochs = 3,
    learning_rate = 0.025,
    max_pairs = 120000,
    seed = 42,
    device = "cpu"
):
    """Train SGNS and return losses and embeddings.

    Parameters:
        corpus_ids (list[list[int]]): Corpus token ids.
        counts (list[int]): Token counts.
        vocab_size (int): Vocabulary size.
        embedding_dim (int): Embedding dimension.
        window_size (int): Context window size.
        negative_samples (int): Number of negative samples.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate.
        max_pairs (int): Max number of pairs used for training.
        seed (int): Random seed.
        device (str): Reserved for API compatibility.

    Returns:
        dict: Training outputs.
    """
    del device
    random.seed(seed)
    np.random.seed(seed)

    pairs = build_training_pairs(corpus_ids, window_size, max_pairs = max_pairs)
    cdf = _build_negative_cdf(counts)

    model = SGNSModel(vocab_size, embedding_dim, seed = seed)
    W_in = model.input_embeddings
    W_out = model.output_embeddings

    losses = []

    for _ in range(epochs):
        random.shuffle(pairs)
        epoch_loss = 0.0

        for center, positive in tqdm(pairs, leave = False):
            neg_ids = _sample_negative(cdf, negative_samples)

            v_c = W_in[center]
            v_p = W_out[positive]

            pos_score = np.dot(v_c, v_p)
            pos_sig = _sigmoid(pos_score)
            pos_grad = pos_sig - 1.0

            grad_center = pos_grad * v_p
            grad_pos = pos_grad * v_c

            loss = -np.log(pos_sig + 1e-12)

            for neg in neg_ids:
                v_n = W_out[neg]
                neg_score = np.dot(v_c, v_n)
                neg_sig = _sigmoid(neg_score)

                grad_center += neg_sig * v_n
                grad_neg = neg_sig * v_c
                W_out[neg] -= learning_rate * grad_neg

                loss += -np.log(1.0 - neg_sig + 1e-12)

            W_in[center] -= learning_rate * grad_center
            W_out[positive] -= learning_rate * grad_pos

            epoch_loss += float(loss)

        losses.append(epoch_loss / max(len(pairs), 1))

    embeddings = W_in + W_out

    return {
        "losses": losses,
        "embeddings": embeddings,
        "pairs_used": len(pairs)
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
