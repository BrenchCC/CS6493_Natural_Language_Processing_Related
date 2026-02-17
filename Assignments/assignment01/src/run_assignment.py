import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bpe_tokenizer import BPETokenizer, default_word_freq
from data_utils import build_vocab, corpus_to_ids, read_corpus
from glove import build_cooccurrence, nearest_neighbors as glove_neighbors, train_glove
from sgns import nearest_neighbors as sgns_neighbors, train_sgns


def parse_args():
    """Parse command line arguments.

    Parameters:
        None.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description = "Run HW1 experiments")
    parser.add_argument("--corpus_path", type = str, default = "Assignments/assignment01/data/wiki_corpus.txt")
    parser.add_argument("--output_dir", type = str, default = "Assignments/assignment01/submission/results")
    parser.add_argument("--min_count", type = int, default = 5)
    parser.add_argument("--max_lines", type = int, default = 8000)
    parser.add_argument("--seed", type = int, default = 42)
    return parser.parse_args()


def set_seed(seed):
    """Set random seeds.

    Parameters:
        seed (int): Seed number.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)


def _softmax(x):
    """Compute stable softmax.

    Parameters:
        x (np.ndarray): Input logits.

    Returns:
        np.ndarray: Softmax values.
    """
    y = x - np.max(x)
    exp_y = np.exp(y)
    return exp_y / np.sum(exp_y)


def run_q2_lm(output_dir, seed):
    """Run Q2 four-gram neural language model experiment in NumPy.

    Parameters:
        output_dir (Path): Output directory.
        seed (int): Random seed.

    Returns:
        dict: Q2 results.
    """
    set_seed(seed)
    sentence = "I am taking CS6493 this semester and studying NLP is really fascinating"
    tokens = sentence.split()

    vocab = sorted(set(tokens))
    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = {i: w for w, i in w2i.items()}

    data = []
    for i in range(3, len(tokens)):
        context = [tokens[i - 3], tokens[i - 2], tokens[i - 1]]
        target = tokens[i]
        data.append((context, target))

    dim_losses = {}

    for dim in [32, 64, 128]:
        rng = np.random.default_rng(seed + dim)
        vocab_size = len(vocab)

        E = (rng.random((vocab_size, dim)) - 0.5) / dim
        W1 = (rng.random((3 * dim, 128)) - 0.5) / np.sqrt(3 * dim)
        b1 = np.zeros(128)
        W2 = (rng.random((128, vocab_size)) - 0.5) / np.sqrt(128)
        b2 = np.zeros(vocab_size)

        lr = 0.08
        losses = []

        for _ in range(10):
            total = 0.0
            for context, target in data:
                ctx_ids = [w2i[w] for w in context]
                t_id = w2i[target]

                x = np.concatenate([E[idx] for idx in ctx_ids])
                h_pre = x @ W1 + b1
                h = np.maximum(h_pre, 0.0)
                logits = h @ W2 + b2
                probs = _softmax(logits)

                loss = -np.log(probs[t_id] + 1e-12)
                total += float(loss)

                dlogits = probs.copy()
                dlogits[t_id] -= 1.0

                dW2 = np.outer(h, dlogits)
                db2 = dlogits

                dh = W2 @ dlogits
                dh_pre = dh * (h_pre > 0)

                dW1 = np.outer(x, dh_pre)
                db1 = dh_pre
                dx = W1 @ dh_pre

                W2 -= lr * dW2
                b2 -= lr * db2
                W1 -= lr * dW1
                b1 -= lr * db1

                for pos, idx in enumerate(ctx_ids):
                    start = pos * dim
                    end = (pos + 1) * dim
                    E[idx] -= lr * dx[start:end]

            losses.append(total / len(data))

        dim_losses[str(dim)] = losses

    q2_payload = {
        "sentence_tokens": tokens,
        "vocab_size": len(vocab),
        "training_examples": data,
        "losses": dim_losses,
        "sample_prediction_context": data[-1][0],
        "sample_prediction_target": data[-1][1],
        "idx_to_word": i2w
    }

    with open(output_dir / "q2_results.json", "w", encoding = "utf-8") as f:
        json.dump(q2_payload, f, ensure_ascii = False, indent = 2)

    plt.figure(figsize = (8, 5))
    for dim in [32, 64, 128]:
        plt.plot(range(1, 11), dim_losses[str(dim)], marker = "o", label = f"dim={dim}")
    plt.xlabel("Epoch")
    plt.ylabel("Average NLL Loss")
    plt.title("Q2 Four-gram LM Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "q2_loss_curve.png", dpi = 160)
    plt.close()

    return q2_payload


def run_q3_embeddings(corpus_path, output_dir, min_count, max_lines, seed):
    """Run Q3 SGNS and GloVe experiments.

    Parameters:
        corpus_path (str): Corpus path.
        output_dir (Path): Output directory.
        min_count (int): Minimum token frequency.
        max_lines (int): Maximum corpus lines used.
        seed (int): Random seed.

    Returns:
        dict: Q3 summary.
    """
    set_seed(seed)

    tokenized = read_corpus(corpus_path, max_lines = max_lines)
    token_to_idx, idx_to_token, counts = build_vocab(tokenized, min_count = min_count)
    corpus_ids = corpus_to_ids(tokenized, token_to_idx)

    cooc = build_cooccurrence(corpus_ids, window_size = 5, max_cooc = 250000)

    query_words = ["Australia", "YMCA", "South", "building"]
    dim_list = [50, 100, 200]

    summary = {
        "num_sentences": len(tokenized),
        "vocab_size": len(idx_to_token),
        "cooc_size": len(cooc),
        "dims": {}
    }

    loss_rows = []

    for dim in dim_list:
        sgns_out = train_sgns(
            corpus_ids,
            counts,
            vocab_size = len(idx_to_token),
            embedding_dim = dim,
            window_size = 5,
            negative_samples = 5,
            epochs = 3,
            learning_rate = 0.02,
            max_pairs = 100000,
            seed = seed,
            device = "cpu"
        )

        glove_out = train_glove(
            cooc,
            vocab_size = len(idx_to_token),
            embedding_dim = dim,
            epochs = 10,
            learning_rate = 0.05,
            x_max = 100.0,
            alpha = 0.75,
            seed = seed
        )

        dim_summary = {
            "sgns_losses": sgns_out["losses"],
            "glove_losses": glove_out["losses"],
            "neighbors": {}
        }

        for word in query_words:
            s_neighbors = sgns_neighbors(word, 8, token_to_idx, idx_to_token, sgns_out["embeddings"])
            g_neighbors = glove_neighbors(word, 8, token_to_idx, idx_to_token, glove_out["embeddings"])
            dim_summary["neighbors"][word] = {
                "sgns": s_neighbors,
                "glove": g_neighbors
            }

        summary["dims"][str(dim)] = dim_summary

        for ep, value in enumerate(sgns_out["losses"], start = 1):
            loss_rows.append(["sgns", dim, ep, value])
        for ep, value in enumerate(glove_out["losses"], start = 1):
            loss_rows.append(["glove", dim, ep, value])

    with open(output_dir / "q3_summary.json", "w", encoding = "utf-8") as f:
        json.dump(summary, f, ensure_ascii = False, indent = 2)

    with open(output_dir / "q3_loss_table.csv", "w", encoding = "utf-8", newline = "") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "dim", "epoch", "loss"])
        writer.writerows(loss_rows)

    plt.figure(figsize = (10, 6))
    for model in ["sgns", "glove"]:
        for dim in dim_list:
            ys = [r[3] for r in loss_rows if r[0] == model and r[1] == dim]
            xs = list(range(1, len(ys) + 1))
            plt.plot(xs, ys, marker = "o", label = f"{model}-d{dim}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Q3 Training Losses")
    plt.legend(ncol = 2)
    plt.tight_layout()
    plt.savefig(output_dir / "q3_loss_curve.png", dpi = 160)
    plt.close()

    return summary


def run_q4_bpe(output_dir):
    """Run Q4 BPE training and tokenization demo.

    Parameters:
        output_dir (Path): Output directory.

    Returns:
        dict: Q4 outputs.
    """
    bpe = BPETokenizer()
    word_freq = default_word_freq()
    history = bpe.fit(word_freq, vocab_size = 16)

    targets = ["hold", "oldest", "older", "pug", "mug", "huggingface"]
    tokenized_targets = bpe.tokenize_words(targets)

    payload = {
        "base_vocab_size": len(bpe.base_vocab),
        "target_vocab_size": 16,
        "final_vocab_size": len(bpe.vocab),
        "merge_history": history,
        "tokens": tokenized_targets
    }

    with open(output_dir / "q4_bpe_results.json", "w", encoding = "utf-8") as f:
        json.dump(payload, f, ensure_ascii = False, indent = 2)

    return payload


def run_q1(output_dir):
    """Create deterministic Q1 answer payload.

    Parameters:
        output_dir (Path): Output directory.

    Returns:
        dict: Q1 outputs.
    """
    vocab = ["cat", "dog", "run", "runs"]
    one_hot = {}
    for i, w in enumerate(vocab):
        vec = [0, 0, 0, 0]
        vec[i] = 1
        one_hot[w] = vec

    payload = {
        "vocab": vocab,
        "one_hot": one_hot,
        "dot_product_distinct_words": 0,
        "params_4x128": 512,
        "params_50000x128": 6400000,
        "practical_issues": [
            "Large memory and slower optimization for huge vocabulary tables.",
            "No parameter sharing for semantically related words."
        ],
        "morphology_conclusion": "One-hot vectors cannot encode run/runs relation automatically."
    }

    with open(output_dir / "q1_results.json", "w", encoding = "utf-8") as f:
        json.dump(payload, f, ensure_ascii = False, indent = 2)

    return payload


def main():
    """Run all assignment experiments and save artifacts.

    Parameters:
        None.

    Returns:
        None
    """
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents = True, exist_ok = True)

    q1 = run_q1(output_dir)
    q2 = run_q2_lm(output_dir, args.seed)
    q3 = run_q3_embeddings(args.corpus_path, output_dir, args.min_count, args.max_lines, args.seed)
    q4 = run_q4_bpe(output_dir)

    final = {
        "q1": q1,
        "q2": {
            "vocab_size": q2["vocab_size"],
            "losses": q2["losses"]
        },
        "q3": {
            "num_sentences": q3["num_sentences"],
            "vocab_size": q3["vocab_size"],
            "cooc_size": q3["cooc_size"]
        },
        "q4": {
            "final_vocab_size": q4["final_vocab_size"],
            "merge_steps": len(q4["merge_history"])
        }
    }

    with open(output_dir / "run_summary.json", "w", encoding = "utf-8") as f:
        json.dump(final, f, ensure_ascii = False, indent = 2)

    print(json.dumps(final, ensure_ascii = False, indent = 2))


if __name__ == "__main__":
    main()
