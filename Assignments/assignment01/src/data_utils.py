import argparse
import re
from collections import Counter


def parse_args():
    """Parse command line arguments for corpus utilities.

    Parameters:
        None.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description = "Corpus utility arguments")
    parser.add_argument("--corpus_path", type = str, required = True, help = "Path to corpus text file")
    parser.add_argument("--min_count", type = int, default = 5, help = "Minimum frequency threshold")
    parser.add_argument("--max_lines", type = int, default = -1, help = "Maximum lines to read (-1 means all)")
    return parser.parse_args()


def tokenize_line(line):
    """Tokenize one line with a light regex while keeping case.

    Parameters:
        line (str): Raw input text line.

    Returns:
        list[str]: Tokenized words.
    """
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", line)


def read_corpus(path, max_lines = -1):
    """Read and tokenize corpus file.

    Parameters:
        path (str): Input corpus file path.
        max_lines (int): Maximum number of lines to read. -1 means all lines.

    Returns:
        list[list[str]]: Tokenized corpus by lines.
    """
    tokenized = []
    with open(path, "r", encoding = "utf-8", errors = "ignore") as f:
        for idx, line in enumerate(f):
            if max_lines > 0 and idx >= max_lines:
                break
            tokens = tokenize_line(line.strip())
            if tokens:
                tokenized.append(tokens)
    return tokenized


def build_vocab(tokenized_corpus, min_count = 5):
    """Build vocabulary and counts from tokenized corpus.

    Parameters:
        tokenized_corpus (list[list[str]]): Tokenized corpus.
        min_count (int): Frequency threshold for keeping a token.

    Returns:
        tuple[dict[str, int], list[str], list[int]]: token_to_idx, idx_to_token, token_counts.
    """
    counter = Counter()
    for sent in tokenized_corpus:
        counter.update(sent)

    items = [(w, c) for w, c in counter.items() if c >= min_count]
    items.sort(key = lambda x: (-x[1], x[0]))

    idx_to_token = ["[unk]"] + [w for w, _ in items]
    token_to_idx = {w: i for i, w in enumerate(idx_to_token)}

    token_counts = [1]
    for w, _ in items:
        token_counts.append(counter[w])

    return token_to_idx, idx_to_token, token_counts


def corpus_to_ids(tokenized_corpus, token_to_idx):
    """Convert tokenized corpus to token id sequences.

    Parameters:
        tokenized_corpus (list[list[str]]): Tokenized corpus.
        token_to_idx (dict[str, int]): Vocabulary mapping.

    Returns:
        list[list[int]]: Corpus as token id sequences.
    """
    unk_id = token_to_idx["[unk]"]
    result = []
    for sent in tokenized_corpus:
        result.append([token_to_idx.get(t, unk_id) for t in sent])
    return result
