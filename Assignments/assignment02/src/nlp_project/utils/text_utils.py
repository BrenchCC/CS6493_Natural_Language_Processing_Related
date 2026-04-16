import math
from collections import Counter


def safe_log(x, eps = 1e-12):
    """Compute safe natural logarithm.

    Parameters:
        x (float): Input value.
        eps (float): Stability epsilon.

    Returns:
        float: Log value.
    """
    return math.log(max(x, eps))


def normalized_tokens(text):
    """Normalize a sentence into lowercase tokens.

    Parameters:
        text (str): Input sentence.

    Returns:
        list[str]: Tokens after punctuation removal.
    """
    punct = ",.!?;:\"'()[]{}"
    cleaned = text.lower()
    for ch in punct:
        cleaned = cleaned.replace(ch, "")
    return [token for token in cleaned.split() if len(token) > 0]


def ngram_counts(tokens, n):
    """Build n-gram frequency counts.

    Parameters:
        tokens (list[str]): Input token list.
        n (int): N-gram order.

    Returns:
        Counter: N-gram frequency counter.
    """
    if len(tokens) < n:
        return Counter()

    ngrams = [tuple(tokens[idx: idx + n]) for idx in range(0, len(tokens) - n + 1)]
    return Counter(ngrams)


def clipped_precision(candidate_tokens, reference_tokens_list, n):
    """Compute clipped n-gram precision.

    Parameters:
        candidate_tokens (list[str]): Candidate token list.
        reference_tokens_list (list[list[str]]): Reference token lists.
        n (int): N-gram order.

    Returns:
        dict: Precision payload including clipping details.
    """
    candidate_counts = ngram_counts(candidate_tokens, n)
    denominator = max(sum(candidate_counts.values()), 1)

    clipped_matches = 0
    match_details = []

    for ngram, cand_count in candidate_counts.items():
        reference_max_count = 0

        for reference_tokens in reference_tokens_list:
            ref_count = ngram_counts(reference_tokens, n).get(ngram, 0)
            reference_max_count = max(reference_max_count, ref_count)

        clip_count = min(cand_count, reference_max_count)
        clipped_matches += clip_count

        match_details.append(
            {
                "ngram": list(ngram),
                "candidate_count": cand_count,
                "reference_max_count": reference_max_count,
                "clipped_count": clip_count
            }
        )

    precision = clipped_matches / denominator

    return {
        "order": n,
        "clipped_matches": clipped_matches,
        "total_candidate_ngrams": denominator,
        "precision": precision,
        "details": match_details
    }
