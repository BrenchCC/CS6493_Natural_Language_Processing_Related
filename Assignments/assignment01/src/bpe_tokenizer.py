import argparse
from collections import Counter, defaultdict


def parse_args():
    """Parse command line arguments for BPE tokenizer.

    Parameters:
        None.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description = "Train and apply BPE tokenizer")
    parser.add_argument("--vocab_size", type = int, default = 16)
    return parser.parse_args()


class BPETokenizer:
    """A simple BPE tokenizer implementation for educational use."""

    def __init__(self):
        """Initialize an empty BPE tokenizer.

        Parameters:
            None.
        """
        self.base_vocab = set()
        self.vocab = set()
        self.merges = []

    def _word_to_symbols(self, word):
        """Split word into base symbols.

        Parameters:
            word (str): Input word.

        Returns:
            list[str]: Character-level symbols.
        """
        return list(word)

    def _get_pair_counts(self, word_symbols_freq):
        """Count adjacent pair frequencies.

        Parameters:
            word_symbols_freq (list[tuple[list[str], int]]): Symbolized words and frequencies.

        Returns:
            dict[tuple[str, str], int]: Pair frequency map.
        """
        pair_counts = defaultdict(int)
        for symbols, freq in word_symbols_freq:
            for i in range(len(symbols) - 1):
                pair_counts[(symbols[i], symbols[i + 1])] += freq
        return pair_counts

    def _merge_pair(self, symbols, pair):
        """Merge one target pair in a symbol sequence.

        Parameters:
            symbols (list[str]): Input symbol sequence.
            pair (tuple[str, str]): Pair to merge.

        Returns:
            list[str]: Merged symbol sequence.
        """
        merged = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                merged.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        return merged

    def fit(self, word_freq_dict, vocab_size):
        """Train BPE merge rules.

        Parameters:
            word_freq_dict (dict[str, int]): Word frequency dictionary.
            vocab_size (int): Target vocabulary size.

        Returns:
            list[dict]: Merge history per iteration.
        """
        self.base_vocab = set()
        for word in word_freq_dict:
            for ch in word:
                self.base_vocab.add(ch)

        self.vocab = set(self.base_vocab)
        self.merges = []

        word_symbols_freq = []
        for w, f in word_freq_dict.items():
            word_symbols_freq.append((self._word_to_symbols(w), f))

        history = []
        while len(self.vocab) < vocab_size:
            pair_counts = self._get_pair_counts(word_symbols_freq)
            if not pair_counts:
                break

            best_pair, best_count = max(pair_counts.items(), key = lambda x: x[1])
            new_token = best_pair[0] + best_pair[1]
            self.merges.append(best_pair)
            self.vocab.add(new_token)

            new_word_symbols_freq = []
            for symbols, freq in word_symbols_freq:
                new_symbols = self._merge_pair(symbols, best_pair)
                new_word_symbols_freq.append((new_symbols, freq))
            word_symbols_freq = new_word_symbols_freq

            history.append({
                "merge": f"{best_pair[0]} + {best_pair[1]} -> {new_token}",
                "count": best_count,
                "vocab_size": len(self.vocab)
            })

            if len(self.vocab) >= vocab_size:
                break

        return history

    def tokenize_word(self, word):
        """Tokenize one word with learned BPE rules.

        Parameters:
            word (str): Input word.

        Returns:
            list[str]: Tokenized subwords.
        """
        for ch in word:
            if ch not in self.base_vocab:
                return ["[unk]"]

        symbols = list(word)
        for pair in self.merges:
            symbols = self._merge_pair(symbols, pair)
        return symbols

    def tokenize_words(self, words):
        """Tokenize multiple words.

        Parameters:
            words (list[str]): Input words.

        Returns:
            dict[str, list[str]]: Mapping from word to tokens.
        """
        return {w: self.tokenize_word(w) for w in words}


def default_word_freq():
    """Return assignment-provided toy word frequencies.

    Parameters:
        None.

    Returns:
        dict[str, int]: Toy word frequency map.
    """
    return {
        "old": 10,
        "older": 5,
        "oldest": 8,
        "hug": 8,
        "pug": 4,
        "hugs": 5
    }


def base_vocab_from_default():
    """Return sorted base vocabulary from assignment toy words.

    Parameters:
        None.

    Returns:
        list[str]: Sorted base vocabulary symbols.
    """
    counter = Counter()
    for w in default_word_freq():
        counter.update(list(w))
    return sorted(counter.keys())
