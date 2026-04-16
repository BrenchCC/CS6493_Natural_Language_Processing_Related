import math
from copy import deepcopy

from tqdm import tqdm

from nlp_project.constants import Q1_SENTENCES, Q1_VOCAB, Q1_BLEU_CANDIDATE, Q1_BLEU_REFERENCES, Q1_DECODE_TRANSITIONS
from nlp_project.utils.text_utils import clipped_precision, normalized_tokens, safe_log


class Q1Solver:
    """Solver for Question 1."""

    def solve_padding_and_indexing(self):
        """Solve Q1.1 padding and indexing.

        Parameters:
            None.

        Returns:
            dict: Tokenized, indexed, and padded outputs.
        """
        tokenized = [sentence.split() for sentence in Q1_SENTENCES]
        indexed = [[Q1_VOCAB[token] for token in tokens] for tokens in tokenized]

        max_len = max(len(sequence) for sequence in indexed)
        padded = [sequence + [0] * (max_len - len(sequence)) for sequence in indexed]

        return {
            "sentences": Q1_SENTENCES,
            "tokenized": tokenized,
            "indexed": indexed,
            "padded": padded,
            "max_length": max_len,
            "pad_token": 0
        }

    def _decode_transitions(self, prefix_tokens):
        """Fetch next-step transitions for one prefix.

        Parameters:
            prefix_tokens (list[str]): Current prefix.

        Returns:
            list[tuple[str, float]]: Candidate token and probability.
        """
        if len(prefix_tokens) > 0 and prefix_tokens[-1] == "</s>":
            return []

        state = tuple(prefix_tokens)
        return Q1_DECODE_TRANSITIONS.get(state, [])

    def solve_decoding(self):
        """Solve Q1.3 greedy and beam decoding.

        Parameters:
            None.

        Returns:
            dict: Greedy and beam-search outputs.
        """
        greedy_tokens = []
        greedy_probability = 1.0
        greedy_steps = []

        while True:
            candidates = self._decode_transitions(greedy_tokens)
            if len(candidates) == 0:
                break

            best_token, best_prob = max(candidates, key = lambda item: item[1])
            greedy_tokens.append(best_token)
            greedy_probability *= best_prob

            greedy_steps.append(
                {
                    "prefix_after_step": deepcopy(greedy_tokens),
                    "picked_token": best_token,
                    "picked_probability": best_prob,
                    "path_probability": greedy_probability
                }
            )

            if best_token == "</s>":
                break

        beam_width = 2
        beams = [{"tokens": [], "probability": 1.0, "ended": False}]
        beam_steps = []

        for _ in range(3):
            expanded = []

            for beam in beams:
                if beam["ended"]:
                    expanded.append(beam)
                    continue

                for token, prob in self._decode_transitions(beam["tokens"]):
                    candidate_tokens = beam["tokens"] + [token]
                    candidate_prob = beam["probability"] * prob
                    expanded.append(
                        {
                            "tokens": candidate_tokens,
                            "probability": candidate_prob,
                            "ended": token == "</s>"
                        }
                    )

            expanded = sorted(expanded, key = lambda item: item["probability"], reverse = True)
            beams = expanded[:beam_width]

            beam_steps.append(
                {
                    "all_candidates": [
                        {
                            "tokens": item["tokens"],
                            "probability": item["probability"]
                        }
                        for item in expanded
                    ],
                    "kept_beams": [
                        {
                            "tokens": item["tokens"],
                            "probability": item["probability"]
                        }
                        for item in beams
                    ]
                }
            )

            if all(item["ended"] for item in beams):
                break

        best_beam = max(beams, key = lambda item: item["probability"])

        return {
            "greedy": {
                "sequence": greedy_tokens,
                "probability": greedy_probability,
                "steps": greedy_steps
            },
            "beam": {
                "beam_width": beam_width,
                "best_sequence": best_beam["tokens"],
                "best_probability": best_beam["probability"],
                "steps": beam_steps
            }
        }

    def solve_bleu(self):
        """Solve Q1.4 BLEU with N = 4.

        Parameters:
            None.

        Returns:
            dict: BLEU components and final score.
        """
        candidate_tokens = normalized_tokens(Q1_BLEU_CANDIDATE)
        reference_tokens = [normalized_tokens(sentence) for sentence in Q1_BLEU_REFERENCES]

        precision_payload = []

        for order in tqdm(range(1, 5), desc = "Computing BLEU n-gram precisions", leave = False):
            precision_payload.append(clipped_precision(candidate_tokens, reference_tokens, order))

        candidate_length = len(candidate_tokens)
        reference_lengths = [len(tokens) for tokens in reference_tokens]
        closest_reference_length = min(
            reference_lengths,
            key = lambda value: (abs(value - candidate_length), value)
        )

        if candidate_length > closest_reference_length:
            brevity_penalty = 1.0
        else:
            brevity_penalty = math.exp(1.0 - closest_reference_length / max(candidate_length, 1))

        geometric_mean = math.exp(
            sum(safe_log(item["precision"]) for item in precision_payload) / 4.0
        )
        bleu = brevity_penalty * geometric_mean

        return {
            "candidate": Q1_BLEU_CANDIDATE,
            "references": Q1_BLEU_REFERENCES,
            "candidate_tokens": candidate_tokens,
            "reference_tokens": reference_tokens,
            "precisions": precision_payload,
            "candidate_length": candidate_length,
            "reference_lengths": reference_lengths,
            "closest_reference_length": closest_reference_length,
            "brevity_penalty": brevity_penalty,
            "geometric_mean": geometric_mean,
            "bleu": bleu
        }

    def solve_all(self):
        """Run all Q1 sub-problems.

        Parameters:
            None.

        Returns:
            dict: Aggregated Q1 outputs.
        """
        return {
            "padding_and_indexing": self.solve_padding_and_indexing(),
            "decoding": self.solve_decoding(),
            "bleu": self.solve_bleu()
        }
