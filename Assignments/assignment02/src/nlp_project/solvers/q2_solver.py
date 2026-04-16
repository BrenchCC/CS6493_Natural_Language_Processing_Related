import math

import torch

from nlp_project.utils.random_utils import set_seed


class Q2Solver:
    """Solver for Question 2."""

    def __init__(self, seed = 42):
        """Initialize solver.

        Parameters:
            seed (int): Random seed.
        """
        self.seed = seed

    def solve_constraint_analysis(self):
        """Solve Q2.1 with numeric symmetry verification.

        Parameters:
            None.

        Returns:
            dict: Matrix diagnostics under WQ = WK.
        """
        set_seed(self.seed)

        n = 5
        d_model = 6
        d_k = 4

        x = torch.randn(n, d_model)
        w_shared = torch.randn(d_model, d_k)

        q = x @ w_shared
        k = x @ w_shared

        score_matrix = q @ k.T
        symmetry_diff = torch.max(torch.abs(score_matrix - score_matrix.T)).item()

        return {
            "n": n,
            "d_model": d_model,
            "d_k": d_k,
            "symmetry_max_abs_diff": symmetry_diff,
            "score_matrix": score_matrix.tolist()
        }

    def _softmax_with_beta(self, query, keys, beta):
        """Compute attention distribution for one beta value.

        Parameters:
            query (torch.Tensor): Query vector with shape [d_k].
            keys (torch.Tensor): Key matrix with shape [n, d_k].
            beta (float): Scaling coefficient.

        Returns:
            torch.Tensor: Attention probabilities with shape [n].
        """
        logits = beta * torch.mv(keys, query)
        return torch.softmax(logits, dim = 0)

    def solve_beta_behavior(self):
        """Solve Q2.2 by simulating three beta regimes.

        Parameters:
            None.

        Returns:
            dict: Attention distributions and context vectors.
        """
        query = torch.tensor([1.0, -0.5, 0.8], dtype = torch.float64)

        keys = torch.tensor(
            [
                [0.9, -0.4, 0.7],
                [1.2, -0.6, 1.1],
                [0.3, 0.2, -0.1],
                [-0.4, 0.7, -0.8]
            ],
            dtype = torch.float64
        )

        values = torch.tensor(
            [
                [0.2, 1.0],
                [1.3, 0.1],
                [-0.2, 0.6],
                [0.0, -0.5]
            ],
            dtype = torch.float64
        )

        d_k = query.numel()
        beta_values = {
            "beta_to_zero": 1e-4,
            "transformer_beta": 1.0 / math.sqrt(d_k),
            "beta_to_infinity": 1e2
        }

        case_outputs = {}

        for case_name, beta in beta_values.items():
            alpha = self._softmax_with_beta(query, keys, beta)
            context = torch.mv(values.T, alpha)
            case_outputs[case_name] = {
                "beta": beta,
                "alpha": alpha.tolist(),
                "context": context.tolist()
            }

        logits = torch.mv(keys, query)
        argmax_index = int(torch.argmax(logits).item())

        uniform_alpha_limit = [1.0 / keys.size(0)] * keys.size(0)
        average_context_limit = torch.mean(values, dim = 0).tolist()

        return {
            "query": query.tolist(),
            "keys": keys.tolist(),
            "values": values.tolist(),
            "raw_logits": logits.tolist(),
            "argmax_index": argmax_index,
            "uniform_alpha_limit": uniform_alpha_limit,
            "average_context_limit": average_context_limit,
            "cases": case_outputs
        }

    def solve_causal_mask(self, n = 4):
        """Solve Q2.3 by constructing an explicit causal mask.

        Parameters:
            n (int): Sequence length.

        Returns:
            dict: Numeric and symbolic mask matrices.
        """
        mask = torch.zeros((n, n), dtype = torch.float64)
        upper_indices = torch.triu_indices(n, n, offset = 1)
        mask[upper_indices[0], upper_indices[1]] = float("-inf")

        symbol_mask = []

        for row in range(n):
            symbol_row = []
            for col in range(n):
                if col > row:
                    symbol_row.append("-inf")
                else:
                    symbol_row.append("0")
            symbol_mask.append(symbol_row)

        return {
            "n": n,
            "numeric_mask": mask.tolist(),
            "symbol_mask": symbol_mask
        }

    def solve_all(self):
        """Run all Q2 sub-problems.

        Parameters:
            None.

        Returns:
            dict: Aggregated Q2 outputs.
        """
        return {
            "constraint_analysis": self.solve_constraint_analysis(),
            "beta_behavior": self.solve_beta_behavior(),
            "causal_mask": self.solve_causal_mask()
        }
