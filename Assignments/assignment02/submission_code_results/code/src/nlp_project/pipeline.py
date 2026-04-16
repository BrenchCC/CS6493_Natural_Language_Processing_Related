import os
import json
import logging

from tqdm import tqdm

from nlp_project.solvers.q1_solver import Q1Solver
from nlp_project.solvers.q2_solver import Q2Solver
from nlp_project.exporters.latex_exporter import export_latex_macros
from nlp_project.utils.random_utils import set_seed


logger = logging.getLogger(__name__)


class ProjectPipeline:
    """Pipeline orchestration layer for NLP Project."""

    def __init__(self, output_dir, generated_tex_path, seed = 42, disable_latex_export = False):
        """Initialize pipeline configuration.

        Parameters:
            output_dir (str): Directory for JSON outputs.
            generated_tex_path (str): Path for generated LaTeX macros.
            seed (int): Random seed.
            disable_latex_export (bool): Whether to skip LaTeX export.
        """
        self.output_dir = output_dir
        self.generated_tex_path = generated_tex_path
        self.seed = seed
        self.disable_latex_export = disable_latex_export

    def _write_json(self, path, payload):
        """Write payload as formatted JSON.

        Parameters:
            path (str): Output file path.
            payload (dict): JSON payload.

        Returns:
            None.
        """
        with open(path, "w", encoding = "utf-8") as file_obj:
            json.dump(payload, file_obj, ensure_ascii = False, indent = 2)

    def run(self):
        """Run full project pipeline.

        Parameters:
            None.

        Returns:
            dict: Aggregated payload.
        """
        logger.info("=" * 80)
        logger.info("CS6493 NLP Project: Running layered pipeline")
        logger.info("=" * 80)

        set_seed(self.seed)
        os.makedirs(self.output_dir, exist_ok = True)

        q1_solver = Q1Solver()
        q2_solver = Q2Solver(seed = self.seed)

        task_specs = [
            ("q1", q1_solver.solve_all),
            ("q2", q2_solver.solve_all)
        ]

        task_outputs = {}

        for task_name, task_fn in tqdm(task_specs, desc = "Running project tasks"):
            logger.info("-" * 60)
            logger.info("Running task group: %s", task_name)
            logger.info("-" * 60)
            task_outputs[task_name] = task_fn()

        payload = {
            "q1": task_outputs["q1"],
            "q2": task_outputs["q2"]
        }

        q1_path = os.path.join(self.output_dir, "q1_results.json")
        q2_path = os.path.join(self.output_dir, "q2_results.json")
        summary_path = os.path.join(self.output_dir, "run_summary.json")

        self._write_json(q1_path, payload["q1"])
        self._write_json(q2_path, payload["q2"])

        summary_payload = {
            "q1": {
                "max_length": payload["q1"]["padding_and_indexing"]["max_length"],
                "greedy_probability": payload["q1"]["decoding"]["greedy"]["probability"],
                "beam_probability": payload["q1"]["decoding"]["beam"]["best_probability"],
                "bleu": payload["q1"]["bleu"]["bleu"]
            },
            "q2": {
                "symmetry_max_abs_diff": payload["q2"]["constraint_analysis"]["symmetry_max_abs_diff"],
                "argmax_index": payload["q2"]["beta_behavior"]["argmax_index"],
                "mask_n": payload["q2"]["causal_mask"]["n"]
            }
        }

        self._write_json(summary_path, summary_payload)

        if not self.disable_latex_export:
            export_latex_macros(payload, self.generated_tex_path)

        logger.info("*" * 50)
        logger.info("All tasks completed. Summary:")
        logger.info(json.dumps(summary_payload, ensure_ascii = False, indent = 2))
        logger.info("*" * 50)

        return payload
