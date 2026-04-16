import os
import sys
import logging
import argparse

sys.path.append(os.getcwd())

from nlp_project.pipeline import ProjectPipeline


def parse_args():
    """Parse command line arguments.

    Parameters:
        None.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description = "Run NLP project calculations")
    parser.add_argument(
        "--output_dir",
        type = str,
        default = "Assignments/assignment02/results"
    )
    parser.add_argument(
        "--generated_tex_path",
        type = str,
        default = "Assignments/assignment02/reports/generated/results_macros.tex"
    )
    parser.add_argument(
        "--seed",
        type = int,
        default = 42
    )
    parser.add_argument(
        "--disable_latex_export",
        action = "store_true",
        help = "Disable exporting LaTeX macro file."
    )
    return parser.parse_args()


def main():
    """Main entry point.

    Parameters:
        None.

    Returns:
        None.
    """
    args = parse_args()

    pipeline = ProjectPipeline(
        output_dir = args.output_dir,
        generated_tex_path = args.generated_tex_path,
        seed = args.seed,
        disable_latex_export = args.disable_latex_export
    )
    pipeline.run()


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers = [logging.StreamHandler()]
    )
    main()
