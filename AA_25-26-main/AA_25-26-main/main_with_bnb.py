
import argparse
import os

from parser.file_selector import select_file
from parser.parser import Parser
from serializer.serializer import SolutionSerializer
from utils.utils import Utils

from scheduler.beam_search_scheduler import BeamSearchScheduler
from scheduler.branch_and_bound_scheduler import BranchAndBoundScheduler


def main():
    parser_arg = argparse.ArgumentParser(description="Run TV scheduling algorithms")

    parser_arg.add_argument(
        "--input", "-i",
        dest="input_file",
        help="Path to input JSON. If omitted, the interactive selector is used."
    )
    parser_arg.add_argument(
        "--algorithm", "-a",
        choices=["bnb", "beam"],
        default="bnb",
        help="Scheduler to run."
    )
    parser_arg.add_argument(
        "--time-limit",
        type=float,
        default=30.0,
        help="Time limit in seconds for Branch-and-Bound."
    )
    parser_arg.add_argument(
        "--restarts",
        type=int,
        default=16,
        help="Randomized warm-start restarts for Branch-and-Bound."
    )
    parser_arg.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )

    args = parser_arg.parse_args()

    if args.input_file:
        file_path = args.input_file
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")
    else:
        file_path = select_file()

    parser = Parser(file_path)
    instance = parser.parse()
    Utils.set_current_instance(instance)

    print("\nOpening time:", instance.opening_time)
    print("Closing time:", instance.closing_time)
    print(f"Total Channels: {len(instance.channels)}")

    if args.algorithm == "bnb":
        print("\nRunning Branch-and-Bound Scheduler")
        scheduler = BranchAndBoundScheduler(
            instance_data=instance,
            time_limit_sec=args.time_limit,
            randomized_restarts=args.restarts,
            seed=args.seed,
            verbose=True
        )
    else:
        print("\nRunning Beam Search Scheduler")
        scheduler = BeamSearchScheduler(
            instance_data=instance,
            beam_width=100,
            lookahead_limit=4,
            density_percentile=25,
            verbose=True
        )

    solution = scheduler.generate_solution()

    print(f"\n✓ Generated solution with total score: {solution.total_score}")

    algorithm_name = type(scheduler).__name__.lower()
    serializer = SolutionSerializer(input_file_path=file_path, algorithm_name=algorithm_name)
    serializer.serialize(solution)

    print("✓ Solution saved to output file")


if __name__ == "__main__":
    main()
