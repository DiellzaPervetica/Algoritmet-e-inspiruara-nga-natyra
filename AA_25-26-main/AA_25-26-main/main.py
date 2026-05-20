import os
import argparse
from pathlib import Path
from parser.file_selector import select_file
from parser.parser import Parser
from serializer.serializer import SolutionSerializer
from utils.utils import Utils
from ga_experiment import GA_EXPERIMENTS, run_all_experiment_batches, run_instance_experiment

from scheduler.beam_search_scheduler import BeamSearchScheduler
from scheduler.branch_and_bound_scheduler import BranchAndBoundScheduler
from scheduler.ils_scheduler import (
    DEFAULT_BNB_TIME,
    DEFAULT_GA_TIME,
    DEFAULT_ILS_TIME,
    DEFAULT_RUNS,
    run_ils_experiment,
)

PROJECT_ROOT = Path(__file__).resolve().parent


def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')


def select_ga_experiment():
    print("\n Select Genetic Algorithm Experiment:")
    for number, _, description, _ in GA_EXPERIMENTS:
        print(f" [{number}] {description}")

    selected = input("Selection: ").strip()
    for experiment in GA_EXPERIMENTS:
        if selected == experiment[0]:
            return experiment

    print("[Error] Invalid GA experiment selected.")
    return None


def main():
    parser_arg = argparse.ArgumentParser(description="Run TV scheduling algorithms")
    parser_arg.add_argument("--input", "-i", dest="input_file", help="Path to input JSON")
    parser_arg.add_argument("--ga-auto", action="store_true", help="Run all GA experiments automatically (non-interactive)")
    parser_arg.add_argument("--ga-runs", type=int, default=10, help="Number of runs per experiment in auto mode")
    parser_arg.add_argument("--ga-time", type=float, default=300.0, help="Per-instance time limit (seconds) in auto mode")
    parser_arg.add_argument("--pipeline-runs", type=int, default=DEFAULT_RUNS, help="Number of runs for Branch and Bound -> GA -> ILS")
    parser_arg.add_argument("--bnb-time", type=float, default=DEFAULT_BNB_TIME, help="Branch and Bound time limit per pipeline run (seconds)")
    parser_arg.add_argument("--pipeline-ga-time", type=float, default=DEFAULT_GA_TIME, help="Genetic Algorithm time limit per pipeline run (seconds)")
    parser_arg.add_argument("--ils-time", type=float, default=DEFAULT_ILS_TIME, help="ILS time limit per pipeline run (seconds)")
    parser_arg.add_argument("--pipeline-ga-profile", choices=["single", "tuned", "strong"], default="single", help="GA parameter profile used inside the pipeline")
    args = parser_arg.parse_args()

    clear_console()

    if args.ga_auto:
        run_all_experiment_batches(runs=args.ga_runs, per_instance_time_sec=args.ga_time)
        return
    print("=" * 60)
    print("            TV SCHEDULING OPTIMIZATION SYSTEM               ")
    print("=" * 60)

    if args.input_file:
        input_path = Path(args.input_file)
        if not input_path.is_absolute():
            input_path = PROJECT_ROOT / input_path
        file_path = str(input_path)
    else:
        file_path = select_file()

    if not file_path:
        print("[Error] No instance selected.")
        return

    instance_name = Path(file_path).stem
    parser = Parser(file_path)
    instance = parser.parse()
    Utils.set_current_instance(instance)

    print(f"\n[Info] Instance: {instance_name}")
    print("\n Select Algorithm:")
    print(" [1] Beam Search")
    print(" [2] Branch and Bound")
    print(" [3] Genetic Algorithm")
    print(" [4] Branch and Bound + Genetic Algorithm + ILS")
    
    choice = input("Selection: ").strip()

    if choice == '1':
        scheduler = BeamSearchScheduler(instance, beam_width=100, lookahead_limit=4)
        sol = scheduler.generate_solution()
        if sol:
            score = int(sol.total_score)
            output_dir = PROJECT_ROOT / "data" / "output" / "beam_search" / instance_name
            filename = f"{instance_name}_run1_score{score}.json"

            serializer = SolutionSerializer(file_path, "beam_search", output_dir=output_dir)
            serializer.serialize(sol, output_filename=filename)
            print(f"Score: {score} | Saved: {filename}")

    elif choice == '2':
        scheduler = BranchAndBoundScheduler(instance, time_limit_sec=30.0)
        sol = scheduler.generate_solution()
        if sol:
            score = int(sol.total_score)
            output_dir = PROJECT_ROOT / "data" / "output" / "branch_and_bound" / instance_name
            filename = f"{instance_name}_run1_score{score}.json"

            serializer = SolutionSerializer(file_path, "branch_and_bound", output_dir=output_dir)
            serializer.serialize(sol, output_filename=filename)
            print(f"Score: {score} | Saved: {filename}")

    elif choice == '3':
        experiment = select_ga_experiment()
        if not experiment:
            return

        _, experiment_name, _, profile = experiment
        run_instance_experiment(instance_name, instance, experiment_name, profile)

    elif choice == '4':
        print("\n[Info] Running pipeline: Branch and Bound -> Genetic Algorithm -> ILS")
        print(
            f"[Info] Runs={args.pipeline_runs} | "
            f"BnB={args.bnb_time:.0f}s | "
            f"GA={args.pipeline_ga_time:.0f}s | "
            f"ILS={args.ils_time:.0f}s | "
            f"profile={args.pipeline_ga_profile}"
        )
        run_ils_experiment(
            instance_name,
            instance,
            runs=args.pipeline_runs,
            bnb_time_per_run=args.bnb_time,
            ga_time_per_run=args.pipeline_ga_time,
            ils_time_per_run=args.ils_time,
            ga_profile=args.pipeline_ga_profile,
        )

if __name__ == "__main__":
    main()
