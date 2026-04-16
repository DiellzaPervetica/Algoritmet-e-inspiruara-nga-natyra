import os
import argparse
import json
import re
from pathlib import Path
from parser.file_selector import select_file
from parser.parser import Parser
from serializer.serializer import SolutionSerializer
from utils.utils import Utils
from models.schedule import Schedule

from scheduler.beam_search_scheduler import BeamSearchScheduler
from scheduler.branch_and_bound_scheduler import BranchAndBoundScheduler
from scheduler.genetic_algorithm_scheduler import GeneticAlgorithmScheduler

PROJECT_ROOT = Path(__file__).resolve().parent


def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')


def _make_schedule(channel_id, program_id, start, end, fitness, unique_program_id):
    try:
        return Schedule(
            channel_id=channel_id,
            program_id=program_id,
            start=int(start),
            end=int(end),
            fitness=float(fitness),
            unique_program_id=unique_program_id,
        )
    except TypeError:
        return Schedule(
            channel_id=channel_id,
            program_id=program_id,
            start_time=int(start),
            end_time=int(end),
            fitness=float(fitness),
            unique_program_id=unique_program_id,
        )


def _extract_last_number(path: Path) -> int:
    matches = re.findall(r"(\d+)", path.stem)
    return int(matches[-1]) if matches else -1


def _find_best_branch_and_bound_output(instance_name: str) -> Path | None:
    legacy_dir = PROJECT_ROOT / "data" / "output" / "branchandboundscheduler"
    modern_dir = PROJECT_ROOT / "data" / "output" / "branch_and_bound" / instance_name

    candidates: list[Path] = []
    if legacy_dir.exists():
        candidates.extend(sorted(legacy_dir.glob(f"{instance_name}*.json")))
    if modern_dir.exists():
        candidates.extend(sorted(modern_dir.glob("*.json")))

    if not candidates:
        return None

    candidates = [path for path in candidates if path.is_file()]
    if not candidates:
        return None

    return max(candidates, key=_extract_last_number)


def _load_initial_schedule_from_output(output_path: Path, instance) -> list[Schedule]:
    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    id_map = {
        (getattr(program, "program_id", ""), getattr(channel, "channel_id", None)): getattr(program, "unique_id", getattr(program, "program_id", ""))
        for channel in instance.channels
        for program in channel.programs
    }

    schedules = []
    for item in data.get("scheduled_programs", []):
        channel_id = item.get("channel_id")
        program_id = item.get("program_id")
        schedules.append(
            _make_schedule(
                channel_id=channel_id,
                program_id=program_id,
                start=item.get("start", item.get("start_time")),
                end=item.get("end", item.get("end_time")),
                fitness=item.get("fitness", 0),
                unique_program_id=id_map.get((program_id, channel_id), program_id),
            )
        )

    return schedules

def main():
    parser_arg = argparse.ArgumentParser(description="Run TV scheduling algorithms")
    parser_arg.add_argument("--input", "-i", dest="input_file", help="Path to input JSON")
    args = parser_arg.parse_args()

    clear_console()
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
        initial_schedule = None
        bnb_output_path = _find_best_branch_and_bound_output(instance_name)
        if bnb_output_path:
            initial_schedule = _load_initial_schedule_from_output(bnb_output_path, instance)
            print(f"[Info] Genetic Algorithm initial schedule loaded from: {bnb_output_path.name}")
        else:
            print("[Info] No saved Branch and Bound output found. Falling back to internal warm start.")

        scheduler = GeneticAlgorithmScheduler(
            instance,
            initial_schedule=initial_schedule,
            time_limit_sec=300.0,
            population_size=10,
            mutation_rate=0.30
        )
        sol = scheduler.generate_solution()
        if sol:
            score = int(sol.total_score)
            output_dir = PROJECT_ROOT / "data" / "output" / "genetic_algorithm" / instance_name
            filename = f"{instance_name}_run1_score{score}.json"

            serializer = SolutionSerializer(file_path, "genetic_algorithm", output_dir=output_dir)
            serializer.serialize(sol, output_filename=filename)
            print(f"Score: {score} | Saved: {filename}")

if __name__ == "__main__":
    main()
