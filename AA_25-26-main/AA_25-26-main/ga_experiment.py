import argparse
import json
import random
import re
import time
from pathlib import Path

from models.schedule import Schedule
from parser.parser import Parser
from scheduler.genetic_algorithm_scheduler import GeneticAlgorithmScheduler


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_ROOT / "data" / "input"
BNB_OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "branchandboundscheduler"
GA_OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "genetic_algorithm"
GA_PARAM_NAMES = (
    "population_size",
    "tournament_size",
    "elite_size",
    "crossover_rate",
    "mutation_rate",
    "max_generations",
    "candidate_pool_size",
    "repair_random_rate",
)

INSTANCE_FILES = [
    "australia_iptv.json",
    "canada_pw.json",
    "china_pw.json",
    "croatia_tv_input.json",
    "france_iptv.json",
    "germany_tv_input.json",
    "kosovo_tv_input.json",
    "netherlands_tv_input.json",
    "singapore_pw.json",
    "spain_iptv.json",
    "toy.json",
    "uk_iptv.json",
    "uk_tv_input.json",
    "usa_tv_input.json",
    "us_iptv.json",
    "youtube_gold.json",
    "youtube_premium.json",
]

SMALL_FAST_PARAMS = {
    "population_size": 4,
    "mutation_rate": 0.20,
}

WIDE_REPAIR_PARAMS = {
    "population_size": 14,
    "crossover_rate": 0.85,
    "mutation_rate": 0.45,
}

TUNED_PARAMS = {
    "australia_iptv": SMALL_FAST_PARAMS,
    "canada_pw": SMALL_FAST_PARAMS,
    "china_pw": {
        "population_size": 12,
        "mutation_rate": 0.30,
    },
    "spain_iptv": WIDE_REPAIR_PARAMS,
    "uk_iptv": WIDE_REPAIR_PARAMS,
    "youtube_gold": SMALL_FAST_PARAMS,
    "youtube_premium": SMALL_FAST_PARAMS,
}

TIME_WEIGHTS = {
    "australia_iptv": 1.1,
    "canada_pw": 0.8,
    "china_pw": 0.8,
    "croatia_tv": 0.5,
    "france_iptv": 2.3,
    "germany_tv": 0.4,
    "kosovo_tv": 0.4,
    "netherlands_tv": 0.4,
    "singapore_pw": 0.5,
    "spain_iptv": 1.8,
    "toy": 0.2,
    "uk_iptv": 1.8,
    "uk_tv": 0.5,
    "usa_tv": 0.4,
    "us_iptv": 0.8,
    "youtube_gold": 0.4,
    "youtube_premium": 3.0,
}


def instance_name(file_name):
    return Path(file_name).stem.replace("_input", "")


def last_number(path):
    matches = re.findall(r"(\d+)", path.stem)
    return int(matches[-1]) if matches else -1


def find_bnb_output(name):
    files = list(BNB_OUTPUT_DIR.glob(f"{name}_output_branchandboundscheduler_*.json"))
    return max(files, key=last_number) if files else None


def make_schedule(channel_id, program_id, start, end, fitness, unique_program_id):
    return Schedule(
        program_id=program_id,
        channel_id=channel_id,
        start=int(start),
        end=int(end),
        fitness=float(fitness),
        unique_program_id=unique_program_id,
    )


def load_initial_schedule(path, instance):
    program_ids = {
        (program.program_id, channel.channel_id): program.unique_id
        for channel in instance.channels
        for program in channel.programs
    }

    data = json.loads(path.read_text(encoding="utf-8"))
    schedules = []

    for item in data.get("scheduled_programs", []):
        channel_id = item["channel_id"]
        program_id = item["program_id"]
        schedules.append(
            make_schedule(
                channel_id=channel_id,
                program_id=program_id,
                start=item["start"],
                end=item["end"],
                fitness=item.get("fitness", 0),
                unique_program_id=program_ids.get((program_id, channel_id), program_id),
            )
        )

    return schedules


def save_solution(solution, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "program_id": item.program_id,
            "channel_id": item.channel_id,
            "start": item.start,
            "end": item.end,
        }
        for item in solution.scheduled_programs
    ]
    path.write_text(json.dumps({"scheduled_programs": rows}, indent=4), encoding="utf-8")


def get_parameters(params, name):
    values = {param: getattr(params, param) for param in GA_PARAM_NAMES}

    if params.profile == "tuned":
        values.update(TUNED_PARAMS.get(name, {}))

    return values


def get_time_limit(params, file_name, index, started):
    elapsed_total = time.perf_counter() - started
    remaining = params.total_time - elapsed_total
    instances_left = len(params.instances) - index + 1

    if remaining <= 1.0:
        return 0.0, remaining

    if params.time_profile == "adaptive":
        current_name = instance_name(file_name)
        remaining_files = params.instances[index - 1 :]
        weight_left = sum(TIME_WEIGHTS.get(instance_name(name), 1.0) for name in remaining_files)
        share = TIME_WEIGHTS.get(current_name, 1.0) / max(weight_left, 0.01)
        time_limit = remaining * share
    else:
        time_limit = remaining / instances_left

    return max(1.0, time_limit - params.time_buffer), remaining


def run_one_instance(file_name, output_root, run_index, time_limit, params, seed):
    input_path = INPUT_DIR / file_name
    name = instance_name(file_name)
    parser = Parser(str(input_path))
    instance = parser.parse()

    bnb_path = find_bnb_output(name)
    initial_schedule = load_initial_schedule(bnb_path, instance) if bnb_path else None
    ga_params = get_parameters(params, name)

    scheduler = GeneticAlgorithmScheduler(
        instance,
        initial_schedule=initial_schedule,
        time_limit_sec=time_limit,
        seed=seed,
        **ga_params,
    )

    started = time.perf_counter()
    solution = scheduler.generate_solution()
    elapsed = time.perf_counter() - started
    score = int(solution.total_score)

    output_name = f"run{run_index}_{score}.json"
    output_path = output_root / name / output_name
    save_solution(solution, output_path)

    return {
        "instance": name,
        "score": score,
        "programs": len(solution.scheduled_programs),
        "elapsed_seconds": round(elapsed, 2),
        "time_limit_seconds": round(time_limit, 2),
        "seed": seed,
        "bnb_output": bnb_path.name if bnb_path else None,
        "initial_population_scores": scheduler.initial_population_scores,
        "generations": scheduler.generations_done,
        "parameters": ga_params,
        "output": str(output_path.relative_to(PROJECT_ROOT)),
    }


def run_batch(run_index, params):
    output_root = GA_OUTPUT_DIR / "experiments" / params.experiment_name if params.experiment_name else GA_OUTPUT_DIR
    summary_dir = output_root / "run_summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(params.seed + run_index)
    started = time.perf_counter()
    rows = []

    for index, file_name in enumerate(params.instances, start=1):
        time_limit, remaining = get_time_limit(params, file_name, index, started)
        if time_limit <= 0:
            break

        seed = rng.randint(1, 1_000_000)

        print(
            f"Run {run_index} | {file_name} | "
            f"time={time_limit:.1f}s | left={max(0.0, remaining):.1f}s"
        )
        rows.append(run_one_instance(file_name, output_root, run_index, time_limit, params, seed))

    summary_path = summary_dir / f"run{run_index}_summary.json"
    summary_path.write_text(json.dumps(rows, indent=4), encoding="utf-8")

    return {
        "run": run_index,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "instances": rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Genetic Algorithm experiments")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--total-time", type=float, default=300.0)
    parser.add_argument("--population-size", type=int, default=8)
    parser.add_argument("--tournament-size", type=int, default=3)
    parser.add_argument("--elite-size", type=int, default=2)
    parser.add_argument("--crossover-rate", type=float, default=0.90)
    parser.add_argument("--mutation-rate", type=float, default=0.25)
    parser.add_argument("--max-generations", type=int, default=10000)
    parser.add_argument("--candidate-pool-size", type=int, default=700)
    parser.add_argument("--repair-random-rate", type=float, default=0.20)
    parser.add_argument("--time-buffer", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--instances", nargs="*", default=INSTANCE_FILES)
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--experiment-name", default="")
    parser.add_argument("--profile", choices=["single", "tuned"], default="single")
    parser.add_argument("--time-profile", choices=["equal", "adaptive"], default="adaptive")
    args = parser.parse_args()

    output_root = GA_OUTPUT_DIR / "experiments" / args.experiment_name if args.experiment_name else GA_OUTPUT_DIR
    output_root.mkdir(parents=True, exist_ok=True)

    if args.clean_output:
        for path in output_root.rglob("*.json"):
            path.unlink()

    all_runs = []

    for run_index in range(1, args.runs + 1):
        all_runs.append(run_batch(run_index, args))

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(all_runs, indent=4), encoding="utf-8")
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
