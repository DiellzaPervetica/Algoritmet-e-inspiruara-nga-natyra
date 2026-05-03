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
GA_EXPERIMENT_RUNS = 10
GA_EXPERIMENT_TIME = 300.0
GA_BASE_PARAMS = {
    "population_size": 50,
    "tournament_size": 3,
    "elite_size": 2,
    "crossover_rate": 0.90,
    "mutation_rate": 0.30,
    "max_generations": 10000,
    "candidate_pool_size": 700,
    "repair_random_rate": 0.20,
}
GA_EXPERIMENTS = [
    ("1", "exp_uniform_balanced", "Parametra te njejte per te gjitha instancat", "single"),
    ("2", "exp_tuned_by_instance", "Parametra te pershtatur sipas instances", "tuned"),
    ("3", "exp_stronger_exploration", "Parametra te pershtatur + eksplorim me i forte", "strong"),
]
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
    "population_size": 25,
    "mutation_rate": 0.25,
}

MEDIUM_PARAMS = {
    "population_size": 40,
    "mutation_rate": 0.30,
}

LARGE_PARAMS = {
    "population_size": 60,
    "crossover_rate": 0.88,
    "mutation_rate": 0.35,
}

HUGE_PARAMS = {
    "population_size": 70,
    "crossover_rate": 0.85,
    "mutation_rate": 0.40,
}

TUNED_PARAMS = {
    "croatia_tv": SMALL_FAST_PARAMS,
    "germany_tv": SMALL_FAST_PARAMS,
    "kosovo_tv": SMALL_FAST_PARAMS,
    "netherlands_tv": SMALL_FAST_PARAMS,
    "toy": SMALL_FAST_PARAMS,
    "singapore_pw": MEDIUM_PARAMS,
    "spain_iptv": MEDIUM_PARAMS,
    "uk_tv": MEDIUM_PARAMS,
    "australia_iptv": LARGE_PARAMS,
    "canada_pw": LARGE_PARAMS,
    "france_iptv": LARGE_PARAMS,
    "uk_iptv": LARGE_PARAMS,
    "china_pw": HUGE_PARAMS,
    "usa_tv": HUGE_PARAMS,
    "us_iptv": HUGE_PARAMS,
    "youtube_gold": HUGE_PARAMS,
    "youtube_premium": HUGE_PARAMS,
}

STRONG_DEFAULT_PARAMS = {
    "population_size": 70,
    "crossover_rate": 0.85,
    "mutation_rate": 0.40,
}

STRONG_SMALL_PARAMS = {
    "population_size": 35,
    "mutation_rate": 0.30,
}

STRONG_MEDIUM_PARAMS = {
    "population_size": 55,
    "mutation_rate": 0.35,
}

STRONG_LARGE_PARAMS = {
    "population_size": 75,
    "crossover_rate": 0.85,
    "mutation_rate": 0.42,
}

STRONG_HARD_PARAMS = {
    "population_size": 85,
    "crossover_rate": 0.85,
    "mutation_rate": 0.45,
}

STRONG_PARAMS = {
    "croatia_tv": STRONG_SMALL_PARAMS,
    "germany_tv": STRONG_SMALL_PARAMS,
    "kosovo_tv": STRONG_SMALL_PARAMS,
    "netherlands_tv": STRONG_SMALL_PARAMS,
    "toy": STRONG_SMALL_PARAMS,
    "singapore_pw": STRONG_MEDIUM_PARAMS,
    "spain_iptv": STRONG_MEDIUM_PARAMS,
    "uk_tv": STRONG_MEDIUM_PARAMS,
    "australia_iptv": STRONG_LARGE_PARAMS,
    "canada_pw": STRONG_LARGE_PARAMS,
    "france_iptv": STRONG_LARGE_PARAMS,
    "uk_iptv": STRONG_LARGE_PARAMS,
    "china_pw": STRONG_HARD_PARAMS,
    "usa_tv": STRONG_HARD_PARAMS,
    "us_iptv": STRONG_HARD_PARAMS,
    "youtube_gold": STRONG_HARD_PARAMS,
    "youtube_premium": STRONG_HARD_PARAMS,
}

def instance_name(file_name):
    return Path(file_name).stem.replace("_input", "")


def build_ga_parameters(profile, name, values):
    selected = dict(values)

    if profile == "tuned":
        selected.update(TUNED_PARAMS.get(name, {}))

    if profile == "strong":
        selected.update(STRONG_DEFAULT_PARAMS)
        selected.update(STRONG_PARAMS.get(name, {}))

    return selected


def last_number(path):
    matches = re.findall(r"(\d+)", path.stem)
    return int(matches[-1]) if matches else -1


def find_bnb_output(name):
    modern_dir = PROJECT_ROOT / "data" / "output" / "branch_and_bound" / name
    files = list(BNB_OUTPUT_DIR.glob(f"{name}_output_branchandboundscheduler_*.json"))
    if modern_dir.exists():
        files.extend(modern_dir.glob("*.json"))
    return max(files, key=last_number) if files else None


def make_schedule(channel_id, program_id, start, end, fitness, unique_program_id):
    try:
        return Schedule(
            program_id=program_id,
            channel_id=channel_id,
            start=int(start),
            end=int(end),
            fitness=float(fitness),
            unique_program_id=unique_program_id,
        )
    except TypeError:
        return Schedule(
            program_id=program_id,
            channel_id=channel_id,
            start_time=int(start),
            end_time=int(end),
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
                start=item.get("start", item.get("start_time")),
                end=item.get("end", item.get("end_time")),
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
    return build_ga_parameters(params.profile, name, values)


def get_time_limit(params, file_name, index, started):
    elapsed_total = time.perf_counter() - started
    remaining = params.total_time - elapsed_total
    instances_left = len(params.instances) - index + 1

    if remaining <= 1.0:
        return 0.0, remaining

    time_limit = remaining / instances_left

    if params.max_instance_time is not None:
        time_limit = min(time_limit, params.max_instance_time)

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
        stagnation_limit=params.stagnation_limit,
        min_runtime_before_stop_sec=params.min_runtime_before_stop,
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


def run_instance_experiment(
    selected_instance_name,
    instance,
    experiment_name,
    profile,
    runs=GA_EXPERIMENT_RUNS,
    time_limit=GA_EXPERIMENT_TIME,
):
    name = instance_name(selected_instance_name)
    output_root = GA_OUTPUT_DIR / "experiments" / experiment_name
    output_dir = output_root / name
    output_dir.mkdir(parents=True, exist_ok=True)

    bnb_path = find_bnb_output(name)
    initial_schedule = load_initial_schedule(bnb_path, instance) if bnb_path else None
    ga_params = build_ga_parameters(profile, name, GA_BASE_PARAMS)
    rng = random.Random(42)
    summary = []

    if bnb_path:
        print(f"[Info] Genetic Algorithm initial schedule loaded from: {bnb_path.name}")
    else:
        print("[Info] No saved Branch and Bound output found. Falling back to internal warm start.")

    print(f"\n[Info] Experiment: {experiment_name}")
    print(f"[Info] Runs: {runs}")
    print(f"[Info] Max time per run: {time_limit:.0f} seconds")
    print(f"[Info] Parameters: {ga_params}")

    for run_index in range(1, runs + 1):
        seed = rng.randint(1, 1_000_000)
        print(f"\nRun {run_index}/{runs} | seed={seed}")

        scheduler = GeneticAlgorithmScheduler(
            instance,
            initial_schedule=initial_schedule,
            time_limit_sec=time_limit,
            seed=seed,
            stagnation_limit=10,
            min_runtime_before_stop_sec=20,
            **ga_params,
        )

        started = time.perf_counter()
        solution = scheduler.generate_solution()
        elapsed = time.perf_counter() - started

        if not solution:
            print("[Error] Genetic Algorithm did not return a solution.")
            continue

        score = int(solution.total_score)
        output_path = output_dir / f"run{run_index}_{score}.json"
        save_solution(solution, output_path)
        print(f"Score: {score} | Saved: {output_path.name} | Elapsed: {elapsed:.2f}s")

        summary.append(
            {
                "run": run_index,
                "score": score,
                "programs": len(solution.scheduled_programs),
                "elapsed_seconds": round(elapsed, 2),
                "time_limit_seconds": time_limit,
                "seed": seed,
                "bnb_output": bnb_path.name if bnb_path else None,
                "initial_population_scores": scheduler.initial_population_scores,
                "generations": scheduler.generations_done,
                "parameters": ga_params,
                "output": str(output_path.relative_to(PROJECT_ROOT)),
            }
        )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=4), encoding="utf-8")
    print(f"\n[Info] Summary saved: {summary_path}")
    return summary


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


def run_experiment_batches(params):
    output_root = GA_OUTPUT_DIR / "experiments" / params.experiment_name if params.experiment_name else GA_OUTPUT_DIR
    output_root.mkdir(parents=True, exist_ok=True)

    if params.clean_output:
        for path in output_root.rglob("*.json"):
            path.unlink()

    all_runs = []
    for run_index in range(1, params.runs + 1):
        all_runs.append(run_batch(run_index, params))

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(all_runs, indent=4), encoding="utf-8")
    print(f"Summary saved: {summary_path}")
    return all_runs


def make_batch_params(experiment_name, profile, runs, per_instance_time_sec):
    total_time = per_instance_time_sec * len(INSTANCE_FILES)
    values = dict(GA_BASE_PARAMS)
    values.update(
        {
            "runs": runs,
            "total_time": total_time,
            "stagnation_limit": 10,
            "min_runtime_before_stop": 20,
            "max_instance_time": per_instance_time_sec,
            "time_buffer": 0,
            "seed": 42,
            "instances": INSTANCE_FILES,
            "clean_output": True,
            "experiment_name": experiment_name,
            "profile": profile,
        }
    )
    return argparse.Namespace(**values)


def run_all_experiment_batches(runs=GA_EXPERIMENT_RUNS, per_instance_time_sec=GA_EXPERIMENT_TIME):
    for _, experiment_name, _, profile in GA_EXPERIMENTS:
        params = make_batch_params(experiment_name, profile, runs, per_instance_time_sec)

        print("\n" + "=" * 60)
        print(f"[AUTO-GA] Starting: {experiment_name}")
        print("=" * 60)
        run_experiment_batches(params)

    total_runs = len(GA_EXPERIMENTS) * runs
    worst_case_hours = (total_runs * len(INSTANCE_FILES) * per_instance_time_sec) / 3600.0
    print(f"\n[AUTO-GA] Done. Total experiment runs: {total_runs}")
    print(f"[AUTO-GA] Per-instance time limit: {per_instance_time_sec:.2f} seconds")
    print(f"[AUTO-GA] Worst-case time estimate: {worst_case_hours:.2f} hours")


def main():
    parser = argparse.ArgumentParser(description="Genetic Algorithm experiments")
    parser.add_argument("--runs", type=int, default=GA_EXPERIMENT_RUNS)
    parser.add_argument("--total-time", type=float, default=GA_EXPERIMENT_TIME * len(INSTANCE_FILES))
    parser.add_argument("--population-size", type=int, default=GA_BASE_PARAMS["population_size"])
    parser.add_argument("--tournament-size", type=int, default=GA_BASE_PARAMS["tournament_size"])
    parser.add_argument("--elite-size", type=int, default=GA_BASE_PARAMS["elite_size"])
    parser.add_argument("--crossover-rate", type=float, default=GA_BASE_PARAMS["crossover_rate"])
    parser.add_argument("--mutation-rate", type=float, default=GA_BASE_PARAMS["mutation_rate"])
    parser.add_argument("--max-generations", type=int, default=GA_BASE_PARAMS["max_generations"])
    parser.add_argument("--candidate-pool-size", type=int, default=GA_BASE_PARAMS["candidate_pool_size"])
    parser.add_argument("--repair-random-rate", type=float, default=GA_BASE_PARAMS["repair_random_rate"])
    parser.add_argument("--stagnation-limit", type=int, default=10)
    parser.add_argument("--min-runtime-before-stop", type=float, default=20)
    parser.add_argument("--max-instance-time", type=float, default=GA_EXPERIMENT_TIME)
    parser.add_argument("--time-buffer", type=float, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--instances", nargs="*", default=INSTANCE_FILES)
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--experiment-name", default="")
    parser.add_argument("--profile", choices=["single", "tuned", "strong"], default="single")
    args = parser.parse_args()

    run_experiment_batches(args)


if __name__ == "__main__":
    main()
