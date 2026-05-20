import argparse
import bisect
import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ga_experiment import GA_BASE_PARAMS, build_ga_parameters, instance_name, save_solution
from models.solution import Solution
from parser.parser import Parser
from scheduler.branch_and_bound_scheduler import BranchAndBoundScheduler
from scheduler.genetic_algorithm_scheduler import GeneticAlgorithmScheduler
from utils.algorithm_utils import AlgorithmUtils
from utils.utils import Utils


ILS_OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "bnb_ga_ils"
DEFAULT_RUNS = 10
DEFAULT_BNB_TIME = 30.0
DEFAULT_GA_TIME = 240.0
DEFAULT_ILS_TIME = 60.0
DEFAULT_MAX_STAGNATION = 400

class ILSScheduler:
    def __init__(
        self,
        instance_data,
        seed=None,
        candidate_pool_size=700,
        top_k=6,
        repair_random_rate=0.35,
        max_removed=8,
    ):
        self.instance = instance_data
        self.rng = random.Random(seed if seed is not None else random.randint(1, 1_000_000))
        self.top_k = top_k
        self.repair_random_rate = repair_random_rate
        self.max_removed = max_removed
        self.score_cache = {}
        self.quality_cache = {}

        Utils.set_current_instance(self.instance)
        self.input_overlap_ids = AlgorithmUtils.compute_input_overlap_ids(self.instance)
        self.max_bonus_by_genre = AlgorithmUtils.build_max_bonus_by_genre(self.instance)
        self.tail_size = max(3, int(getattr(self.instance, "max_consecutive_genre", 3)))

        self.programs_by_start = {}
        for channel in self.instance.channels:
            for program in channel.programs:
                program.channel_id = getattr(program, "channel_id", channel.channel_id) or channel.channel_id
                if getattr(program, "unique_id", None) not in self.input_overlap_ids:
                    self.programs_by_start.setdefault(Utils.get_start(program), []).append(program)

        for start, programs in self.programs_by_start.items():
            self.programs_by_start[start] = sorted(
                programs,
                key=self._program_value,
                reverse=True,
            )[:candidate_pool_size]
        self.start_times = sorted(self.programs_by_start)

    def _program_value(self, program):
        return getattr(program, "score", 0) + self.max_bonus_by_genre.get(getattr(program, "genre", ""), 0)

    def _score(self, schedule):
        filtered = AlgorithmUtils.filter_valid_schedule(schedule, self.instance, self.input_overlap_ids)
        key = Utils.schedule_key(filtered)
        if key not in self.score_cache:
            self.score_cache[key] = AlgorithmUtils.score_filtered_schedule(filtered, self.instance)
        return self.score_cache[key]

    def _quality(self, item):
        uid = Utils.get_uid(item)
        if uid not in self.quality_cache:
            self.quality_cache[uid] = AlgorithmUtils.get_segment_quality(self.instance, item, self.input_overlap_ids)
        return self.quality_cache[uid]

    def _can_add(self, schedule, item):
        program = Utils.get_program_by_unique_id(self.instance, Utils.get_uid(item))
        start = Utils.get_start(item)
        end = Utils.get_end(item)

        if not program or Utils.get_uid(item) in self.input_overlap_ids:
            return False
        if start < self.instance.opening_time or end > self.instance.closing_time or end <= start:
            return False
        if schedule and start < Utils.get_end(schedule[-1]):
            return False

        full_length = Utils.get_end(program) - Utils.get_start(program)
        item_length = end - start
        if full_length >= self.instance.min_duration and item_length < self.instance.min_duration:
            return False
        if full_length < self.instance.min_duration and item_length != full_length:
            return False

        for block in self.instance.priority_blocks:
            if start < block.end and end > block.start and item.channel_id not in block.allowed_channels:
                return False

        genre = getattr(program, "genre", "")
        streak = 1
        for previous in reversed(schedule[-self.tail_size:]):
            previous_program = Utils.get_program_by_unique_id(self.instance, Utils.get_uid(previous))
            if not previous_program or getattr(previous_program, "genre", "") != genre:
                break
            streak += 1
        return streak <= self.instance.max_consecutive_genre

    def _best_candidate(self, schedule, used_ids, cursor, end_time):
        candidates = []
        left = bisect.bisect_left(self.start_times, cursor)
        right = bisect.bisect_right(self.start_times, end_time)

        for start in self.start_times[left:right]:
            for program in self.programs_by_start[start]:
                uid = getattr(program, "unique_id", None)
                if uid in used_ids or Utils.get_end(program) > end_time:
                    continue

                item = Utils.make_schedule(
                    program.channel_id, program.program_id, Utils.get_start(program), Utils.get_end(program),
                    self._program_value(program), uid,
                )
                if self._can_add(schedule, item):
                    wait_penalty = 0.03 * max(0, start - cursor)
                    switch_penalty = self.instance.switch_penalty if schedule and schedule[-1].channel_id != program.channel_id else 0
                    candidates.append((self._program_value(program) - wait_penalty - switch_penalty, item))

        candidates = sorted(candidates, key=lambda pair: pair[0], reverse=True)[:self.top_k]
        if not candidates:
            return None
        return self.rng.choice(candidates)[1] if len(candidates) > 1 and self.rng.random() < self.repair_random_rate else candidates[0][1]

    def _fill_interval(self, schedule, used_ids, start_time, end_time):
        cursor = start_time
        while cursor < end_time:
            item = self._best_candidate(schedule, used_ids, cursor, end_time)
            if item is None:
                break
            schedule.append(item)
            used_ids.add(Utils.get_uid(item))
            cursor = Utils.get_end(item)
        return cursor

    def _repair(self, schedule):
        base = AlgorithmUtils.filter_valid_schedule(schedule, self.instance, self.input_overlap_ids)
        repaired = []
        used_ids = {Utils.get_uid(item) for item in base}
        cursor = self.instance.opening_time

        for item in sorted(base, key=Utils.sort_schedule_item):
            cursor = self._fill_interval(repaired, used_ids, cursor, Utils.get_start(item))
            if Utils.get_start(item) >= cursor and self._can_add(repaired, item):
                repaired.append(item)
                cursor = Utils.get_end(item)

        self._fill_interval(repaired, used_ids, cursor, self.instance.closing_time)
        return AlgorithmUtils.filter_valid_schedule(repaired, self.instance, self.input_overlap_ids)

    def _perturb(self, schedule, max_removed=None):
        if not schedule:
            return self._repair([])

        child = list(schedule)
        max_removed = max_removed or self.max_removed
        remove_count = min(len(child), self.rng.randint(1, max_removed))
        weak_indexes = sorted(range(len(child)), key=lambda index: self._quality(child[index]))

        if len(child) > remove_count and self.rng.random() < 0.45:
            start = self.rng.randint(0, len(child) - remove_count)
            del child[start:start + remove_count]
        else:
            candidate_pool = weak_indexes[:max(remove_count, min(len(child), 12))]
            removed = set(self.rng.sample(candidate_pool, remove_count))
            child = [item for index, item in enumerate(child) if index not in removed]

        return self._repair(child)

    def improve(self, schedule, time_limit_sec=DEFAULT_ILS_TIME, max_stagnation=DEFAULT_MAX_STAGNATION):
        deadline = time.perf_counter() + time_limit_sec
        original = AlgorithmUtils.filter_valid_schedule(schedule, self.instance, self.input_overlap_ids)
        best = max(original, self._repair(original), key=self._score)
        best_score = self._score(best)
        current = list(best)
        current_score = best_score
        iterations = 0
        stagnation = 0

        while time.perf_counter() < deadline:
            iterations += 1
            aggressive = stagnation >= max_stagnation
            source = best if self.rng.random() < 0.75 else current
            max_removed = self.max_removed * 2 if aggressive else self.max_removed
            candidate = self._perturb(source, max_removed=max_removed)
            candidate_score = self._score(candidate)

            if candidate_score > best_score:
                best = candidate
                best_score = candidate_score
                current = list(candidate)
                current_score = candidate_score
                stagnation = 0
            else:
                if candidate_score >= current_score or self.rng.random() < 0.08:
                    current = list(candidate)
                    current_score = candidate_score
                stagnation += 1
                if aggressive:
                    current = self._perturb(best, max_removed=max_removed)
                    current_score = self._score(current)
                    stagnation = 0

        return Solution(best, best_score), iterations, stagnation


def _score(solution):
    return int(solution.total_score) if solution else None


def run_bnb_ga_ils_once(instance, name, run_index, seed, bnb_time, ga_time, ils_time, ga_profile):
    started = time.perf_counter()
    print(f"\nRun {run_index} | seed={seed}")

    bnb_start = time.perf_counter()
    bnb = BranchAndBoundScheduler(instance, time_limit_sec=bnb_time, seed=seed, verbose=False).generate_solution()
    bnb_elapsed = time.perf_counter() - bnb_start
    print(f"[BnB] score={_score(bnb)} | time={bnb_elapsed:.2f}s")

    ga_params = build_ga_parameters(ga_profile, name, GA_BASE_PARAMS)
    ga_start = time.perf_counter()
    ga = GeneticAlgorithmScheduler(
        instance,
        initial_schedule=bnb.scheduled_programs if bnb else None,
        time_limit_sec=ga_time,
        seed=seed,
        stagnation_limit=10,
        min_runtime_before_stop_sec=min(20.0, max(1.0, ga_time)),
        **ga_params,
    )
    ga_solution = ga.generate_solution()
    ga_elapsed = time.perf_counter() - ga_start
    print(f"[GA ] score={_score(ga_solution)} | generations={ga.generations_done} | time={ga_elapsed:.2f}s")

    ils_start = time.perf_counter()
    ils_solution, iterations, stagnation = ILSScheduler(instance, seed=seed + 100_000).improve(ga_solution.scheduled_programs, ils_time)
    ils_elapsed = time.perf_counter() - ils_start
    final = max(ga_solution, ils_solution, key=lambda solution: solution.total_score)
    print(f"[ILS] score={_score(ils_solution)} | iterations={iterations} | time={ils_elapsed:.2f}s")
    print(f"[Best] final={_score(final)} | improvement={_score(final) - _score(ga_solution)}")

    return final, dict(
        run=run_index,
        seed=seed,
        bnb_score=_score(bnb),
        ga_score=_score(ga_solution),
        ils_score=_score(ils_solution),
        final_score=_score(final),
        improvement_over_ga=_score(final) - _score(ga_solution),
        bnb_seconds=round(bnb_elapsed, 2),
        ga_seconds=round(ga_elapsed, 2),
        ils_seconds=round(ils_elapsed, 2),
        total_seconds=round(time.perf_counter() - started, 2),
        ga_generations=ga.generations_done,
        ils_iterations=iterations,
        ils_stagnation=stagnation,
    )


def run_ils_experiment(selected_instance_name, instance, runs=DEFAULT_RUNS, bnb_time_per_run=DEFAULT_BNB_TIME,
                       ga_time_per_run=DEFAULT_GA_TIME, ils_time_per_run=DEFAULT_ILS_TIME, ga_profile="single"):
    name = instance_name(selected_instance_name)
    output_dir = ILS_OUTPUT_DIR / name
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in output_dir.glob("run*.json"):
        path.unlink()

    rng = random.Random(42)
    summary = []
    print(f"\nPipeline: Branch and Bound -> GA -> ILS")
    print(f"Instance: {name} | runs={runs} | output={output_dir.relative_to(PROJECT_ROOT)}")

    for run_index in range(1, runs + 1):
        seed = rng.randint(1, 1_000_000)
        solution, row = run_bnb_ga_ils_once(
            instance, name, run_index, seed, bnb_time_per_run, ga_time_per_run, ils_time_per_run, ga_profile
        )
        output_path = output_dir / f"run{run_index}_{int(solution.total_score)}.json"
        save_solution(solution, output_path)
        row["output"] = str(output_path.relative_to(PROJECT_ROOT))
        summary.append(row)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=4), encoding="utf-8")
    print(f"\nSummary saved: {summary_path.relative_to(PROJECT_ROOT)}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run Branch and Bound -> GA -> ILS")
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--bnb-time", type=float, default=DEFAULT_BNB_TIME)
    parser.add_argument("--ga-time", type=float, default=DEFAULT_GA_TIME)
    parser.add_argument("--ils-time", type=float, default=DEFAULT_ILS_TIME)
    parser.add_argument("--ga-profile", choices=["single", "tuned", "strong"], default="single")
    args = parser.parse_args()
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path

    instance = Parser(str(input_path)).parse()
    Utils.set_current_instance(instance)
    run_ils_experiment(
        input_path.stem, instance, args.runs, args.bnb_time, args.ga_time, args.ils_time, args.ga_profile
    )


if __name__ == "__main__":
    main()
