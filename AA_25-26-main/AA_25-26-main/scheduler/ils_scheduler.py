import bisect
import json
import random
import time
from pathlib import Path

from ga_experiment import GA_OUTPUT_DIR, load_initial_schedule, instance_name, last_number, save_solution
from models.solution import Solution
from utils.algorithm_utils import AlgorithmUtils
from utils.utils import Utils

ILS_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "output" / "ga_and_ils"
DEFAULT_BASE_EXPERIMENT = "exp_uniform_balanced"
DEFAULT_RUNS = 10
DEFAULT_TIME_PER_RUN = 300.0
DEFAULT_MAX_STAGNATION = 100


def find_best_ga_output(name, experiment_name=DEFAULT_BASE_EXPERIMENT):
    output_dir = GA_OUTPUT_DIR / "experiments" / experiment_name / name
    files = list(output_dir.glob("run*.json"))
    return max(files, key=last_number) if files else None


class ILS:
    def __init__(
        self,
        instance,
        seed=None,
        candidate_pool_size=700,
        top_k=3,
        random_repair=0.15,
    ):
        self.instance = instance
        self.rng = random.Random(seed if seed is not None else random.randint(1, 1_000_000))
        self.top_k = top_k
        self.random_repair = random_repair
        self.score_cache = {}
        self.quality_cache = {}
        Utils.set_current_instance(instance)
        self.input_overlap_ids = AlgorithmUtils.compute_input_overlap_ids(instance)
        self.max_bonus_by_genre = AlgorithmUtils.build_max_bonus_by_genre(instance)
        self.tail_size = max(3, int(getattr(instance, "max_consecutive_genre", 3)))
        self.programs_by_start = {}

        for channel in instance.channels:
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
            self.quality_cache[uid] = AlgorithmUtils.get_segment_quality(
                self.instance,
                item,
                self.input_overlap_ids,
            )
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

        program_length = Utils.get_end(program) - Utils.get_start(program)
        item_length = end - start
        if program_length >= self.instance.min_duration and item_length < self.instance.min_duration:
            return False
        if program_length < self.instance.min_duration and item_length != program_length:
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

    def _pick_candidate(self, schedule, used_ids, cursor, end_time):
        candidates = []
        left = bisect.bisect_left(self.start_times, cursor)
        right = bisect.bisect_right(self.start_times, end_time)

        for start in self.start_times[left:right]:
            for program in self.programs_by_start[start]:
                uid = getattr(program, "unique_id", None)
                if uid in used_ids or Utils.get_end(program) > end_time:
                    continue

                item = Utils.make_schedule(
                    program.channel_id,
                    program.program_id,
                    Utils.get_start(program),
                    Utils.get_end(program),
                    self._program_value(program),
                    uid,
                )

                if not self._can_add(schedule, item):
                    continue

                wait_penalty = 0.03 * max(0, start - cursor)
                switch_penalty = self.instance.switch_penalty if schedule and schedule[-1].channel_id != program.channel_id else 0
                value = self._program_value(program) - wait_penalty - switch_penalty
                candidates.append((value, item))
                candidates = sorted(candidates, key=lambda pair: pair[0], reverse=True)[:self.top_k]

        if not candidates:
            return None
        if len(candidates) > 1 and self.rng.random() < self.random_repair:
            return self.rng.choice(candidates)[1]
        return candidates[0][1]

    def _fill_interval(self, schedule, used_ids, start_time, end_time):
        cursor = start_time
        while cursor < end_time:
            item = self._pick_candidate(schedule, used_ids, cursor, end_time)
            if item is None:
                break
            schedule.append(item)
            used_ids.add(Utils.get_uid(item))
            cursor = Utils.get_end(item)
        return cursor

    def _repair(self, schedule):
        base = AlgorithmUtils.filter_valid_schedule(schedule, self.instance, self.input_overlap_ids)
        fixed = []
        used_ids = {Utils.get_uid(item) for item in base}
        cursor = self.instance.opening_time

        for item in sorted(base, key=Utils.sort_schedule_item):
            cursor = self._fill_interval(fixed, used_ids, cursor, Utils.get_start(item))
            if Utils.get_start(item) >= cursor and self._can_add(fixed, item):
                fixed.append(item)
                cursor = Utils.get_end(item)

        self._fill_interval(fixed, used_ids, cursor, self.instance.closing_time)
        return AlgorithmUtils.filter_valid_schedule(fixed, self.instance, self.input_overlap_ids)

    def _weak_indexes(self, schedule, count):
        return sorted(range(len(schedule)), key=lambda index: self._quality(schedule[index]))[:count]

    def _local_search(self, schedule, deadline, attempts):
        best = max(schedule, self._repair(schedule), key=self._score)
        best_score = self._score(best)

        for _ in range(attempts):
            if time.perf_counter() >= deadline or not best:
                break

            remove_index = self.rng.choice(self._weak_indexes(best, min(8, len(best))))
            candidate = self._repair(best[:remove_index] + best[remove_index + 1:])
            candidate_score = self._score(candidate)

            if candidate_score > best_score:
                best = candidate
                best_score = candidate_score

        return best

    def _perturb(self, schedule, size):
        if not schedule:
            return self._repair([])

        child = list(schedule)
        remove_count = min(len(child), self.rng.randint(1, max(1, size)))

        if len(child) > remove_count and self.rng.random() < 0.5:
            start = self.rng.randint(0, len(child) - remove_count)
            del child[start:start + remove_count]
        else:
            weak_indexes = self._weak_indexes(child, min(len(child), remove_count * 4))
            indexes_to_remove = set(self.rng.sample(weak_indexes, remove_count))
            child = [item for index, item in enumerate(child) if index not in indexes_to_remove]

        return self._repair(child)

    def improve(
        self,
        schedule,
        time_limit_sec=300.0,
        max_stagnation=100,
        local_attempts=40,
        perturbation_size=4,
    ):
        deadline = time.perf_counter() + time_limit_sec
        original = AlgorithmUtils.filter_valid_schedule(schedule, self.instance, self.input_overlap_ids)
        best = max(original, self._local_search(original, deadline, local_attempts), key=self._score)
        best_score = self._score(best)
        stagnation = 0
        iterations = 0

        while time.perf_counter() < deadline and stagnation < max_stagnation:
            iterations += 1
            candidate = self._local_search(self._perturb(best, perturbation_size), deadline, local_attempts)
            candidate_score = self._score(candidate)

            if candidate_score > best_score:
                best = candidate
                best_score = candidate_score
            else:
                stagnation += 1

        return Solution(best, best_score), iterations, stagnation


def run_ils_experiment(
    selected_instance_name,
    instance,
    runs=DEFAULT_RUNS,
    time_per_run=DEFAULT_TIME_PER_RUN,
    base_experiment=DEFAULT_BASE_EXPERIMENT,
    max_stagnation=DEFAULT_MAX_STAGNATION,
):
    name = instance_name(selected_instance_name)
    base_path = find_best_ga_output(name, base_experiment)

    if not base_path:
        print(f"[Error] No GA output found for {name} in experiment {base_experiment}.")
        print("[Info] Run Genetic Algorithm first, then try ILS.")
        return []

    base_schedule = load_initial_schedule(base_path, instance)
    output_dir = ILS_OUTPUT_DIR / name
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_rng = random.Random(42)
    scheduler = ILS(instance, seed=seed_rng.randint(1, 1_000_000))
    base_score = scheduler._score(base_schedule)
    summary = []

    print(f"[Info] ILS base solution loaded from: {base_path.name}")
    print(f"[Info] Base score: {base_score}")
    print(f"[Info] Runs: {runs}")
    print(f"[Info] Max time per run: {time_per_run:.0f} seconds")
    print(f"[Info] Max stagnation: {max_stagnation}")

    for run_index in range(1, runs + 1):
        scheduler.rng = random.Random(seed_rng.randint(1, 1_000_000))
        solution, iterations, stagnation = scheduler.improve(
            base_schedule,
            time_limit_sec=time_per_run,
            max_stagnation=max_stagnation,
        )
        solution = max(solution, Solution(base_schedule, base_score), key=lambda item: item.total_score)

        score = int(solution.total_score)
        output_path = output_dir / f"run{run_index}_{score}.json"
        save_solution(solution, output_path)

        row = {
            "run": run_index,
            "base_file": base_path.name,
            "before_score": int(base_score),
            "after_score": score,
            "improvement": score - int(base_score),
            "iterations": iterations,
            "stagnation": stagnation,
            "output": str(output_path.relative_to(Path(__file__).resolve().parents[1])),
        }
        summary.append(row)
        print(f"Run {run_index}/{runs}: {row['before_score']} -> {row['after_score']} | Saved: {output_path.name}")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=4), encoding="utf-8")
    print(f"\n[Info] Summary saved: {summary_path}")
    return summary
