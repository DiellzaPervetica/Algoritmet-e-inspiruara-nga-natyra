import random
import time
from typing import Dict, List, Set, Tuple

from models.instance_data import InstanceData
from models.solution import Solution
from scheduler.branch_and_bound_scheduler import BranchAndBoundScheduler
from utils.algorithm_utils import AlgorithmUtils
from utils.scheduler_utils import SchedulerUtils
from utils.utils import Utils


class GeneticAlgorithmScheduler:
    def __init__(
        self,
        instance_data: InstanceData,
        initial_schedule: List[object] | None = None,
        time_limit_sec: float = 300.0,
        population_size: int = 10,
        mutation_rate: float = 0.30,
        seed: int | None = None,
    ):
        self.instance = instance_data
        self.initial_schedule = list(initial_schedule) if initial_schedule else None
        self.time_limit = max(1.0, float(time_limit_sec))
        self.pop_size = max(2, int(population_size))
        self.mut_rate = mutation_rate
        self.seed = seed if seed is not None else random.randint(1, 1000000)
        self.rng = random.Random(self.seed)

        self.f_cache: Dict[Tuple, int] = {}
        self.v_cache: Dict[Tuple, Set[int]] = {}
        self.progs = []
        self.p_dict = {}

        for channel in self.instance.channels:
            for prog in channel.programs:
                prog.channel_id = getattr(prog, "channel_id", channel.channel_id) or channel.channel_id
                prog.unique_id = getattr(prog, "unique_id", None) or getattr(prog, "program_id", f"p_{self.rng.randint(1000, 9999)}")
                prog.score = getattr(prog, "score", getattr(prog, "fitness", 0))
                self.progs.append(prog)
                self.p_dict[prog.unique_id] = prog

        Utils.set_current_instance(self.instance)
        self.progs.sort(key=Utils.get_start)
        self.max_bonus_by_genre = AlgorithmUtils.build_max_bonus_by_genre(self.instance)
        self.input_overlap_ids = AlgorithmUtils.compute_input_overlap_ids(self.instance)
        self.progs_by_score = sorted(
            self.progs,
            key=lambda prog: (
                AlgorithmUtils.get_fill_candidate_value(prog, self.max_bonus_by_genre),
                Utils.get_end(prog) - Utils.get_start(prog),
            ),
            reverse=True,
        )

    def _fitness(self, sched):
        filtered = AlgorithmUtils.filter_valid_schedule(sched, self.instance, self.input_overlap_ids)
        key = Utils.schedule_key(filtered)
        if key not in self.f_cache:
            self.f_cache[key] = AlgorithmUtils.score_filtered_schedule(filtered, self.instance)
        return self.f_cache[key]

    def _valid_channels(self, sched, nxt_start):
        key = (
            nxt_start,
            tuple(
                (
                    Utils.get_uid(item),
                    getattr(item, "channel_id", None),
                    Utils.get_start(item),
                    Utils.get_end(item),
                )
                for item in sched[-3:]
            ),
        )
        if key not in self.v_cache:
            self.v_cache[key] = {
                self.instance.channels[i].channel_id
                for i in SchedulerUtils.get_valid_schedules(sched, self.instance, nxt_start)
            }
        return self.v_cache[key]

    def _fill_interval(self, new_sched, used_ids, start_time, end_time):
        cursor = start_time
        for prog in self.progs_by_score:
            uid = getattr(prog, "unique_id", None)
            if uid in used_ids or uid in self.input_overlap_ids:
                continue
            if Utils.get_start(prog) < cursor or Utils.get_end(prog) > end_time:
                continue
            if prog.channel_id not in self._valid_channels(new_sched, Utils.get_start(prog)):
                continue
            new_sched.append(
                Utils.make_schedule(
                    prog.channel_id,
                    prog.program_id,
                    Utils.get_start(prog),
                    Utils.get_end(prog),
                    AlgorithmUtils.get_fill_candidate_value(prog, self.max_bonus_by_genre),
                    uid,
                )
            )
            used_ids.add(uid)
            cursor = Utils.get_end(prog)
        return cursor

    def _fill_gaps(self, sched):
        sched = AlgorithmUtils.filter_valid_schedule(sched, self.instance, self.input_overlap_ids)
        new_sched = []
        used_ids = {Utils.get_uid(item) for item in sched}
        cursor = self.instance.opening_time

        for item in sched:
            cursor = self._fill_interval(new_sched, used_ids, cursor, Utils.get_start(item))
            if Utils.get_start(item) >= cursor and Utils.get_uid(item) not in self.input_overlap_ids:
                if item.channel_id in self._valid_channels(new_sched, Utils.get_start(item)):
                    new_sched.append(item)
                    cursor = Utils.get_end(item)

        self._fill_interval(new_sched, used_ids, cursor, self.instance.closing_time)
        return sorted(new_sched, key=Utils.sort_schedule_item)

    def _select(self, population):
        ranked = sorted(((ind, self._fitness(ind)) for ind in population), key=lambda pair: pair[1], reverse=True)
        elites = [ind for ind, _ in ranked[: min(2, len(ranked))]]
        mating_pool = [ind for ind, _ in ranked[: max(2, len(ranked) // 2)]]
        return elites, mating_pool, ranked[0][1], ranked[0][0]

    def _pick_parent(self, mating_pool):
        size = min(3, len(mating_pool))
        contenders = self.rng.sample(mating_pool, size) if len(mating_pool) > size else list(mating_pool)
        return max(contenders, key=self._fitness)

    def _cross(self, parent1, parent2):
        lower = self.instance.opening_time + 60
        upper = self.instance.closing_time - 60
        cut_point = self.rng.randint(lower, upper) if lower < upper else (self.instance.opening_time + self.instance.closing_time) // 2

        child = [item for item in parent1 if Utils.get_end(item) <= cut_point]
        used_ids = {Utils.get_uid(item) for item in child}
        for item in parent2:
            uid = Utils.get_uid(item)
            if Utils.get_start(item) < cut_point or uid in used_ids or uid in self.input_overlap_ids:
                continue
            if item.channel_id in self._valid_channels(child, Utils.get_start(item)):
                child.append(item)
                used_ids.add(uid)

        return self._fill_gaps(child)

    def _mut(self, sched, aggressive=False):
        if not sched or (not aggressive and self.rng.random() > self.mut_rate):
            return sched

        sched = list(sched)
        if len(sched) == 1:
            sched.clear()
        elif aggressive and len(sched) > 2 and self.rng.random() < 0.35:
            start_idx = self.rng.randrange(len(sched))
            width = self.rng.randint(1, min(3, len(sched) - start_idx))
            del sched[start_idx:start_idx + width]
        remove_count = self.rng.randint(1, 3) if aggressive else 1
        for _ in range(remove_count):
            if not sched:
                break
            worst_idx = min(
                range(len(sched)),
                key=lambda i: AlgorithmUtils.get_segment_quality(self.instance, sched[i], self.input_overlap_ids) if self.rng.random() < 0.7 else self.rng.random(),
            )
            sched.pop(worst_idx)

        return self._fill_gaps(sched)

    def _make_next_candidate(self, parent1, parent2, aggressive_mode=False):
        stronger_parent = max([parent1, parent2], key=self._fitness)
        child = max([self._cross(parent1, parent2), self._cross(parent2, parent1)], key=self._fitness)
        if aggressive_mode or self.rng.random() < self.mut_rate:
            child = self._mut(child, aggressive=aggressive_mode)

        fallback = stronger_parent
        if aggressive_mode:
            fallback = max([stronger_parent, self._mut(list(stronger_parent), aggressive=True)], key=self._fitness)
        return max([fallback, child], key=self._fitness)

    def _get_base_schedule(self):
        candidates = []
        if self.initial_schedule:
            candidates.append(AlgorithmUtils.filter_valid_schedule(self.initial_schedule, self.instance, self.input_overlap_ids))

        bnb = BranchAndBoundScheduler(
            self.instance,
            time_limit_sec=min(5.0, max(1.0, self.time_limit * 0.2)),
            randomized_restarts=4,
            seed=self.seed,
            verbose=False,
        )
        sol = bnb.generate_solution()
        if sol:
            candidates.append(AlgorithmUtils.filter_valid_schedule(sol.scheduled_programs, self.instance, self.input_overlap_ids))
        return max(candidates, key=self._fitness) if candidates else []

    def generate_solution(self) -> Solution:
        base = self._get_base_schedule()
        if not base:
            return Solution([], 0)

        pop = [list(base)] + [self._mut(list(base), aggressive=True) for _ in range(self.pop_size - 1)]
        _, _, best_score, best_sched = self._select(pop)

        start_time = time.time()
        stagnation = 0
        while time.time() - start_time < self.time_limit:
            elites, mating_pool, current_score, current_best = self._select(pop)
            if current_score > best_score:
                best_score, best_sched, stagnation = current_score, current_best, 0
            else:
                stagnation += 1

            aggressive_mode = stagnation >= 10
            pop = list(elites)
            while len(pop) < self.pop_size:
                if aggressive_mode and len(pop) == len(elites):
                    pop.append(self._mut(list(best_sched or base), aggressive=True))
                    continue
                pop.append(self._make_next_candidate(self._pick_parent(mating_pool), self._pick_parent(mating_pool), aggressive_mode))

        best_sched = AlgorithmUtils.filter_valid_schedule(best_sched or [], self.instance, self.input_overlap_ids)
        improved = self._fill_gaps(best_sched)
        final_sched = max([best_sched, improved], key=self._fitness)
        return Solution(final_sched, self._fitness(final_sched))