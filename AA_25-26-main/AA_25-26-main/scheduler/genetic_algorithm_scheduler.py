import bisect
import random
import time

from models.solution import Solution
from utils.algorithm_utils import AlgorithmUtils
from utils.utils import Utils


REPAIR_TOP_K = 3


class GeneticAlgorithmScheduler:
    def __init__(
        self,
        instance_data,
        initial_schedule=None,
        time_limit_sec=300.0,
        population_size=50,
        tournament_size=3,
        elite_size=2,
        crossover_rate=0.85,
        mutation_rate=0.35,
        max_generations=10000,
        candidate_pool_size=700,
        repair_random_rate=0.20,
        seed=None,
        stagnation_limit=5,
        min_runtime_before_stop_sec=None,
    ):
        self.instance = instance_data
        self.initial_schedule = list(initial_schedule) if initial_schedule else None
        self.time_limit = time_limit_sec
        self.pop_size = population_size
        self.tournament_size = tournament_size
        self.elite_size = elite_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.repair_random_rate = repair_random_rate
        self.rng = random.Random(seed if seed is not None else random.randint(1, 1000000))
        self.start_time = 0.0
        self.stagnation_limit = stagnation_limit
        self.min_runtime_before_stop_sec = (
            min_runtime_before_stop_sec
            if min_runtime_before_stop_sec is not None
            else min(120.0, self.time_limit * 0.4)
        )

        self.f_cache = {}
        self.add_cache = {}
        self.quality_cache = {}
        self.initial_population_scores = []
        self.generations_done = 0

        programs = []
        for channel in self.instance.channels:
            for prog in channel.programs:
                prog.channel_id = getattr(prog, "channel_id", channel.channel_id) or channel.channel_id
                prog.unique_id = getattr(prog, "unique_id", None) or getattr(prog, "program_id", "")
                prog.score = getattr(prog, "score", getattr(prog, "fitness", 0))
                programs.append(prog)

        Utils.set_current_instance(self.instance)
        self.max_bonus_by_genre = AlgorithmUtils.build_max_bonus_by_genre(self.instance)
        self.input_overlap_ids = AlgorithmUtils.compute_input_overlap_ids(self.instance)
        self.tail_size = max(3, int(getattr(self.instance, "max_consecutive_genre", 3)))
        self.progs_by_start = {}
        for prog in programs:
            if getattr(prog, "unique_id", None) in self.input_overlap_ids:
                continue
            self.progs_by_start.setdefault(Utils.get_start(prog), []).append(prog)

        for start, grouped_programs in self.progs_by_start.items():
            self.progs_by_start[start] = sorted(
                grouped_programs,
                key=lambda prog: AlgorithmUtils.get_fill_candidate_value(prog, self.max_bonus_by_genre),
                reverse=True,
            )[:candidate_pool_size]

        self.start_times = sorted(self.progs_by_start)

    def _elapsed_time(self):
        return time.perf_counter() - self.start_time

    def _time_left(self):
        return self.time_limit - self._elapsed_time()

    def _fitness(self, sched):
        filtered = AlgorithmUtils.filter_valid_schedule(sched, self.instance, self.input_overlap_ids)
        key = Utils.schedule_key(filtered)
        if key not in self.f_cache:
            self.f_cache[key] = AlgorithmUtils.score_filtered_schedule(filtered, self.instance)
        return self.f_cache[key]

    def _can_add(self, sched, item):
        tail = sched[-self.tail_size :]
        key = (
            Utils.get_uid(item),
            getattr(item, "channel_id", None),
            Utils.get_start(item),
            Utils.get_end(item),
            tuple(
                (
                    Utils.get_uid(prev),
                    getattr(prev, "channel_id", None),
                    Utils.get_start(prev),
                    Utils.get_end(prev),
                )
                for prev in tail
            ),
        )
        if key not in self.add_cache:
            self.add_cache[key] = self._can_append(sched, item)
        return self.add_cache[key]

    def _can_append(self, sched, item):
        if Utils.get_uid(item) in self.input_overlap_ids:
            return False

        program = Utils.get_program_by_unique_id(self.instance, Utils.get_uid(item))
        if not program:
            return False

        start = Utils.get_start(item)
        end = Utils.get_end(item)
        length = end - start
        full_length = Utils.get_end(program) - Utils.get_start(program)

        if start < self.instance.opening_time or end > self.instance.closing_time or length <= 0:
            return False
        if sched and start < Utils.get_end(sched[-1]):
            return False

        if full_length >= self.instance.min_duration and length < self.instance.min_duration:
            return False
        if full_length < self.instance.min_duration and length != full_length:
            return False

        for block in self.instance.priority_blocks:
            if start < block.end and end > block.start and item.channel_id not in block.allowed_channels:
                return False

        genre = getattr(program, "genre", "")
        streak = 1
        for prev in reversed(sched):
            prev_program = Utils.get_program_by_unique_id(self.instance, Utils.get_uid(prev))
            if not prev_program or getattr(prev_program, "genre", "") != genre:
                break
            streak += 1

        return streak <= self.instance.max_consecutive_genre

    def _quality(self, sched_item):
        uid = Utils.get_uid(sched_item)
        if uid not in self.quality_cache:
            self.quality_cache[uid] = AlgorithmUtils.get_segment_quality(
                self.instance,
                sched_item,
                self.input_overlap_ids,
            )
        return self.quality_cache[uid]

    def _best_fill_candidate(self, sched, used_ids, cursor, end_time):
        top_candidates = []

        left = bisect.bisect_left(self.start_times, cursor)
        right = bisect.bisect_right(self.start_times, end_time)

        for start in self.start_times[left:right]:
            if self._time_left() <= 0:
                break

            for prog in self.progs_by_start[start]:
                if self._time_left() <= 0:
                    break

                uid = getattr(prog, "unique_id", None)
                end = Utils.get_end(prog)

                if uid in used_ids or end > end_time:
                    continue

                base_value = AlgorithmUtils.get_fill_candidate_value(prog, self.max_bonus_by_genre)
                item = Utils.make_schedule(
                    prog.channel_id,
                    prog.program_id,
                    start,
                    end,
                    base_value,
                    uid,
                )

                if not self._can_add(sched, item):
                    continue

                wait_penalty = 0.03 * max(0, start - cursor)
                switch_penalty = self.instance.switch_penalty if sched and sched[-1].channel_id != prog.channel_id else 0
                value = base_value - wait_penalty - switch_penalty

                top_candidates.append((value, item))
                top_candidates.sort(key=lambda pair: pair[0], reverse=True)
                top_candidates = top_candidates[:REPAIR_TOP_K]

        if not top_candidates:
            return None

        if len(top_candidates) > 1 and self.rng.random() < self.repair_random_rate:
            return self.rng.choice(top_candidates)[1]

        return top_candidates[0][1]

    def _fill_interval(self, sched, used_ids, start_time, end_time):
        cursor = start_time

        while cursor < end_time and self._time_left() > 0:
            item = self._best_fill_candidate(sched, used_ids, cursor, end_time)
            if item is None:
                break

            uid = Utils.get_uid(item)
            sched.append(item)
            used_ids.add(uid)
            cursor = Utils.get_end(item)

        return cursor

    def _repair(self, sched):
        sched = AlgorithmUtils.filter_valid_schedule(sched, self.instance, self.input_overlap_ids)
        new_sched = []
        used_ids = {Utils.get_uid(item) for item in sched}
        cursor = self.instance.opening_time

        for item in sorted(sched, key=Utils.sort_schedule_item):
            if self._time_left() <= 0:
                break

            cursor = self._fill_interval(new_sched, used_ids, cursor, Utils.get_start(item))
            if Utils.get_start(item) >= cursor and Utils.get_uid(item) not in self.input_overlap_ids and self._can_add(new_sched, item):
                new_sched.append(item)
                cursor = Utils.get_end(item)

        self._fill_interval(new_sched, used_ids, cursor, self.instance.closing_time)
        return AlgorithmUtils.filter_valid_schedule(new_sched, self.instance, self.input_overlap_ids)

    def _select_population(self, population):
        ranked = sorted(((ind, self._fitness(ind)) for ind in population), key=lambda pair: pair[1], reverse=True)
        elites = [list(ind) for ind, _ in ranked[: self.elite_size]]
        mating_pool = [ind for ind, _ in ranked[: max(2, len(ranked) // 2)]]
        return elites, mating_pool, ranked[0][1], ranked[0][0]

    def _pick_parent(self, mating_pool):
        size = min(self.tournament_size, len(mating_pool))
        contenders = self.rng.sample(mating_pool, size) if len(mating_pool) > size else list(mating_pool)
        return max(contenders, key=self._fitness)

    def _cross(self, parent1, parent2):
        if self.rng.random() > self.crossover_rate:
            return list(max(parent1, parent2, key=self._fitness))

        lower = self.instance.opening_time + 30
        upper = self.instance.closing_time - 30
        cut_point = self.rng.randint(lower, upper) if lower < upper else (self.instance.opening_time + self.instance.closing_time) // 2

        child = [item for item in parent1 if Utils.get_end(item) <= cut_point]
        used_ids = {Utils.get_uid(item) for item in child}

        for item in parent2:
            uid = Utils.get_uid(item)
            if Utils.get_start(item) < cut_point or uid in used_ids or uid in self.input_overlap_ids:
                continue
            if self._can_add(child, item):
                child.append(item)
                used_ids.add(uid)

        return self._repair(child)

    def _mutate(self, sched, aggressive=False, force=False):
        if not sched or (not force and not aggressive and self.rng.random() > self.mutation_rate):
            return list(sched)

        child = list(sched)

        if aggressive and len(child) >= 5 and self.rng.random() < 0.15:
            width = self.rng.randint(2, min(3, len(child)))
            start = min(
                range(len(child) - width + 1),
                key=lambda i: sum(self._quality(item) for item in child[i:i + width]) / width,
            )
            del child[start:start + width]
            return self._repair(child)

        remove_count = self.rng.randint(2, 4) if aggressive else 1

        for _ in range(remove_count):
            if not child:
                break
            index = min(
                range(len(child)),
                key=lambda i: self._quality(child[i]) if self.rng.random() < 0.75 else self.rng.random(),
            )
            child.pop(index)

        return self._repair(child)

    def _make_child(self, parent1, parent2, aggressive=False):
        stronger_parent = max(parent1, parent2, key=self._fitness)
        child = self._cross(parent1, parent2)

        if aggressive or self.rng.random() < self.mutation_rate:
            child = self._mutate(child, aggressive=aggressive)

        if aggressive:
            return max(child, self._mutate(stronger_parent, aggressive=True), key=self._fitness)
        return max(child, stronger_parent, key=self._fitness)

    def _base_schedule(self):
        if self.initial_schedule:
            return AlgorithmUtils.filter_valid_schedule(self.initial_schedule, self.instance, self.input_overlap_ids)
        return self._repair([])

    def _initial_population(self, base):
        population = [list(base)]
        while len(population) < self.pop_size and self._time_left() > 0:
            aggressive = len(population) % 4 == 0
            population.append(self._mutate(base, aggressive=aggressive, force=True))
        self.initial_population_scores = [self._fitness(ind) for ind in population]
        return population

    def generate_solution(self):
        self.start_time = time.perf_counter()
        base = self._base_schedule()
        population = self._initial_population(base)

        _, _, best_score, best_sched = self._select_population(population)
        stagnation = 0
        generation = 0

        while generation < self.max_generations and self._time_left() > 0:
            elites, mating_pool, current_score, current_best = self._select_population(population)

            if current_score > best_score:
                best_score = current_score
                best_sched = list(current_best)
            else:
                stagnation += 1

            aggressive = stagnation >= 8
            population = list(elites)
            seen_keys = {Utils.schedule_key(ind) for ind in population}

            while len(population) < self.pop_size and self._time_left() > 0:
                if aggressive and len(population) == len(elites):
                    candidate = self._mutate(best_sched, aggressive=True)
                    population.append(candidate)
                    seen_keys.add(Utils.schedule_key(candidate))
                    continue

                parent1 = self._pick_parent(mating_pool)
                parent2 = self._pick_parent(mating_pool)
                candidate = self._make_child(parent1, parent2, aggressive)
                candidate_key = Utils.schedule_key(candidate)

                if candidate_key in seen_keys and self._time_left() > 0:
                    candidate = self._mutate(best_sched, aggressive=True)
                    candidate_key = Utils.schedule_key(candidate)

                population.append(candidate)
                seen_keys.add(candidate_key)

            generation += 1

            if (
                stagnation >= self.stagnation_limit
                and self._elapsed_time() >= self.min_runtime_before_stop_sec
            ):
                break

        self.generations_done = generation
        final_sched = AlgorithmUtils.filter_valid_schedule(best_sched or [], self.instance, self.input_overlap_ids)

        if self._time_left() > 0.2:
            improved_sched = self._repair(final_sched)
            final_sched = max(final_sched, improved_sched, key=self._fitness)

        return Solution(final_sched, self._fitness(final_sched))
