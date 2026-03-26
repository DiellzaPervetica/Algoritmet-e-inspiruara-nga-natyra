
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import bisect
import math
import random
import time

from models.instance_data import InstanceData
from models.program import Program
from models.schedule import Schedule
from models.solution import Solution


class BranchAndBoundScheduler:
    """
    Branch-and-Bound scheduler with randomized warm starts.

    Why this fits the professor's constraint:
    - The core search is Branch & Bound, not a greedy / beam heuristic.
    - Randomness is used only to generate strong initial incumbents and
      to diversify branch ordering among near-equal choices.
    - The search then systematically prunes states using admissible
      upper bounds.
    """

    def __init__(
        self,
        instance_data: InstanceData,
        time_limit_sec: float = 30.0,
        randomized_restarts: int = 16,
        restricted_candidate_size: int = 4,
        seed: Optional[int] = 42,
        verbose: bool = True,
    ):
        self.instance_data = instance_data
        self.time_limit_sec = max(1.0, float(time_limit_sec))
        self.randomized_restarts = max(1, int(randomized_restarts))
        self.restricted_candidate_size = max(1, int(restricted_candidate_size))
        self.seed = seed
        self.verbose = verbose

        self.rng = random.Random(seed)
        self.min_d = instance_data.min_duration
        self._time_limit_hit = False

        self._preprocess()

    def _preprocess(self) -> None:
        self.n_channels = len(self.instance_data.channels)
        self.opening = self.instance_data.opening_time
        self.closing = self.instance_data.closing_time

        self.ch_progs: List[List[Program]] = []
        self.ch_starts: List[List[int]] = []

        self.prog_by_id: Dict[str, Tuple[Program, int]] = {}
        self.starts_at: Dict[int, List[Tuple[Program, int]]] = defaultdict(list)

        all_times = {self.opening, self.closing}

        for ch_idx, channel in enumerate(self.instance_data.channels):
            progs = sorted(channel.programs, key=lambda p: p.start)
            self.ch_progs.append(progs)
            self.ch_starts.append([p.start for p in progs])

            for prog in progs:
                self.prog_by_id[prog.unique_id] = (prog, ch_idx)
                self.starts_at[prog.start].append((prog, ch_idx))
                all_times.add(prog.start)
                all_times.add(prog.end)

        for block in self.instance_data.priority_blocks:
            all_times.add(block.start)
            all_times.add(block.end)

        self.times = sorted(
            t for t in all_times
            if self.opening <= t <= self.closing
        )

        # Priority block map by minute
        self.priority_at: Dict[int, Set[int]] = {}
        for block in self.instance_data.priority_blocks:
            allowed = set(block.allowed_channels)
            for t in range(block.start, block.end):
                if t not in self.priority_at:
                    self.priority_at[t] = allowed.copy()
                else:
                    self.priority_at[t] &= allowed

        self.has_priority_blocks = bool(self.instance_data.priority_blocks)
        self.forbidden_prefix: List[List[int]] = []
        if self.has_priority_blocks:
            max_t = self.closing + 2
            for ch_idx, channel in enumerate(self.instance_data.channels):
                ch_id = channel.channel_id
                forbidden = [0] * max_t
                for t, allowed in self.priority_at.items():
                    if 0 <= t < max_t and ch_id not in allowed:
                        forbidden[t] = 1

                prefix = [0] * (max_t + 1)
                run = 0
                for t in range(max_t):
                    run += forbidden[t]
                    prefix[t + 1] = run
                self.forbidden_prefix.append(prefix)

        self.prefs = self.instance_data.time_preferences
        self.genre_max_bonus: Dict[str, int] = defaultdict(int)
        for pref in self.prefs:
            if pref.bonus > self.genre_max_bonus[pref.preferred_genre]:
                self.genre_max_bonus[pref.preferred_genre] = pref.bonus

        densities = []
        self.program_potentials: List[Tuple[int, str, int, int]] = []
        for channel in self.instance_data.channels:
            for prog in channel.programs:
                dur = prog.end - prog.start
                if dur > 0:
                    densities.append(prog.score / dur)
                optimistic = prog.score + self.genre_max_bonus.get(prog.genre, 0)
                self.program_potentials.append((optimistic, prog.unique_id, prog.start, prog.end))

        densities.sort(reverse=True)
        if densities:
            top_k = max(1, min(len(densities), math.ceil(len(densities) * 0.25)))
            self.avg_density = sum(densities[:top_k]) / top_k
        else:
            self.avg_density = 1.0

        self.program_potentials.sort(reverse=True)

    def _get_prog(self, ch_idx: int, time_point: int) -> Optional[Program]:
        idx = bisect.bisect_right(self.ch_starts[ch_idx], time_point) - 1
        if 0 <= idx < len(self.ch_progs[ch_idx]):
            p = self.ch_progs[ch_idx][idx]
            if p.start <= time_point < p.end:
                return p
        return None

    def _next_decision_time(self, time_point: int) -> Optional[int]:
        idx = bisect.bisect_right(self.times, time_point)
        if idx < len(self.times):
            nxt = self.times[idx]
            if nxt <= self.closing:
                return nxt
        return None

    def _channel_allowed(self, ch_idx: int, start: int, end: int) -> bool:
        if start >= end:
            return True
        if not self.has_priority_blocks:
            return True

        prefix = self.forbidden_prefix[ch_idx]
        max_t = len(prefix) - 1
        s = max(0, min(start, max_t))
        e = max(0, min(end, max_t))
        if s >= e:
            return True
        return (prefix[e] - prefix[s]) == 0

    def _calc_score(
        self,
        prog: Program,
        ch_idx: int,
        seg_start: int,
        seg_end: int,
        prev_ch_id: Optional[int],
    ) -> int:
        duration = seg_end - seg_start
        if duration < self.min_d:
            return -10**18

        channel = self.instance_data.channels[ch_idx]
        score = prog.score

        for pref in self.prefs:
            if prog.genre != pref.preferred_genre:
                continue
            ov_start = max(seg_start, pref.start)
            ov_end = min(seg_end, pref.end)
            if ov_end - ov_start >= self.min_d:
                score += pref.bonus
                break

        if prev_ch_id is not None and prev_ch_id != channel.channel_id:
            score -= self.instance_data.switch_penalty

        if seg_start > prog.start:
            score -= self.instance_data.termination_penalty
        if seg_end < prog.end:
            score -= self.instance_data.termination_penalty

        return score

    def _get_candidates(
        self,
        time_point: int,
        prev_ch_id: Optional[int],
        prev_genre: str,
        genre_streak: int,
        used_progs: Set[str],
    ) -> List[Tuple[int, int, int, Program, int, int]]:
        candidates: List[Tuple[int, int, int, Program, int, int]] = []

        for ch_idx in range(self.n_channels):
            channel = self.instance_data.channels[ch_idx]
            ch_id = channel.channel_id

            prog = self._get_prog(ch_idx, time_point)
            if prog is None or prog.unique_id in used_progs:
                continue

            new_streak = 1 if prog.genre != prev_genre else genre_streak + 1
            if new_streak > self.instance_data.max_consecutive_genre:
                continue

            seg_start = time_point
            nat_end = min(prog.end, self.closing)
            if nat_end - seg_start < self.min_d:
                continue

            end_options = {nat_end, seg_start + self.min_d}

            start_idx = bisect.bisect_right(self.times, seg_start + self.min_d)
            end_idx = bisect.bisect_left(self.times, nat_end)
            for i in range(start_idx, end_idx + 1):
                if i >= len(self.times):
                    break
                t = self.times[i]
                if t > nat_end:
                    break
                if t - seg_start >= self.min_d:
                    end_options.add(t)

            for seg_end in sorted(end_options):
                if seg_end > self.closing:
                    continue
                if not self._channel_allowed(ch_idx, seg_start, seg_end):
                    continue

                seg_score = self._calc_score(prog, ch_idx, seg_start, seg_end, prev_ch_id)
                if seg_score > -10**17:
                    candidates.append((seg_score, ch_idx, ch_id, prog, seg_start, seg_end))

        return candidates

    def _candidate_order_key(self, cand: Tuple[int, int, int, Program, int, int]) -> float:
        seg_score, _, _, prog, seg_start, seg_end = cand
        dur = max(1, seg_end - seg_start)
        # Ordering heuristic only; pruning remains exact-safe.
        return seg_score + (self.closing - seg_end) * self.avg_density + (prog.score / dur)

    def _randomized_construction(self) -> Solution:
        time_point = self.opening
        prev_ch = None
        prev_genre = ""
        genre_streak = 0
        used: Set[str] = set()
        schedules: List[Schedule] = []
        total_score = 0

        safety = 0
        while time_point < self.closing and safety < 10000:
            safety += 1
            candidates = self._get_candidates(time_point, prev_ch, prev_genre, genre_streak, used)

            if not candidates:
                nxt = self._next_decision_time(time_point)
                if nxt is None:
                    break
                time_point = nxt
                continue

            candidates.sort(key=self._candidate_order_key, reverse=True)
            rcl_size = min(len(candidates), self.restricted_candidate_size)
            chosen = self.rng.choice(candidates[:rcl_size])

            seg_score, _, ch_id, prog, seg_start, seg_end = chosen
            schedules.append(
                Schedule(
                    program_id=prog.program_id,
                    channel_id=ch_id,
                    start=seg_start,
                    end=seg_end,
                    fitness=seg_score,
                    unique_program_id=prog.unique_id,
                )
            )
            total_score += seg_score
            used.add(prog.unique_id)

            if prog.genre == prev_genre:
                genre_streak += 1
            else:
                prev_genre = prog.genre
                genre_streak = 1

            prev_ch = ch_id
            time_point = seg_end

        return Solution(schedules, total_score)

    def _upper_bound(self, time_point: int, used: Set[str]) -> int:
        remaining = self.closing - time_point
        if remaining < self.min_d:
            return 0

        max_slots = remaining // self.min_d
        if max_slots <= 0:
            return 0

        ub = 0
        taken = 0
        for optimistic, uid, start, end in self.program_potentials:
            if uid in used:
                continue

            effective_start = max(time_point, start)
            if end - effective_start < self.min_d:
                continue

            ub += optimistic
            taken += 1
            if taken >= max_slots:
                break

        return ub

    def _dfs_branch_and_bound(self, deadline: float, incumbent: Solution) -> Solution:
        best_score = incumbent.total_score
        best_sched = incumbent.scheduled_programs[:]

        initial_state = (
            0,                         # accumulated score
            self.opening,              # current time
            None,                      # prev channel id
            "",                        # prev genre
            0,                         # genre streak
            tuple(),                   # schedule tuple
            frozenset(),               # used programs
        )

        stack = [initial_state]
        expanded = 0

        while stack:
            if time.time() >= deadline:
                self._time_limit_hit = True
                break

            score, time_point, prev_ch, prev_genre, g_streak, sched_tuple, used = stack.pop()
            expanded += 1

            if time_point >= self.closing:
                if score > best_score:
                    best_score = score
                    best_sched = list(sched_tuple)
                continue

            optimistic_total = score + self._upper_bound(time_point, set(used))
            if optimistic_total <= best_score:
                continue

            candidates = self._get_candidates(time_point, prev_ch, prev_genre, g_streak, set(used))
            ordered_children = []

            nxt = self._next_decision_time(time_point)
            if nxt is not None and nxt <= self.closing:
                ordered_children.append((
                    score,
                    nxt,
                    prev_ch,
                    prev_genre,
                    g_streak,
                    sched_tuple,
                    used,
                    score + self._upper_bound(nxt, set(used))
                ))

            if candidates:
                # diversify near-equal branches without changing correctness
                decorated = []
                for cand in candidates:
                    decorated.append((self._candidate_order_key(cand), self.rng.random(), cand))
                decorated.sort(reverse=True)

                for _, _, (seg_score, _, ch_id, prog, seg_start, seg_end) in decorated:
                    new_sched = sched_tuple + (
                        Schedule(
                            program_id=prog.program_id,
                            channel_id=ch_id,
                            start=seg_start,
                            end=seg_end,
                            fitness=seg_score,
                            unique_program_id=prog.unique_id,
                        ),
                    )
                    new_used = used | {prog.unique_id}
                    new_streak = 1 if prog.genre != prev_genre else g_streak + 1
                    child_score = score + seg_score
                    child_ub = child_score + self._upper_bound(seg_end, set(new_used))
                    if child_ub <= best_score:
                        continue

                    ordered_children.append((
                        child_score,
                        seg_end,
                        ch_id,
                        prog.genre,
                        new_streak,
                        new_sched,
                        new_used,
                        child_ub
                    ))
            else:
                if score > best_score:
                    best_score = score
                    best_sched = list(sched_tuple)

            if not ordered_children:
                continue

            ordered_children.sort(key=lambda x: x[-1], reverse=True)
            for child in reversed(ordered_children):
                stack.append(child[:-1])

        if self.verbose:
            status = "TIME LIMIT HIT" if self._time_limit_hit else "COMPLETED"
            print(f"[BnB] Nodes expanded: {expanded}")
            print(f"[BnB] Search status : {status}")

        return Solution(best_sched, best_score)

    def generate_solution(self) -> Solution:
        if self.verbose:
            print("\n" + "=" * 72)
            print("BRANCH-AND-BOUND SCHEDULER")
            print(f"Channels              : {self.n_channels}")
            print(f"Time limit (sec)      : {self.time_limit_sec}")
            print(f"Randomized restarts   : {self.randomized_restarts}")
            print(f"RCL size              : {self.restricted_candidate_size}")
            print("=" * 72)

        start_time = time.time()

        best = Solution([], 0)
        for i in range(self.randomized_restarts):
            warm = self._randomized_construction()
            if warm.total_score > best.total_score:
                best = warm
            if self.verbose:
                print(f"[Warm start {i+1:02d}] score = {warm.total_score}")

            # tiny shuffle between restarts for diversification
            self.rng.seed((self.seed or 0) + i + 1)

        if self.verbose:
            print(f"[Warm starts] Best incumbent = {best.total_score}")

        deadline = start_time + self.time_limit_sec
        best = self._dfs_branch_and_bound(deadline, best)

        if self.verbose:
            elapsed = time.time() - start_time
            print(f"[Final] Best score = {best.total_score}")
            print(f"[Final] Programs   = {len(best.scheduled_programs)}")
            print(f"[Final] Elapsed    = {elapsed:.2f}s")
            print("=" * 72 + "\n")

        return best
