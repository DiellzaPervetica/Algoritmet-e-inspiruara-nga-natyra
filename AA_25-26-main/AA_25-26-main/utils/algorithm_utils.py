from typing import Dict, List, Set

from models.channel import Channel
from models.instance_data import InstanceData
from models.program import Program
from models.schedule import Schedule
from utils.utils import Utils


class AlgorithmUtils:

    @staticmethod
    def get_best_fit(scheduled_programs: List[Schedule], instance_data: InstanceData, schedule_time: int,
                     valid_channel_indexes: List[int]) ->  tuple[Channel, Program, int]:

        # returns best channel to pick at the time and what score will it provide if we switch to it

        max_score = 0
        best_channel = None
        best_program = None

        for channel_index in valid_channel_indexes:
            channel = instance_data.channels[channel_index]
            program = Utils.get_channel_program_by_time(channel, schedule_time)

            if not program:
                continue

            score = 0

            score += program.score
            score += AlgorithmUtils.get_time_preference_bonus(instance_data, program, schedule_time)
            score += AlgorithmUtils.get_switch_penalty(scheduled_programs, instance_data, channel)
            score += AlgorithmUtils.get_delay_penalty(scheduled_programs, instance_data, program, schedule_time)
            score += AlgorithmUtils.get_early_termination_penalty(scheduled_programs, instance_data, program, schedule_time)

            if score > max_score:
                max_score = score
                best_channel = channel
                best_program = program

        return best_channel, best_program, max_score

    @staticmethod
    def get_time_preference_bonus(instance_data: InstanceData, program: Program, schedule_time: int):
        """
        Calculate time preference bonus for a program.
        CRITICAL: The bonus only applies if the program overlaps with the preferred time window
        for AT LEAST the minimum duration D minutes.

        Per PDF specification: "In order to collect the bonus, the program must fall within
        the preferred interval with at least the minimum scheduled duration D."
        """
        score = 0
        for preference in instance_data.time_preferences:
            if program.genre == preference.preferred_genre:
                # Calculate overlap between program and preference interval
                overlap_start = max(program.start, preference.start)
                overlap_end = min(program.end, preference.end)
                overlap_duration = overlap_end - overlap_start

                # Only award bonus if overlap is at least min_duration
                if overlap_duration >= instance_data.min_duration:
                    score += preference.bonus

        return score

    @staticmethod
    def get_switch_penalty(scheduled_programs: List[Schedule], instance_data: InstanceData, channel: Channel):
        penalty = 0
        if not scheduled_programs:
            return penalty

        last_schedule = scheduled_programs[-1]
        if last_schedule.channel_id != channel.channel_id:
            penalty -= instance_data.switch_penalty

        return penalty

    @staticmethod
    def get_delay_penalty(scheduled_programs: List[Schedule], instance_data: InstanceData, program: Program,
                          schedule_time: int):
        """
        Penalty for switching to a program after its scheduled start time.
        Since we now always schedule programs at their original times, this should not apply.
        """
        penalty = 0
        # No delay penalty - we always schedule programs at their original start time
        return penalty

    @staticmethod
    def get_early_termination_penalty(scheduled_programs: List[Schedule], instance_data: InstanceData, program: Program,
                                      schedule_time: int):
        """
        Penalty for terminating the previous program before its scheduled end time.
        This occurs when we switch to a new program while the previous one is still running.
        Since we now prevent overlaps, this checks if switching to the new program would
        cut off the previous one early.
        """
        penalty = 0
        if not scheduled_programs:
            return penalty

        last_schedule = scheduled_programs[-1]
        
        # If the new program starts before the previous program's natural end
        # (and we're not continuing the same program), we're cutting it short
        if last_schedule.unique_program_id != program.unique_id and program.start < last_schedule.end:
            penalty -= instance_data.termination_penalty

        return penalty

    @staticmethod
    def build_max_bonus_by_genre(instance_data: InstanceData) -> Dict[str, int]:
        bonuses: Dict[str, int] = {}
        for channel in instance_data.channels:
            for program in channel.programs:
                genre = getattr(program, "genre", "")
                current = bonuses.get(genre, 0)
                for pref in instance_data.time_preferences:
                    if genre == pref.preferred_genre and pref.bonus > current:
                        current = pref.bonus
                bonuses[genre] = current
        return bonuses

    @staticmethod
    def compute_input_overlap_ids(instance_data: InstanceData) -> Set[str]:
        overlapped: Set[str] = set()
        for channel in instance_data.channels:
            programs = sorted(channel.programs, key=Utils.sort_schedule_item)
            active = []
            for program in programs:
                start = Utils.get_start(program)
                end = Utils.get_end(program)
                active = [prev for prev in active if Utils.get_end(prev) > start]
                for prev in active:
                    if start < Utils.get_end(prev) and end > Utils.get_start(prev):
                        overlapped.update([getattr(prev, "unique_id", None), getattr(program, "unique_id", None)])
                active.append(program)
        overlapped.discard(None)
        return overlapped

    @staticmethod
    def get_segment_bonus(instance_data: InstanceData, sched_item, program: Program):
        score = 0
        seg_start = Utils.get_start(sched_item)
        seg_end = Utils.get_end(sched_item)
        for pref in instance_data.time_preferences:
            if getattr(program, "genre", "") != pref.preferred_genre:
                continue
            overlap_start = max(seg_start, pref.start)
            overlap_end = min(seg_end, pref.end)
            if overlap_end - overlap_start >= instance_data.min_duration:
                score += pref.bonus
        return score

    @staticmethod
    def get_segment_quality(instance_data: InstanceData, sched_item, input_overlap_ids: Set[str]):
        program = Utils.get_program_by_unique_id(instance_data, Utils.get_uid(sched_item))
        if not program:
            return float("-inf")

        seg_len = Utils.get_end(sched_item) - Utils.get_start(sched_item)
        if seg_len <= 0:
            return float("-inf")

        quality = getattr(program, "score", 0) + AlgorithmUtils.get_segment_bonus(instance_data, sched_item, program)
        if Utils.get_start(sched_item) > Utils.get_start(program):
            quality -= instance_data.termination_penalty
        if Utils.get_end(sched_item) < Utils.get_end(program):
            quality -= instance_data.termination_penalty
        if getattr(program, "unique_id", None) in input_overlap_ids:
            quality -= 3 * instance_data.termination_penalty
        return quality

    @staticmethod
    def get_fill_candidate_value(program: Program, max_bonus_by_genre: Dict[str, int]):
        return getattr(program, "score", 0) + max_bonus_by_genre.get(getattr(program, "genre", ""), 0)

    @staticmethod
    def filter_valid_schedule(sched, instance_data: InstanceData, input_overlap_ids: Set[str]):
        sched = sorted(sched, key=Utils.sort_schedule_item)
        if not sched:
            return []

        valid_mask = [True] * len(sched)
        for i, sched_item in enumerate(sched):
            program = Utils.get_program_by_unique_id(instance_data, Utils.get_uid(sched_item))
            if not program:
                valid_mask[i] = False
                continue

            seg_start = Utils.get_start(sched_item)
            seg_end = Utils.get_end(sched_item)
            seg_len = seg_end - seg_start
            full_len = Utils.get_end(program) - Utils.get_start(program)

            invalid_len = (full_len >= instance_data.min_duration and seg_len < instance_data.min_duration) or (
                full_len < instance_data.min_duration and seg_len != full_len
            )
            invalid_bounds = seg_start < instance_data.opening_time or seg_end > instance_data.closing_time or seg_len <= 0
            if invalid_bounds or invalid_len or getattr(program, "unique_id", None) in input_overlap_ids:
                valid_mask[i] = False
                continue

            for block in instance_data.priority_blocks:
                if seg_start < block.end and seg_end > block.start and sched_item.channel_id not in block.allowed_channels:
                    valid_mask[i] = False
                    break

        run = 0
        last_genre = ""
        for i, sched_item in enumerate(sched):
            program = Utils.get_program_by_unique_id(instance_data, Utils.get_uid(sched_item))
            genre = getattr(program, "genre", "") if program else ""
            run = run + 1 if genre and genre == last_genre else 1
            last_genre = genre
            if genre and run > instance_data.max_consecutive_genre:
                valid_mask[i] = False

        active = []
        for i, current in enumerate(sched):
            current_start = Utils.get_start(current)
            current_end = Utils.get_end(current)
            active = [j for j in active if Utils.get_end(sched[j]) > current_start]
            for j in active:
                prev = sched[j]
                if current_start < Utils.get_end(prev) and current_end > Utils.get_start(prev):
                    valid_mask[j] = False
                    valid_mask[i] = False
            active.append(i)

        return sched if all(valid_mask) else [item for item, ok in zip(sched, valid_mask) if ok]

    @staticmethod
    def score_filtered_schedule(sched, instance_data: InstanceData):
        if not sched:
            return 0

        stats = {}
        bonus_sum = 0
        switches = 0
        late = 0

        for i, sched_item in enumerate(sched):
            uid = Utils.get_uid(sched_item)
            program = Utils.get_program_by_unique_id(instance_data, uid)
            if not program:
                continue

            seg_start = Utils.get_start(sched_item)
            seg_end = Utils.get_end(sched_item)
            seg_len = seg_end - seg_start
            full_len = Utils.get_end(program) - Utils.get_start(program)

            prog_stats = stats.setdefault(
                uid,
                {
                    "full_length": full_len,
                    "has_long_segment": False,
                    "has_full_short": False,
                    "reached_end": False,
                    "score": getattr(program, "score", 0),
                },
            )
            prog_stats["has_long_segment"] |= full_len >= instance_data.min_duration and seg_len >= instance_data.min_duration
            prog_stats["has_full_short"] |= full_len < instance_data.min_duration and seg_len == full_len
            prog_stats["reached_end"] |= seg_end >= Utils.get_end(program)

            bonus_sum += AlgorithmUtils.get_segment_bonus(instance_data, sched_item, program)
            late += int(seg_start > Utils.get_start(program))
            switches += int(i > 0 and sched[i - 1].channel_id != sched_item.channel_id)

        base_sum = 0
        early = 0
        for prog_stats in stats.values():
            eligible = prog_stats["has_long_segment"] if prog_stats["full_length"] >= instance_data.min_duration else prog_stats["has_full_short"]
            base_sum += prog_stats["score"] if eligible else 0
            early += int(not prog_stats["reached_end"])

        return int(
            base_sum
            + bonus_sum
            - switches * instance_data.switch_penalty
            - (late + early) * instance_data.termination_penalty
        )

    @staticmethod
    def score_schedule(sched, instance_data: InstanceData, input_overlap_ids: Set[str]):
        filtered = AlgorithmUtils.filter_valid_schedule(sched, instance_data, input_overlap_ids)
        return AlgorithmUtils.score_filtered_schedule(filtered, instance_data)
