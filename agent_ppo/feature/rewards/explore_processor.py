import math
from collections import deque

import numpy as np


MAP_SIZE = 128
EXPLORE_GRID_REWARD = 0.05
FRONTIER_BONUS_SCALE = 0.015
EXPLORE_STREAK_BONUS_SCALE = 0.005
EXPLORE_STREAK_BONUS_CAP = 0.03
MOVE_REWARD_SCALE = 0.004
MOVE_REWARD_CAP = 2.0
REVISIT_GRID_PENALTY = -0.004
STALL_PENALTY_SCALE = 0.005
STALL_PENALTY_CAP = 0.18
RECENT_GRID_WINDOW = 16
VISIT_COUNT_NORM_CAP = 6.0
NO_PROGRESS_NORM_CAP = 20.0


class ExploreProcessor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.visited_grid_counts = {}
        self.grid_size = 6
        self.max_grid_index = int((MAP_SIZE - 1) // self.grid_size)
        self.last_hero_pos = None
        self.stall_steps = 0
        self.no_progress_steps = 0
        self.explore_streak_steps = 0
        self.recent_grids = deque(maxlen=RECENT_GRID_WINDOW)

    def get_grid(self, hero_pos):
        return (int(hero_pos["x"]) // self.grid_size, int(hero_pos["z"]) // self.grid_size)

    def get_frontier_ratio(self, grid, visit_count_after) -> float:
        frontier_total = 0
        frontier_unvisited = 0
        grid_x, grid_z = grid
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dz == 0:
                    continue
                next_x = grid_x + dx
                next_z = grid_z + dz
                if not (0 <= next_x <= self.max_grid_index and 0 <= next_z <= self.max_grid_index):
                    continue
                frontier_total += 1
                if self.visited_grid_counts.get((next_x, next_z), 0) == 0:
                    frontier_unvisited += 1

        if frontier_total <= 0:
            return 0.0

        raw_frontier_ratio = float(frontier_unvisited) / float(frontier_total)
        frontier_decay = 1.0 / math.sqrt(max(1.0, float(visit_count_after)))
        return float(raw_frontier_ratio * frontier_decay)

    def get_context(self, hero_pos):
        grid = self.get_grid(hero_pos)
        visit_count_before = int(self.visited_grid_counts.get(grid, 0))
        visit_count_after = visit_count_before + 1
        explore_new_grid = int(visit_count_before == 0)
        frontier_ratio = self.get_frontier_ratio(grid=grid, visit_count_after=visit_count_after)
        no_progress_steps = 0 if explore_new_grid else (self.no_progress_steps + 1)
        return {
            "grid": grid,
            "explore_new_grid": explore_new_grid,
            "visit_count_after": visit_count_after,
            "current_grid_visit_count_norm": float(
                min(visit_count_after, VISIT_COUNT_NORM_CAP) / VISIT_COUNT_NORM_CAP
            ),
            "frontier_ratio": frontier_ratio,
            "no_progress_steps": no_progress_steps,
            "no_progress_steps_norm": float(
                min(no_progress_steps, NO_PROGRESS_NORM_CAP) / NO_PROGRESS_NORM_CAP
            ),
        }

    def get_feats(self, hero_pos):
        context = self.get_context(hero_pos)
        return np.array(
            [
                context["current_grid_visit_count_norm"],
                context["frontier_ratio"],
                context["no_progress_steps_norm"],
            ],
            dtype=np.float32,
        )

    def calc_reward(self, hero_pos, step_no, danger_score):
        context = self.get_context(hero_pos)
        grid = context["grid"]
        explore_new_grid = context["explore_new_grid"]

        if explore_new_grid:
            explore_reward = EXPLORE_GRID_REWARD
            explore_streak_steps = self.explore_streak_steps + 1
        else:
            explore_reward = 0.0
            explore_streak_steps = 0

        revisit_penalty = REVISIT_GRID_PENALTY if grid in self.recent_grids else 0.0
        cur_pos = (hero_pos["x"], hero_pos["z"])
        move_reward = 0.0
        if self.last_hero_pos is None:
            moved = True
        else:
            delta_x = float(cur_pos[0] - self.last_hero_pos[0])
            delta_z = float(cur_pos[1] - self.last_hero_pos[1])
            move_dist = math.sqrt(delta_x * delta_x + delta_z * delta_z)
            moved = move_dist > 1e-6
            move_reward = MOVE_REWARD_SCALE * min(move_dist, MOVE_REWARD_CAP) / MOVE_REWARD_CAP

        frontier_bonus = FRONTIER_BONUS_SCALE * context["frontier_ratio"] if moved else 0.0
        explore_streak_bonus = (
            min(EXPLORE_STREAK_BONUS_SCALE * explore_streak_steps, EXPLORE_STREAK_BONUS_CAP)
            if explore_new_grid
            else 0.0
        )
        safe_explore_weight = float(np.clip(1.1 - 0.7 * float(danger_score), 0.35, 1.0))
        positive_reward = safe_explore_weight * (
            explore_reward + frontier_bonus + explore_streak_bonus + move_reward
        )

        if moved:
            self.stall_steps = 0
        else:
            self.stall_steps += 1
        stall_penalty = -min(STALL_PENALTY_SCALE * self.stall_steps, STALL_PENALTY_CAP)

        self.visited_grid_counts[grid] = context["visit_count_after"]
        self.no_progress_steps = context["no_progress_steps"]
        self.explore_streak_steps = explore_streak_steps
        self.last_hero_pos = cur_pos
        self.recent_grids.append(grid)
        return {
            "reward": float(positive_reward + revisit_penalty + stall_penalty),
            "explore_new_grid": int(explore_new_grid),
            "frontier_bonus": float(safe_explore_weight * frontier_bonus),
            "explore_streak_bonus": float(safe_explore_weight * explore_streak_bonus),
        }
