import math
from collections import deque

import numpy as np


MAP_SIZE = 128
EXPLORE_GRID_REWARD = 0.1
FRONTIER_BONUS_SCALE = 0.015
EXPLORE_STREAK_BONUS_SCALE = 0.008
EXPLORE_STREAK_BONUS_CAP = 0.05
MOVE_REWARD_SCALE = 0.004
MOVE_REWARD_CAP = 2.0
REVISIT_GRID_PENALTY = -0.025
NON_RECENT_REVISIT_REWARD = 0.001
STALL_PENALTY_SCALE = 0.012
STALL_PENALTY_CAP = 0.30
NO_PROGRESS_PENALTY_START = 3
NO_PROGRESS_PENALTY_SCALE = 0.026
NO_PROGRESS_PENALTY_CAP = 0.45
LOCAL_LOOP_PENALTY_START = 2
LOCAL_LOOP_PENALTY_SCALE = 0.018
LOCAL_LOOP_PENALTY_CAP = 0.35
RECENT_GRID_WINDOW = 20
VISIT_COUNT_NORM_CAP = 6.0
NO_PROGRESS_NORM_CAP = 20.0
MEMORY_EMA_ALPHA = 0.15
LOOP_LOOKBACK_WINDOW = 10
LOOP_DISTANCE_THRESHOLD = 5.0
LOOP_WINDOW_PENALTY = 0.10

SAFE_MEMORY_DIRS = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
]


class ExploreProcessor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.visited_grid_counts = {}
        self.grid_size = 4
        self.max_grid_index = int((MAP_SIZE - 1) // self.grid_size)
        self.last_hero_pos = None
        self.stall_steps = 0
        self.no_progress_steps = 0
        self.explore_streak_steps = 0
        self.recent_grids = deque(maxlen=RECENT_GRID_WINDOW)
        self.position_history = deque(maxlen=LOOP_LOOKBACK_WINDOW + 1)
        self.danger_grid_ema = {}

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
        lookback_valid, lookback_dx_norm, lookback_dz_norm, _ = self.get_lookback_position_info(hero_pos)
        return np.array(
            [
                context["current_grid_visit_count_norm"],
                context["frontier_ratio"],
                context["no_progress_steps_norm"],
                lookback_valid,
                lookback_dx_norm,
                lookback_dz_norm,
            ],
            dtype=np.float32,
        )

    def get_lookback_position_info(self, hero_pos):
        if len(self.position_history) < LOOP_LOOKBACK_WINDOW:
            return 0.0, 0.0, 0.0, 0.0

        lookback_pos = self.position_history[-LOOP_LOOKBACK_WINDOW]
        delta_x = float(hero_pos["x"] - lookback_pos[0])
        delta_z = float(hero_pos["z"] - lookback_pos[1])
        loop_dist = math.sqrt(delta_x * delta_x + delta_z * delta_z)
        return (
            1.0,
            float(np.clip(delta_x / LOOP_DISTANCE_THRESHOLD, -1.0, 1.0)),
            float(np.clip(delta_z / LOOP_DISTANCE_THRESHOLD, -1.0, 1.0)),
            loop_dist,
        )

    def get_memory_feats(self, hero_pos):
        grid = self.get_grid(hero_pos)
        current_danger = float(self.danger_grid_ema.get(grid, 0.0))
        best_safe_score = 0.0
        best_dir = (0.0, 0.0)
        known_neighbors = 0
        safe_neighbors = 0
        grid_x, grid_z = grid

        for dir_x, dir_z in SAFE_MEMORY_DIRS:
            next_grid = (grid_x + dir_x, grid_z + dir_z)
            if not (0 <= next_grid[0] <= self.max_grid_index and 0 <= next_grid[1] <= self.max_grid_index):
                continue
            if next_grid in self.danger_grid_ema:
                known_neighbors += 1
            danger_memory = float(self.danger_grid_ema.get(next_grid, 0.25))
            safe_score = float(np.clip(1.0 - danger_memory, 0.0, 1.0))
            if safe_score >= 0.65:
                safe_neighbors += 1
            if safe_score > best_safe_score:
                best_safe_score = safe_score
                best_dir = (float(dir_x), float(dir_z))

        local_safe_ratio = float(safe_neighbors / max(1, len(SAFE_MEMORY_DIRS)))
        memory_confidence = float(np.clip(known_neighbors / max(1, len(SAFE_MEMORY_DIRS)), 0.0, 1.0))
        return np.array(
            [
                current_danger,
                best_safe_score,
                best_dir[0],
                best_dir[1],
                local_safe_ratio * memory_confidence,
            ],
            dtype=np.float32,
        )

    def update_safety_memory(self, grid, danger_score, terrain_stats=None):
        terrain_stats = terrain_stats or {}
        danger_score = float(np.clip(danger_score, 0.0, 1.0))
        dead_end_risk = float(np.clip(terrain_stats.get("dead_end_risk", 0.0), 0.0, 1.0))
        readiness_score = float(np.clip(terrain_stats.get("readiness_score", 1.0), 0.0, 1.0))
        danger_signal = float(
            np.clip(
                0.55 * danger_score + 0.25 * dead_end_risk + 0.20 * (1.0 - readiness_score),
                0.0,
                1.0,
            )
        )
        old_signal = float(self.danger_grid_ema.get(grid, danger_signal))
        self.danger_grid_ema[grid] = (1.0 - MEMORY_EMA_ALPHA) * old_signal + MEMORY_EMA_ALPHA * danger_signal

    def calc_revisit_adjustment(self, grid, explore_new_grid, visit_count_after):
        if explore_new_grid:
            return 0.0

        if grid in self.recent_grids:
            return REVISIT_GRID_PENALTY

        # 非 recent 的旧格子给一个小额衰减奖励，鼓励合理回撤与绕怪，
        # 但访问次数越多，奖励越低，避免来回刷同一批安全格子。
        revisit_count = max(1.0, float(visit_count_after - 1))
        return float(NON_RECENT_REVISIT_REWARD / math.sqrt(revisit_count))

    def calc_positioning_need(self, terrain_stats) -> float:
        terrain_stats = terrain_stats or {}
        dead_end_risk = float(terrain_stats.get("dead_end_risk", 0.0))
        readiness_score = float(terrain_stats.get("readiness_score", 1.0))
        dead_end_need = max(0.0, dead_end_risk - 0.25) / 0.75
        readiness_need = max(0.0, 0.58 - readiness_score) / 0.58
        return float(np.clip(0.65 * dead_end_need + 0.35 * readiness_need, 0.0, 1.0))

    def calc_reward(self, hero_pos, step_no, danger_score, terrain_stats=None):
        context = self.get_context(hero_pos)
        grid = context["grid"]
        explore_new_grid = context["explore_new_grid"]

        if explore_new_grid:
            explore_reward = EXPLORE_GRID_REWARD
            explore_streak_steps = self.explore_streak_steps + 1
        else:
            explore_reward = 0.0
            explore_streak_steps = 0

        revisit_adjustment = self.calc_revisit_adjustment(
            grid=grid,
            explore_new_grid=explore_new_grid,
            visit_count_after=context["visit_count_after"],
        )
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

        positioning_need = self.calc_positioning_need(terrain_stats)
        safe_wait_weight = float(np.clip(1.0 - 0.65 * float(danger_score), 0.35, 1.0))
        no_progress_over = max(0, context["no_progress_steps"] - NO_PROGRESS_PENALTY_START)
        no_progress_penalty = -min(
            NO_PROGRESS_PENALTY_SCALE
            * no_progress_over
            * safe_wait_weight
            * (0.45 + positioning_need),
            NO_PROGRESS_PENALTY_CAP,
        )
        local_loop_over = max(0, context["visit_count_after"] - LOCAL_LOOP_PENALTY_START)
        local_loop_penalty = -min(
            LOCAL_LOOP_PENALTY_SCALE
            * local_loop_over
            * safe_wait_weight
            * (0.40 + positioning_need),
            LOCAL_LOOP_PENALTY_CAP,
        )
        lookback_valid, _, _, loop_dist = self.get_lookback_position_info(hero_pos)
        window_loop_penalty = (
            -LOOP_WINDOW_PENALTY
            if lookback_valid > 0.0 and loop_dist < LOOP_DISTANCE_THRESHOLD
            else 0.0
        )

        self.visited_grid_counts[grid] = context["visit_count_after"]
        self.update_safety_memory(grid=grid, danger_score=danger_score, terrain_stats=terrain_stats)
        self.no_progress_steps = context["no_progress_steps"]
        self.explore_streak_steps = explore_streak_steps
        self.last_hero_pos = cur_pos
        self.recent_grids.append(grid)
        self.position_history.append(cur_pos)
        return {
            "reward": float(
                positive_reward
                + revisit_adjustment
                + stall_penalty
                + no_progress_penalty
                + local_loop_penalty
                + window_loop_penalty
            ),
            "explore_new_grid": int(explore_new_grid),
            "frontier_bonus": float(safe_explore_weight * frontier_bonus),
            "explore_streak_bonus": float(safe_explore_weight * explore_streak_bonus),
            "no_progress_penalty": float(no_progress_penalty),
            "local_loop_penalty": float(local_loop_penalty),
            "window_loop_penalty": float(window_loop_penalty),
            "positioning_need": float(positioning_need),
        }
