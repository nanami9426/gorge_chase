import numpy as np


MOVE_ACTION_NUM = 8
MAX_CLEARANCE_STEPS = 4.0
FLASH_LOOKAHEAD_STEPS = (3, 4, 5)
SQRT_HALF = float(np.sqrt(0.5))
OPEN_FIELD_TARGET = 0.65
# 开阔度相较上一帧提升时的奖励系数
OPENNESS_DELTA_REWARD_SCALE = 0.08
CLEARANCE_DELTA_REWARD_SCALE = 0.055

# 已经处在较开阔区域时的额外驻留奖励
OPEN_FIELD_BONUS_SCALE = 0.016
READINESS_TARGET = 0.62
READINESS_DELTA_REWARD_SCALE = 0.075
READINESS_HOLD_REWARD_SCALE = 0.018
DEAD_END_RISK_THRESHOLD = 0.42
DEAD_END_RISK_PENALTY_SCALE = 0.08
BAD_TERRAIN_STALL_PENALTY_SCALE = 0.055
BAD_TERRAIN_STALL_PENALTY_CAP = 0.32

# 贴墙压力超过该阈值后开始处罚
WALL_PRESSURE_THRESHOLD = 0.42

# 贴墙压力处罚强度
WALL_PRESSURE_PENALTY_SCALE = 0.07

# 可逃方向比例过低时的阈值
LOW_ESCAPE_THRESHOLD = 0.375

# 可逃方向太少时的处罚强度
LOW_ESCAPE_PENALTY_SCALE = 0.055

# 死角压力处罚强度
CORNER_PRESSURE_PENALTY_SCALE = 0.045

# 中等危险开始逐步偏向显式逃跑
ESCAPE_WEIGHT_START = 0.25
ESCAPE_WEIGHT_RANGE = 0.35

# 逃跑方向和动作对齐奖励强度
ESCAPE_ACTION_REWARD_SCALE = 0.12
FLASH_ACTION_REWARD_SCALE = 0.20
FLASH_HIGH_QUALITY_BONUS_SCALE = 0.12
FLASH_LOW_QUALITY_PENALTY_SCALE = 0.08
FLASH_HIGH_QUALITY_THRESHOLD = 0.70

# 仍然沿墙平移而不是向开阔区脱离时的额外处罚
WALL_FOLLOW_PRESSURE_THRESHOLD = 0.55
WALL_FOLLOW_PENALTY_SCALE = 0.045

# 贴墙停滞处罚参数
WALL_STALL_PENALTY_SCALE = 0.065
WALL_STALL_PENALTY_CAP = 0.45

MOVE_DIRS = [
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
    (1, 0),
    (1, 1),
]

MOVE_VECS = [
    (1.0, 0.0),
    (SQRT_HALF, SQRT_HALF),
    (0.0, 1.0),
    (-SQRT_HALF, SQRT_HALF),
    (-1.0, 0.0),
    (-SQRT_HALF, -SQRT_HALF),
    (0.0, -1.0),
    (SQRT_HALF, -SQRT_HALF),
]


class TerrainProcessor:
    """基于局部地图计算开阔度相关特征与奖励。"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.last_hero_pos = None
        self.last_openness = None
        self.last_clearance = None
        self.last_readiness_score = None
        self.last_escape_dir_scores = None
        self.last_flash_dir_scores = None
        self.last_escape_weight = 0.0

        # 连续贴墙停住的步数
        self.wall_stall_steps = 0
        self.bad_terrain_stall_steps = 0

    def is_cell_passable(self, map_info, row, col) -> bool:
        if map_info is None or not map_info or not map_info[0]:
            return True
        if row < 0 or row >= len(map_info):
            return False
        if col < 0 or col >= len(map_info[0]):
            return False
        return bool(map_info[row][col] != 0)

    def calc_ring_ratio(self, map_info, center_row, center_col, radius) -> float:
        # 统计以英雄为中心、指定半径那一圈格子的可通行比例。
        if map_info is None or not map_info or not map_info[0]:
            return 1.0

        passable = 0
        total = 0
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr == 0 and dc == 0:
                    continue
                if max(abs(dr), abs(dc)) != radius:
                    # 只算边框的
                    continue
                total += 1
                if self.is_cell_passable(map_info, center_row + dr, center_col + dc):
                    passable += 1
        return float(passable) / float(total) if total > 0 else 1.0

    def calc_openness_ratio(self, map_info) -> float:
        # 计算当前位置的综合开阔度，把 1/2/3 圈的可通行比例做加权平均
        if map_info is None or not map_info or not map_info[0]:
            return 1.0

        center_row = len(map_info) // 2
        center_col = len(map_info[0]) // 2
        ring1 = self.calc_ring_ratio(map_info, center_row, center_col, radius=1)
        ring2 = self.calc_ring_ratio(map_info, center_row, center_col, radius=2)
        ring3 = self.calc_ring_ratio(map_info, center_row, center_col, radius=3)
        openness = 0.50 * ring1 + 0.30 * ring2 + 0.20 * ring3
        return float(np.clip(openness, 0.0, 1.0))

    def calc_directional_clearances(self, map_info):
        # 计算 8 个方向上的净空情况，能连续走多远不撞墙
        if map_info is None or not map_info or not map_info[0]:
            return [1.0] * MOVE_ACTION_NUM

        center_row = len(map_info) // 2
        center_col = len(map_info[0]) // 2
        clearances = []
        for dr, dc in MOVE_DIRS:
            steps = 0
            for dist in range(1, int(MAX_CLEARANCE_STEPS) + 1):
                row = center_row + dr * dist
                col = center_col + dc * dist
                if not self.is_cell_passable(map_info, row, col):
                    break
                steps += 1
            clearances.append(float(steps) / MAX_CLEARANCE_STEPS)
        return clearances

    def calc_local_escape_ratio(self, map_info, center_row, center_col) -> float:
        if map_info is None or not map_info or not map_info[0]:
            return 1.0

        passable = 0
        for dr, dc in MOVE_DIRS:
            if self.is_cell_passable(map_info, center_row + dr, center_col + dc):
                passable += 1
        return float(passable) / float(MOVE_ACTION_NUM)

    def calc_landing_openness(self, map_info, row, col) -> float:
        if map_info is None or not map_info or not map_info[0]:
            return 1.0
        if not self.is_cell_passable(map_info, row, col):
            return 0.0

        ring1 = self.calc_ring_ratio(map_info, row, col, radius=1)
        ring2 = self.calc_ring_ratio(map_info, row, col, radius=2)
        local_escape_ratio = self.calc_local_escape_ratio(map_info, row, col)
        return float(np.clip(0.45 * ring1 + 0.35 * ring2 + 0.20 * local_escape_ratio, 0.0, 1.0))

    def calc_clearance_stats(self, map_info):
        clearances = self.calc_directional_clearances(map_info)
        avg_clearance = float(sum(clearances) / len(clearances)) if clearances else 1.0
        min_clearance = float(min(clearances)) if clearances else 1.0
        return clearances, avg_clearance, min_clearance

    def normalize_monster_vec(self, monster_vec):
        if monster_vec is None:
            return None

        vec_x = float(monster_vec[0])
        vec_z = float(monster_vec[1])
        norm = float(np.sqrt(vec_x * vec_x + vec_z * vec_z))
        if norm <= 1e-6:
            return None
        return (vec_x / norm, vec_z / norm)

    def calc_escape_dir_scores(self, clearances, move_mask, monster_vec):
        monster_vec = self.normalize_monster_vec(monster_vec)
        move_mask = list(move_mask[:MOVE_ACTION_NUM]) if move_mask is not None else [1] * MOVE_ACTION_NUM

        escape_dir_scores = []
        for idx, clearance in enumerate(clearances):
            left_adj = clearances[(idx - 1) % MOVE_ACTION_NUM]
            right_adj = clearances[(idx + 1) % MOVE_ACTION_NUM]
            sector_open = 0.50 * clearance + 0.25 * left_adj + 0.25 * right_adj

            if monster_vec is None:
                away = 0.5
            else:
                move_vec = MOVE_VECS[idx]
                away = (1.0 - (move_vec[0] * monster_vec[0] + move_vec[1] * monster_vec[1])) / 2.0
                away = float(np.clip(away, 0.0, 1.0))

            score = float(move_mask[idx]) * (0.35 * sector_open + 0.25 * clearance + 0.40 * away)
            escape_dir_scores.append(float(np.clip(score, 0.0, 1.0)))
        return escape_dir_scores

    def calc_flash_dir_scores(self, map_info, legal_action, clearances, monster_vec):
        monster_vec = self.normalize_monster_vec(monster_vec)
        legal_action = list(legal_action) if legal_action is not None else [1] * (MOVE_ACTION_NUM * 2)
        flash_mask = [
            int(legal_action[idx + MOVE_ACTION_NUM])
            if idx + MOVE_ACTION_NUM < len(legal_action)
            else 1
            for idx in range(MOVE_ACTION_NUM)
        ]

        if map_info is None or not map_info or not map_info[0]:
            return [
                float(mask)
                * (
                    0.5
                    if monster_vec is None
                    else (1.0 - (MOVE_VECS[idx][0] * monster_vec[0] + MOVE_VECS[idx][1] * monster_vec[1])) / 2.0
                )
                for idx, mask in enumerate(flash_mask)
            ]

        center_row = len(map_info) // 2
        center_col = len(map_info[0]) // 2
        flash_dir_scores = []
        for idx, (dr, dc) in enumerate(MOVE_DIRS):
            if not flash_mask[idx]:
                flash_dir_scores.append(0.0)
                continue

            move_vec = MOVE_VECS[idx]
            if monster_vec is None:
                away = 0.5
            else:
                away = (1.0 - (move_vec[0] * monster_vec[0] + move_vec[1] * monster_vec[1])) / 2.0
                away = float(np.clip(away, 0.0, 1.0))

            wall_cut_bonus = max(0.0, 1.0 - float(clearances[idx]))
            best_score = 0.0
            for dist in FLASH_LOOKAHEAD_STEPS:
                row = center_row + dr * dist
                col = center_col + dc * dist
                if not self.is_cell_passable(map_info, row, col):
                    continue

                landing_open = self.calc_landing_openness(map_info, row, col)
                landing_escape_ratio = self.calc_local_escape_ratio(map_info, row, col)
                score = (
                    0.42 * away
                    + 0.25 * landing_open
                    + 0.18 * landing_escape_ratio
                    + 0.15 * wall_cut_bonus
                )
                best_score = max(best_score, score)

            flash_dir_scores.append(float(np.clip(best_score, 0.0, 1.0)))
        return flash_dir_scores

    def calc_positioning_weight(self, danger_score) -> float:
        return float(max(0.25, 1.0 - float(np.clip(danger_score, 0.0, 1.0))))

    def calc_escape_weight(self, danger_score) -> float:
        return float(
            np.clip(
                (float(danger_score) - ESCAPE_WEIGHT_START) / ESCAPE_WEIGHT_RANGE,
                0.0,
                1.0,
            )
        )

    def calc_trap_risk(self, escape_ratio, wall_pressure, corner_pressure) -> float:
        low_escape_gap = max(0.0, LOW_ESCAPE_THRESHOLD - float(escape_ratio))
        low_escape_pressure = low_escape_gap / max(LOW_ESCAPE_THRESHOLD, 1e-6)
        trap_risk = 0.45 * float(wall_pressure) + 0.35 * float(corner_pressure) + 0.20 * low_escape_pressure
        return float(np.clip(trap_risk, 0.0, 1.0))

    def calc_route_diversity(self, escape_dir_scores, flash_dir_scores) -> float:
        move_options = sum(1 for score in escape_dir_scores if float(score) >= 0.45)
        flash_options = sum(1 for score in flash_dir_scores if float(score) >= 0.65)
        return float(np.clip((move_options + 0.5 * flash_options) / float(MOVE_ACTION_NUM), 0.0, 1.0))

    def calc_readiness_score(
        self,
        openness,
        escape_ratio,
        avg_clearance,
        escape_dir_scores,
        flash_dir_scores,
    ) -> float:
        sorted_escape_scores = sorted((float(score) for score in escape_dir_scores), reverse=True)
        top_move_score = float(np.mean(sorted_escape_scores[:3])) if sorted_escape_scores else 0.0
        best_flash_score = max((float(score) for score in flash_dir_scores), default=0.0)
        readiness_score = (
            0.22 * float(openness)
            + 0.20 * float(escape_ratio)
            + 0.18 * float(avg_clearance)
            + 0.30 * top_move_score
            + 0.10 * best_flash_score
        )
        return float(np.clip(readiness_score, 0.0, 1.0))

    def calc_dead_end_risk(self, trap_risk, readiness_score, route_diversity, min_clearance) -> float:
        readiness_gap = max(0.0, READINESS_TARGET - float(readiness_score)) / max(READINESS_TARGET, 1e-6)
        diversity_gap = 1.0 - float(route_diversity)
        clearance_gap = 1.0 - float(min_clearance)
        dead_end_risk = (
            0.40 * float(trap_risk)
            + 0.30 * readiness_gap
            + 0.20 * diversity_gap
            + 0.10 * clearance_gap
        )
        return float(np.clip(dead_end_risk, 0.0, 1.0))

    def extract_stats(self, map_info, move_mask, monster_vec=None, legal_action=None):
        openness = self.calc_openness_ratio(map_info)
        escape_ratio = (
            float(sum(move_mask[:MOVE_ACTION_NUM])) / float(MOVE_ACTION_NUM)
            if move_mask is not None
            else 1.0
        )
        clearances, avg_clearance, min_clearance = self.calc_clearance_stats(map_info)
        wall_pressure = 1.0 - (0.45 * openness + 0.35 * escape_ratio + 0.20 * avg_clearance)
        wall_pressure = float(np.clip(wall_pressure, 0.0, 1.0))
        corner_pressure = 1.0 - 0.5 * (escape_ratio + min_clearance)
        corner_pressure = float(np.clip(corner_pressure, 0.0, 1.0))
        trap_risk = self.calc_trap_risk(
            escape_ratio=escape_ratio,
            wall_pressure=wall_pressure,
            corner_pressure=corner_pressure,
        )
        escape_dir_scores = self.calc_escape_dir_scores(clearances, move_mask, monster_vec)
        flash_dir_scores = self.calc_flash_dir_scores(
            map_info=map_info,
            legal_action=legal_action,
            clearances=clearances,
            monster_vec=monster_vec,
        )
        route_diversity = self.calc_route_diversity(
            escape_dir_scores=escape_dir_scores,
            flash_dir_scores=flash_dir_scores,
        )
        readiness_score = self.calc_readiness_score(
            openness=openness,
            escape_ratio=escape_ratio,
            avg_clearance=avg_clearance,
            escape_dir_scores=escape_dir_scores,
            flash_dir_scores=flash_dir_scores,
        )
        dead_end_risk = self.calc_dead_end_risk(
            trap_risk=trap_risk,
            readiness_score=readiness_score,
            route_diversity=route_diversity,
            min_clearance=min_clearance,
        )
        return {
            "openness": openness,
            "escape_ratio": escape_ratio,
            "avg_clearance": avg_clearance,
            "min_clearance": min_clearance,
            "wall_pressure": wall_pressure,
            "corner_pressure": corner_pressure,
            "trap_risk": trap_risk,
            "readiness_score": readiness_score,
            "dead_end_risk": dead_end_risk,
            "route_diversity": route_diversity,
            "best_flash_score": max(flash_dir_scores) if flash_dir_scores else 0.0,
            "clearances": clearances,
            "escape_dir_scores": escape_dir_scores,
            "flash_dir_scores": flash_dir_scores,
        }

    def get_feats(self, terrain_stats):
        return np.array(
            [
                terrain_stats["openness"],
                terrain_stats["escape_ratio"],
                terrain_stats["avg_clearance"],
                terrain_stats["min_clearance"],
                terrain_stats["wall_pressure"],
                terrain_stats["corner_pressure"],
                terrain_stats["readiness_score"],
                terrain_stats["dead_end_risk"],
                terrain_stats["route_diversity"],
                terrain_stats["best_flash_score"],
                *terrain_stats["escape_dir_scores"],
                *terrain_stats["flash_dir_scores"],
            ],
            dtype=np.float32,
        )

    def calc_reward(self, hero_pos, terrain_stats, last_action, danger_score) -> float:
        openness = terrain_stats["openness"]
        escape_ratio = terrain_stats["escape_ratio"]
        avg_clearance = terrain_stats["avg_clearance"]
        wall_pressure = terrain_stats["wall_pressure"]
        corner_pressure = terrain_stats["corner_pressure"]
        readiness_score = terrain_stats["readiness_score"]
        dead_end_risk = terrain_stats["dead_end_risk"]
        escape_dir_scores = terrain_stats["escape_dir_scores"]
        flash_dir_scores = terrain_stats["flash_dir_scores"]

        preemptive_weight = float(max(0.6, 1.0 - 0.5 * float(np.clip(danger_score, 0.0, 1.0))))
        escape_weight = self.calc_escape_weight(danger_score)
        reward = 0.0

        if self.last_openness is not None:
            # 当前比上一帧更开阔
            reward += preemptive_weight * OPENNESS_DELTA_REWARD_SCALE * (openness - self.last_openness)
        if self.last_clearance is not None:
            # 当前平均净空更大
            reward += preemptive_weight * CLEARANCE_DELTA_REWARD_SCALE * (avg_clearance - self.last_clearance)
        if self.last_readiness_score is not None:
            # 怪物还远时也要提前优化逃生余量，避免等怪加速后才发现自己在死胡同。
            reward += preemptive_weight * READINESS_DELTA_REWARD_SCALE * (
                readiness_score - self.last_readiness_score
            )

        # 如果已经身处较开阔区域，给少量正反馈
        reward += preemptive_weight * OPEN_FIELD_BONUS_SCALE * max(0.0, openness - OPEN_FIELD_TARGET)
        reward += preemptive_weight * READINESS_HOLD_REWARD_SCALE * max(0.0, readiness_score - READINESS_TARGET)

        if wall_pressure > WALL_PRESSURE_THRESHOLD:
            # 贴墙压力过大时，按超过阈值的程度处罚
            pressure_ratio = (wall_pressure - WALL_PRESSURE_THRESHOLD) / max(1.0 - WALL_PRESSURE_THRESHOLD, 1e-6)
            reward -= preemptive_weight * WALL_PRESSURE_PENALTY_SCALE * pressure_ratio

        if dead_end_risk > DEAD_END_RISK_THRESHOLD:
            risk_ratio = (dead_end_risk - DEAD_END_RISK_THRESHOLD) / max(1.0 - DEAD_END_RISK_THRESHOLD, 1e-6)
            reward -= preemptive_weight * DEAD_END_RISK_PENALTY_SCALE * risk_ratio

        if escape_ratio < LOW_ESCAPE_THRESHOLD:
            # 能走的方向太少，说明位置偏狭窄，不利于后续拉扯
            escape_gap = (LOW_ESCAPE_THRESHOLD - escape_ratio) / max(LOW_ESCAPE_THRESHOLD, 1e-6)
            reward -= preemptive_weight * LOW_ESCAPE_PENALTY_SCALE * escape_gap

        # 避免模型偏好死角
        reward -= preemptive_weight * CORNER_PRESSURE_PENALTY_SCALE * corner_pressure

        if self.last_escape_dir_scores is not None and last_action is not None:
            action_idx = int(last_action)
            if 0 <= action_idx < MOVE_ACTION_NUM:
                dir_idx = action_idx % MOVE_ACTION_NUM
                mean_escape_score = float(np.mean(self.last_escape_dir_scores))
                reward += self.last_escape_weight * ESCAPE_ACTION_REWARD_SCALE * (
                    float(self.last_escape_dir_scores[dir_idx]) - mean_escape_score
                )
            elif MOVE_ACTION_NUM <= action_idx < MOVE_ACTION_NUM * 2 and self.last_flash_dir_scores is not None:
                dir_idx = action_idx % MOVE_ACTION_NUM
                flash_score = float(self.last_flash_dir_scores[dir_idx])
                mean_flash_score = float(np.mean(self.last_flash_dir_scores))
                flash_quality_gap = flash_score - mean_flash_score
                reward += self.last_escape_weight * FLASH_ACTION_REWARD_SCALE * flash_quality_gap
                if flash_score >= FLASH_HIGH_QUALITY_THRESHOLD:
                    reward += (
                        self.last_escape_weight
                        * FLASH_HIGH_QUALITY_BONUS_SCALE
                        * (flash_score - FLASH_HIGH_QUALITY_THRESHOLD)
                        / max(1.0 - FLASH_HIGH_QUALITY_THRESHOLD, 1e-6)
                    )
                elif flash_score < mean_flash_score:
                    reward -= self.last_escape_weight * FLASH_LOW_QUALITY_PENALTY_SCALE * (mean_flash_score - flash_score)

        cur_pos = (hero_pos["x"], hero_pos["z"])
        moved = True
        if self.last_hero_pos is not None:
            moved = cur_pos != self.last_hero_pos

        if (
            moved
            and wall_pressure >= WALL_FOLLOW_PRESSURE_THRESHOLD
            and self.last_escape_dir_scores is not None
            and self.last_openness is not None
            and last_action is not None
        ):
            action_idx = int(last_action)
            if 0 <= action_idx < MOVE_ACTION_NUM:
                dir_idx = action_idx % MOVE_ACTION_NUM
                escape_score = float(self.last_escape_dir_scores[dir_idx])
                mean_escape_score = float(np.mean(self.last_escape_dir_scores))
                if escape_score < mean_escape_score and openness <= self.last_openness + 1e-6:
                    reward -= WALL_FOLLOW_PENALTY_SCALE * wall_pressure * (1.0 - escape_score)

        if moved or wall_pressure <= WALL_PRESSURE_THRESHOLD:
            # 一旦移动了，或者已经不算贴墙，就清空停滞计数
            self.wall_stall_steps = 0
        else:
            # 贴墙停滞会在危险时继续加重，低危险时也要避免“贴墙挂机”
            self.wall_stall_steps += 1
            stall_penalty = min(
                WALL_STALL_PENALTY_SCALE
                * self.wall_stall_steps
                * wall_pressure
                * (0.85 + 0.75 * float(np.clip(danger_score, 0.0, 1.0))),
                WALL_STALL_PENALTY_CAP,
            )
            reward -= stall_penalty

        if moved or dead_end_risk <= DEAD_END_RISK_THRESHOLD:
            self.bad_terrain_stall_steps = 0
        else:
            self.bad_terrain_stall_steps += 1
            reward -= min(
                BAD_TERRAIN_STALL_PENALTY_SCALE * self.bad_terrain_stall_steps * dead_end_risk,
                BAD_TERRAIN_STALL_PENALTY_CAP,
            )

        self.last_hero_pos = cur_pos
        self.last_openness = openness
        self.last_clearance = avg_clearance
        self.last_readiness_score = readiness_score
        self.last_escape_dir_scores = np.array(escape_dir_scores, dtype=np.float32)
        self.last_flash_dir_scores = np.array(flash_dir_scores, dtype=np.float32)
        self.last_escape_weight = escape_weight
        return reward
