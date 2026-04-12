import numpy as np


MOVE_ACTION_NUM = 8
MAX_CLEARANCE_STEPS = 4.0
OPEN_FIELD_TARGET = 0.65
# 开阔度相较上一帧提升时的奖励系数
OPENNESS_DELTA_REWARD_SCALE = 0.06
CLEARANCE_DELTA_REWARD_SCALE = 0.04

# 已经处在较开阔区域时的额外驻留奖励
OPEN_FIELD_BONUS_SCALE = 0.012

# 贴墙压力超过该阈值后开始处罚
WALL_PRESSURE_THRESHOLD = 0.42

# 贴墙压力处罚强度
WALL_PRESSURE_PENALTY_SCALE = 0.024

# 可逃方向比例过低时的阈值
LOW_ESCAPE_THRESHOLD = 0.375

# 可逃方向太少时的处罚强度
LOW_ESCAPE_PENALTY_SCALE = 0.018

SAFE_WALL_STALL_SCALE = 0.01

# 安全期贴墙处罚上限
SAFE_WALL_STALL_CAP = 0.18

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


class TerrainProcessor:
    """基于局部地图计算开阔度相关特征与奖励。"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.last_hero_pos = None
        self.last_openness = None
        self.last_clearance = None
        
        # 连续贴墙停住的步数
        self.wall_stall_steps = 0

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

    def calc_clearance_stats(self, map_info):
        # 计算 8 个方向上的净空情况，能连续走多远不撞墙
        if map_info is None or not map_info or not map_info[0]:
            return 1.0, 1.0

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

        avg_clearance = float(sum(clearances) / len(clearances)) if clearances else 1.0
        min_clearance = float(min(clearances)) if clearances else 1.0
        return avg_clearance, min_clearance

    def extract_stats(self, map_info, move_mask):
        openness = self.calc_openness_ratio(map_info)
        escape_ratio = (
            float(sum(move_mask[:MOVE_ACTION_NUM])) / float(MOVE_ACTION_NUM)
            if move_mask is not None
            else 1.0
        )
        avg_clearance, min_clearance = self.calc_clearance_stats(map_info)
        wall_pressure = 1.0 - (0.45 * openness + 0.35 * escape_ratio + 0.20 * avg_clearance)
        wall_pressure = float(np.clip(wall_pressure, 0.0, 1.0))
        corner_pressure = 1.0 - 0.5 * (escape_ratio + min_clearance)
        corner_pressure = float(np.clip(corner_pressure, 0.0, 1.0))
        return openness, escape_ratio, avg_clearance, wall_pressure, corner_pressure

    def get_feats(self, map_info, move_mask):
        openness, escape_ratio, avg_clearance, _, _ = self.extract_stats(map_info, move_mask)
        return np.array([openness, escape_ratio, avg_clearance], dtype=np.float32)

    def calc_reward(self, hero_pos, map_info, move_mask, danger_score) -> float:
        openness, escape_ratio, avg_clearance, wall_pressure, corner_pressure = self.extract_stats(
            map_info, move_mask
        )
        # danger_score 越低，safe_score 越高，说明越应该主动优化站位
        safe_score = float(np.clip(1.0 - danger_score, 0.0, 1.0))
        reward = 0.0

        if self.last_openness is not None:
            # 当前比上一帧更开阔
            reward += safe_score * OPENNESS_DELTA_REWARD_SCALE * (openness - self.last_openness)
        if self.last_clearance is not None:
            # 当前平均净空更大
            reward += safe_score * CLEARANCE_DELTA_REWARD_SCALE * (avg_clearance - self.last_clearance)

        # 如果已经身处较开阔区域，给少量正反馈
        reward += safe_score * OPEN_FIELD_BONUS_SCALE * max(0.0, openness - OPEN_FIELD_TARGET)

        if wall_pressure > WALL_PRESSURE_THRESHOLD:
            # 贴墙压力过大时，按超过阈值的程度处罚
            pressure_ratio = (wall_pressure - WALL_PRESSURE_THRESHOLD) / max(1.0 - WALL_PRESSURE_THRESHOLD, 1e-6)
            reward -= safe_score * WALL_PRESSURE_PENALTY_SCALE * pressure_ratio

        if escape_ratio < LOW_ESCAPE_THRESHOLD:
            # 能走的方向太少，说明位置偏狭窄，不利于后续拉扯
            escape_gap = (LOW_ESCAPE_THRESHOLD - escape_ratio) / max(LOW_ESCAPE_THRESHOLD, 1e-6)
            reward -= safe_score * LOW_ESCAPE_PENALTY_SCALE * escape_gap

        # 避免模型偏好死角
        reward -= safe_score * 0.012 * corner_pressure

        cur_pos = (hero_pos["x"], hero_pos["z"])
        moved = True
        if self.last_hero_pos is not None:
            moved = cur_pos != self.last_hero_pos

        if moved or wall_pressure <= WALL_PRESSURE_THRESHOLD or safe_score <= 1e-6:
            # 一旦移动了，或者已经不算贴墙，或者局势危险，就清空停滞计数
            self.wall_stall_steps = 0
        else:
            # 安全期贴墙且不动，会逐步加重处罚
            self.wall_stall_steps += 1
            stall_penalty = min(
                SAFE_WALL_STALL_SCALE * self.wall_stall_steps * wall_pressure * safe_score,
                SAFE_WALL_STALL_CAP,
            )
            reward -= stall_penalty

        self.last_hero_pos = cur_pos
        self.last_openness = openness
        self.last_clearance = avg_clearance
        return reward
