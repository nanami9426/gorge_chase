import numpy as np


MAP_SIZE = 128.0
MAX_DIST_BUCKET = 5.0
MAX_MAP_DISTANCE = MAP_SIZE * 1.41
TREASURE_APPROACH_REWARD_SCALE = 2.80
TREASURE_NEAR_APPROACH_BONUS = 3.20
BUFF_APPROACH_REWARD_SCALE = 0.65
TREASURE_PICKUP_REWARD = 8.0
BUFF_PICKUP_REWARD = 3.0
TREASURE_NEAR_DIST_NORM = 10.0 / MAX_MAP_DISTANCE # 近距离宝箱的阈值，大约是6格以内
TREASURE_STALL_PENALTY_SCALE = 0.045
TREASURE_STALL_PENALTY_CAP = 0.45
TREASURE_NO_PROGRESS_PENALTY_START = 3
TREASURE_NO_PROGRESS_PENALTY_SCALE = 0.030
TREASURE_NO_PROGRESS_PENALTY_CAP = 0.55
BUFF_PRIORITY_FAR_DIFF_NORM = 0.06
BUFF_PRIORITY_FAR_CAP = 0.5
TREASURE_DANGER_DAMP_START = 0.62
TREASURE_DANGER_DAMP_END = 0.92
SAFE_TREASURE_DANGER_THRESHOLD = 0.58
SAFE_TREASURE_READINESS_THRESHOLD = 0.45
SAFE_TREASURE_APPROACH_BONUS = 0.55
MAX_TREASURE_COUNT = 10.0
SQRT_HALF = float(np.sqrt(0.5))

DIRECTION_TO_VECTOR = {
    0: (0.0, 0.0),
    1: (1.0, 0.0),
    2: (SQRT_HALF, SQRT_HALF),
    3: (0.0, 1.0),
    4: (-SQRT_HALF, SQRT_HALF),
    5: (-1.0, 0.0),
    6: (-SQRT_HALF, -SQRT_HALF),
    7: (0.0, -1.0),
    8: (SQRT_HALF, -SQRT_HALF),
}


def _norm(v, v_max, v_min=0.0):
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0


class OrganProcessor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.last_treasure_dist_norm = None
        self.last_treasure_key = None
        self.last_buff_dist_norm = None
        self.last_treasures_collected = None
        self.last_collected_buff = None
        self.treasure_stall_steps = 0
        self.treasure_no_progress_steps = 0

    def get_feats(self, organs, hero_pos):
        available_organs = self.build_available_organs(organs, hero_pos)
        nearest_treasure = self.select_nearest_organ(available_organs, sub_type=1)
        nearest_buff = self.select_nearest_organ(available_organs, sub_type=2)
        # 特征固定编码为：最近宝箱 4 维 + 最近 buff 4 维
        organs_feat = np.concatenate(
            [
                self.encode_organ_feat(nearest_treasure),
                self.encode_organ_feat(nearest_buff),
            ]
        )
        return organs_feat
    
    def direction_to_vector(self, direction_idx):
        return DIRECTION_TO_VECTOR.get(int(direction_idx), (0.0, 0.0))

    def calc_organ_dist_norm(self, organ, hero_pos):
        # 计算单个物件到英雄的归一化距离。
        organ_pos = organ.get("pos")
        if isinstance(organ_pos, dict) and "x" in organ_pos and "z" in organ_pos:
            raw_dist = np.sqrt((hero_pos["x"] - organ_pos["x"]) ** 2 + (hero_pos["z"] - organ_pos["z"]) ** 2)
            return _norm(raw_dist, MAX_MAP_DISTANCE)
        # 如果坐标字段缺失，使用hero_l2_distance桶编号
        return _norm(organ.get("hero_l2_distance", MAX_DIST_BUCKET), MAX_DIST_BUCKET)

    def build_available_organs(self, organs, hero_pos):
        # 标准化后的可拾取物件列表
        available_organs = []
        for organ in organs:
            if int(organ.get("status", 0)) != 1: # status 状态（1=可获取）
                continue
            sub_type = int(organ.get("sub_type", 0)) # 物件类型（1=宝箱，2=加速 buff）
            if sub_type not in (1, 2):
                continue
            dir_x, dir_z = self.direction_to_vector(organ.get("hero_relative_direction", 0)) # 物件相对于英雄的方位（0-8）
            available_organs.append(
                {
                    "key": (sub_type, int(organ.get("config_id", -1))), # 配置 ID（实体 ID）
                    "sub_type": sub_type,
                    "dist_norm": self.calc_organ_dist_norm(organ, hero_pos),
                    "dir_x": dir_x,
                    "dir_z": dir_z,
                }
            )
        return available_organs

    def select_nearest_organ(self, organs, sub_type=None):
        # 从候选物件中挑出最近的目标。
        candidates = organs
        if sub_type is not None:
            candidates = [organ for organ in organs if organ["sub_type"] == sub_type]
        if not candidates:
            return None
        return min(candidates, key=lambda organ: organ["dist_norm"])

    def select_target_organ(self, organs):
        # 训练时优先追宝箱，只有没有宝箱时才考虑 buff。
        nearest_treasure = self.select_nearest_organ(organs, sub_type=1)
        if nearest_treasure is not None:
            return nearest_treasure
        return self.select_nearest_organ(organs, sub_type=2)

    def encode_organ_feat(self, organ_info):
        # 如果当前类型没有可拾取目标，则返回全零
        if organ_info is None:
            return np.zeros(4, dtype=np.float32)
        return np.array(
            # has_target, dist_norm, dir_x, dir_z
            [1.0, organ_info["dist_norm"], organ_info["dir_x"], organ_info["dir_z"]],
            dtype=np.float32,
        )

    def score_treasure_priority(self, treasure, terrain_stats=None, danger_score=0.0):
        terrain_stats = terrain_stats or {}
        dist_score = 1.0 - float(np.clip(treasure["dist_norm"], 0.0, 1.0))
        readiness_score = float(np.clip(terrain_stats.get("readiness_score", 0.5), 0.0, 1.0))
        dead_end_risk = float(np.clip(terrain_stats.get("dead_end_risk", 0.0), 0.0, 1.0))
        trap_risk = float(np.clip(terrain_stats.get("trap_risk", 0.0), 0.0, 1.0))
        danger_score = float(np.clip(danger_score, 0.0, 1.0))
        safety_score = float(np.clip(0.55 * readiness_score + 0.25 * (1.0 - dead_end_risk) + 0.20 * (1.0 - trap_risk), 0.0, 1.0))
        loot_window = self.calc_treasure_approach_weight(danger_score)
        priority = float(
            np.clip(
                0.55 * dist_score
                + 0.30 * safety_score
                + 0.15 * loot_window
                - 0.35 * danger_score
                - 0.20 * dead_end_risk,
                0.0,
                1.0,
            )
        )
        return priority

    def get_priority_feats(self, organs, hero_pos, terrain_stats=None, danger_score=0.0):
        available_organs = self.build_available_organs(organs, hero_pos)
        treasures = [organ for organ in available_organs if int(organ["sub_type"]) == 1]
        if not treasures:
            return np.zeros(8, dtype=np.float32)

        scored_treasures = [
            (self.score_treasure_priority(treasure, terrain_stats=terrain_stats, danger_score=danger_score), treasure)
            for treasure in treasures
        ]
        scored_treasures.sort(key=lambda item: item[0], reverse=True)
        best_priority, best_treasure = scored_treasures[0]
        nearest_treasure = min(treasures, key=lambda treasure: treasure["dist_norm"])
        nearest_priority = self.score_treasure_priority(
            nearest_treasure,
            terrain_stats=terrain_stats,
            danger_score=danger_score,
        )
        readiness_score = float(np.clip((terrain_stats or {}).get("readiness_score", 0.5), 0.0, 1.0))
        safe_to_loot = float(
            best_priority >= 0.38
            and float(danger_score) <= TREASURE_DANGER_DAMP_END
            and (float(danger_score) <= 0.78 or readiness_score >= 0.75)
        )
        return np.array(
            [
                1.0,
                best_treasure["dist_norm"],
                best_treasure["dir_x"],
                best_treasure["dir_z"],
                best_priority,
                safe_to_loot,
                float(np.clip(best_priority - nearest_priority + 0.5, 0.0, 1.0)),
                float(np.clip(len(treasures) / MAX_TREASURE_COUNT, 0.0, 1.0)),
            ],
            dtype=np.float32,
        )

    def calc_buff_priority_weight(self, hero, nearest_treasure, nearest_buff, danger_score=0.0, trap_risk=0.0) -> float:
        if nearest_buff is None:
            return 0.0

        hero = hero or {}
        danger_score = float(np.clip(danger_score, 0.0, 1.0))
        trap_risk = float(np.clip(trap_risk, 0.0, 1.0))
        buff_remaining_time = float(hero.get("buff_remaining_time", 0.0))
        if buff_remaining_time <= 0.0:
            buff_priority_weight = 1.0
        elif buff_remaining_time <= 8.0:
            buff_priority_weight = 0.85
        else:
            buff_priority_weight = 0.45

        escape_need_scale = 1.0 + 0.75 * danger_score + 0.35 * trap_risk
        buff_priority_weight *= escape_need_scale

        if (
            nearest_treasure is not None
            and nearest_buff["dist_norm"] - nearest_treasure["dist_norm"] > BUFF_PRIORITY_FAR_DIFF_NORM
        ):
            danger_far_cap = BUFF_PRIORITY_FAR_CAP + 0.85 * danger_score + 0.35 * trap_risk
            buff_priority_weight = min(buff_priority_weight, danger_far_cap)
        return float(np.clip(buff_priority_weight, 0.0, 2.25))

    def calc_treasure_approach_weight(self, danger_score) -> float:
        danger_score = float(np.clip(danger_score, 0.0, 1.0))
        if danger_score <= TREASURE_DANGER_DAMP_START:
            return 1.0
        if danger_score >= TREASURE_DANGER_DAMP_END:
            return 0.0
        return float(
            1.0
            - (danger_score - TREASURE_DANGER_DAMP_START)
            / max(TREASURE_DANGER_DAMP_END - TREASURE_DANGER_DAMP_START, 1e-6)
        )

    def calc_reward(self, env_info, organs, hero_pos, hero=None, terrain_stats=None, danger_score=0.0):
        available_organs = self.build_available_organs(organs, hero_pos)
        available_treasure_count = sum(1 for organ in available_organs if int(organ["sub_type"]) == 1)
        available_buff_count = sum(1 for organ in available_organs if int(organ["sub_type"]) == 2)
        nearest_treasure = self.select_nearest_organ(available_organs, sub_type=1)
        nearest_buff = self.select_nearest_organ(available_organs, sub_type=2)
        treasures = [organ for organ in available_organs if int(organ["sub_type"]) == 1]
        terrain_stats = terrain_stats or {}
        trap_risk = float(terrain_stats.get("trap_risk", 0.0))
        readiness_score = float(terrain_stats.get("readiness_score", 0.0))
        target_treasure = None
        if treasures:
            target_treasure = max(
                treasures,
                key=lambda treasure: self.score_treasure_priority(
                    treasure,
                    terrain_stats=terrain_stats,
                    danger_score=danger_score,
                ),
            )
        approach_scale = 1.0
        if trap_risk >= 0.45:
            approach_scale = float(np.clip(1.0 - 0.5 * trap_risk, 0.5, 1.0))
        treasure_approach_weight = self.calc_treasure_approach_weight(danger_score)
        treasure_approach_scale = approach_scale * treasure_approach_weight
        if (
            float(danger_score) <= SAFE_TREASURE_DANGER_THRESHOLD
            and readiness_score >= SAFE_TREASURE_READINESS_THRESHOLD
        ):
            treasure_approach_scale *= 1.0 + SAFE_TREASURE_APPROACH_BONUS
        buff_approach_scale = max(approach_scale, 0.85 + 0.45 * float(np.clip(danger_score, 0.0, 1.0)))
        buff_priority_weight = self.calc_buff_priority_weight(
            hero=hero,
            nearest_treasure=nearest_treasure,
            nearest_buff=nearest_buff,
            danger_score=danger_score,
            trap_risk=trap_risk,
        )

        treasure_reward = 0.0
        buff_reward = 0.0
        target_treasure_key = None if target_treasure is None else target_treasure["key"]
        same_treasure_target = (
            target_treasure is not None
            and self.last_treasure_key == target_treasure_key
            and self.last_treasure_dist_norm is not None
        )
        if target_treasure is not None and same_treasure_target:
            # 判断这一帧是不是比上一帧更接近当前最高优先级宝箱
            close_ratio = max(
                0.0,
                (TREASURE_NEAR_DIST_NORM - target_treasure["dist_norm"]) / max(TREASURE_NEAR_DIST_NORM, 1e-6),
            )
            treasure_scale = TREASURE_APPROACH_REWARD_SCALE + TREASURE_NEAR_APPROACH_BONUS * close_ratio
            delta_dist = self.last_treasure_dist_norm - target_treasure["dist_norm"]

            if delta_dist > 0:
                # 安全时靠近宝箱给奖励；危险过高时暂停这类 shaping，避免顶怪贪箱。
                treasure_reward += treasure_approach_scale * treasure_scale * delta_dist
            else:
                # 高危险时允许主动远离宝箱去拉开身位，不因为宝箱目标给负反馈。
                treasure_reward += treasure_approach_scale * TREASURE_APPROACH_REWARD_SCALE * delta_dist
        if nearest_buff is not None and self.last_buff_dist_norm is not None:
            delta_buff_dist = self.last_buff_dist_norm - nearest_buff["dist_norm"]
            buff_reward += buff_approach_scale * buff_priority_weight * BUFF_APPROACH_REWARD_SCALE * delta_buff_dist

        # 拾取奖励，通过 treasures_collected 和 collected_buff 判断本帧是否真的完成了拾取
        treasures_collected = int(env_info.get("treasures_collected", 0))
        collected_buff = int(env_info.get("collected_buff", 0))

        treasure_gain = 0
        buff_gain = 0
        if self.last_treasures_collected is not None:
            treasure_gain = max(0, treasures_collected - self.last_treasures_collected)
        if self.last_collected_buff is not None:
            buff_gain = max(0, collected_buff - self.last_collected_buff)

        treasure_reward += treasure_gain * TREASURE_PICKUP_REWARD
        buff_reward += buff_gain * BUFF_PICKUP_REWARD

        # 已经非常接近宝箱却连续几步没有拿到，说明很可能在墙边抖动或绕不进去，给予额外惩罚
        treasure_stall_penalty = 0.0
        if target_treasure is not None and treasure_gain == 0:
            target_dist = float(target_treasure["dist_norm"])
            is_near_treasure = target_dist <= TREASURE_NEAR_DIST_NORM
            not_getting_closer = (
                same_treasure_target
                and target_dist >= self.last_treasure_dist_norm - 1e-6
            )
            if not_getting_closer and treasure_approach_weight > 0.0:
                self.treasure_no_progress_steps += 1
                no_progress_over = max(0, self.treasure_no_progress_steps - TREASURE_NO_PROGRESS_PENALTY_START)
                treasure_stall_penalty -= min(
                    treasure_approach_weight * TREASURE_NO_PROGRESS_PENALTY_SCALE * no_progress_over,
                    TREASURE_NO_PROGRESS_PENALTY_CAP,
                )
            else:
                self.treasure_no_progress_steps = 0

            if is_near_treasure and not_getting_closer and treasure_approach_weight > 0.0:
                self.treasure_stall_steps += 1
                treasure_stall_penalty -= min(
                    treasure_approach_weight * TREASURE_STALL_PENALTY_SCALE * self.treasure_stall_steps,
                    TREASURE_STALL_PENALTY_CAP,
                )
            else:
                self.treasure_stall_steps = 0
        else:
            self.treasure_stall_steps = 0
            self.treasure_no_progress_steps = 0

        # 在奖励结算后刷新缓存
        self.last_treasure_dist_norm = None if target_treasure is None else target_treasure["dist_norm"]
        self.last_treasure_key = target_treasure_key
        self.last_buff_dist_norm = None if nearest_buff is None else nearest_buff["dist_norm"]
        self.last_treasures_collected = treasures_collected
        self.last_collected_buff = collected_buff
        return {
            "treasure_reward": float(treasure_reward),
            "buff_reward": float(buff_reward),
            "treasure_stall_penalty": float(treasure_stall_penalty),
            "buff_priority_weight": float(buff_priority_weight),
            "treasure_approach_weight": float(treasure_approach_weight),
            "buffs_collected": int(collected_buff),
            "available_treasure_count": int(available_treasure_count),
            "available_buff_count": int(available_buff_count),
            "nearest_buff_dist_norm": float(1.0 if nearest_buff is None else nearest_buff["dist_norm"]),
        }
