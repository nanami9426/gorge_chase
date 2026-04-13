import numpy as np


MAP_SIZE = 128.0
MAX_DIST_BUCKET = 5.0
MAX_MAP_DISTANCE = MAP_SIZE * 1.41
TREASURE_APPROACH_REWARD_SCALE = 0.30
TREASURE_NEAR_APPROACH_BONUS = 0.90
BUFF_APPROACH_REWARD_SCALE = 0.20
TREASURE_PICKUP_REWARD = 2.5
BUFF_PICKUP_REWARD = 1.25
TREASURE_NEAR_DIST_NORM = 6.0 / MAX_MAP_DISTANCE # 近距离宝箱的阈值，大约是6格以内
TREASURE_STALL_PENALTY_SCALE = 0.02
TREASURE_STALL_PENALTY_CAP = 0.20
BUFF_PRIORITY_FAR_DIFF_NORM = 0.06
BUFF_PRIORITY_FAR_CAP = 0.5
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
        self.last_buff_dist_norm = None
        self.last_treasures_collected = None
        self.last_collected_buff = None
        self.treasure_stall_steps = 0

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

    def calc_buff_priority_weight(self, hero, nearest_treasure, nearest_buff) -> float:
        if nearest_buff is None:
            return 0.0

        hero = hero or {}
        buff_remaining_time = float(hero.get("buff_remaining_time", 0.0))
        if buff_remaining_time <= 0.0:
            buff_priority_weight = 1.0
        elif buff_remaining_time <= 8.0:
            buff_priority_weight = 0.7
        else:
            buff_priority_weight = 0.35

        if (
            nearest_treasure is not None
            and nearest_buff["dist_norm"] - nearest_treasure["dist_norm"] > BUFF_PRIORITY_FAR_DIFF_NORM
        ):
            buff_priority_weight = min(buff_priority_weight, BUFF_PRIORITY_FAR_CAP)
        return float(buff_priority_weight)

    def calc_reward(self, env_info, organs, hero_pos, hero=None, terrain_stats=None, danger_score=0.0):
        available_organs = self.build_available_organs(organs, hero_pos)
        nearest_treasure = self.select_nearest_organ(available_organs, sub_type=1)
        nearest_buff = self.select_nearest_organ(available_organs, sub_type=2)
        terrain_stats = terrain_stats or {}
        trap_risk = float(terrain_stats.get("trap_risk", 0.0))
        approach_scale = 1.0
        if trap_risk >= 0.45:
            approach_scale = float(np.clip(1.0 - 0.5 * trap_risk, 0.5, 1.0))
        buff_priority_weight = self.calc_buff_priority_weight(
            hero=hero,
            nearest_treasure=nearest_treasure,
            nearest_buff=nearest_buff,
        )

        treasure_reward = 0.0
        buff_reward = 0.0
        if nearest_treasure is not None and self.last_treasure_dist_norm is not None:
            # 判断这一帧是不是比上一帧更接近宝箱
            close_ratio = max(
                0.0,
                (TREASURE_NEAR_DIST_NORM - nearest_treasure["dist_norm"]) / max(TREASURE_NEAR_DIST_NORM, 1e-6),
            )
            treasure_scale = TREASURE_APPROACH_REWARD_SCALE + TREASURE_NEAR_APPROACH_BONUS * close_ratio
            delta_dist = self.last_treasure_dist_norm - nearest_treasure["dist_norm"]
            
            if delta_dist > 0:
                # 靠近宝箱时放大奖励
                treasure_reward += approach_scale * treasure_scale * delta_dist
            else:
                # 若为了绕墙暂时远离，只按基础系数处罚，避免负反馈过重
                treasure_reward += TREASURE_APPROACH_REWARD_SCALE * delta_dist
        if nearest_buff is not None and self.last_buff_dist_norm is not None:
            delta_buff_dist = self.last_buff_dist_norm - nearest_buff["dist_norm"]
            buff_reward += approach_scale * buff_priority_weight * BUFF_APPROACH_REWARD_SCALE * delta_buff_dist

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
        if nearest_treasure is not None and treasure_gain == 0:
            is_near_treasure = nearest_treasure["dist_norm"] <= TREASURE_NEAR_DIST_NORM
            not_getting_closer = (
                self.last_treasure_dist_norm is not None
                and nearest_treasure["dist_norm"] >= self.last_treasure_dist_norm - 1e-6
            )
            if is_near_treasure and not_getting_closer:
                self.treasure_stall_steps += 1
                treasure_stall_penalty = -min(
                    TREASURE_STALL_PENALTY_SCALE * self.treasure_stall_steps,
                    TREASURE_STALL_PENALTY_CAP,
                )
            else:
                self.treasure_stall_steps = 0
        else:
            self.treasure_stall_steps = 0

        # 在奖励结算后刷新缓存
        self.last_treasure_dist_norm = None if nearest_treasure is None else nearest_treasure["dist_norm"]
        self.last_buff_dist_norm = None if nearest_buff is None else nearest_buff["dist_norm"]
        self.last_treasures_collected = treasures_collected
        self.last_collected_buff = collected_buff
        return {
            "treasure_reward": float(treasure_reward),
            "buff_reward": float(buff_reward),
            "treasure_stall_penalty": float(treasure_stall_penalty),
            "buff_priority_weight": float(buff_priority_weight),
            "buffs_collected": int(collected_buff),
        }
