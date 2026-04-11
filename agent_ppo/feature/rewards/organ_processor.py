import numpy as np


MAP_SIZE = 128.0
MAX_DIST_BUCKET = 5.0
MAX_MAP_DISTANCE = MAP_SIZE * 1.41
ORGAN_APPROACH_REWARD_SCALE = 0.08
TREASURE_PICKUP_REWARD = 1.0
BUFF_PICKUP_REWARD = 0.5
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
        self.last_target_key = None
        self.last_target_dist_norm = None
        self.last_treasures_collected = None
        self.last_collected_buff = None

    def calc_reward(self, env_info, organs, hero_pos):
        available_organs = self.build_available_organs(organs, hero_pos)
        organ_reward = self.calc_reward(env_info, available_organs)
        return organ_reward
    
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

    def encode_organ_feat(self, organ_info):
        # 如果当前类型没有可拾取目标，则返回全零
        if organ_info is None:
            return np.zeros(4, dtype=np.float32)
        return np.array(
            # has_target, dist_norm, dir_x, dir_z
            [1.0, organ_info["dist_norm"], organ_info["dir_x"], organ_info["dir_z"]],
            dtype=np.float32,
        )

    def calc_reward(self, env_info, available_organs) -> float:
        current_target = self.select_nearest_organ(available_organs)
        # 接近奖励，仅当当前主目标和上一帧主目标是同一个实体时，才按距离差值发奖励。
        approach_reward = 0.0
        if (
            current_target is not None
            and self.last_target_key == current_target["key"]
            and self.last_target_dist_norm is not None
        ):
            approach_reward = ORGAN_APPROACH_REWARD_SCALE * (
                self.last_target_dist_norm - current_target["dist_norm"]
            )

        # 拾取奖励，通过 treasures_collected 和 collected_buff 判断本帧是否真的完成了拾取
        treasures_collected = int(env_info.get("treasures_collected", 0))
        collected_buff = int(env_info.get("collected_buff", 0))

        treasure_gain = 0
        buff_gain = 0
        if self.last_treasures_collected is not None:
            treasure_gain = max(0, treasures_collected - self.last_treasures_collected)
        if self.last_collected_buff is not None:
            buff_gain = max(0, collected_buff - self.last_collected_buff)

        pickup_reward = treasure_gain * TREASURE_PICKUP_REWARD + buff_gain * BUFF_PICKUP_REWARD

        # 在奖励结算后刷新缓存
        if current_target is None:
            self.last_target_key = None
            self.last_target_dist_norm = None
        else:
            self.last_target_key = current_target["key"]
            self.last_target_dist_norm = current_target["dist_norm"]

        self.last_treasures_collected = treasures_collected
        self.last_collected_buff = collected_buff
        return approach_reward + pickup_reward
