import numpy as np


# 闪现动作在离散动作空间中的下标范围：[8, 15]
FLASH_ACTION_START = 8
FLASH_ACTION_END = 16
# 只有危险度达到该阈值，才允许使用闪现
FLASH_DANGER_THRESHOLD = 0.50
# 在安全状态下误用闪现时的惩罚强度
SAFE_FLASH_PENALTY_SCALE = 0.18


class FlashProcessor:
    def __init__(self):
        self.reset()

    def reset(self):
        # 缓存上一帧的危险度，用来判断上一帧做出的闪现动作是否属于误用
        self.last_danger_score = 0.0

    def get_nearest_monster_feat(self, monster_feats):
        if not monster_feats:
            return None
        return min(monster_feats, key=lambda feat: float(feat[4]))

    def get_nearest_monster_stats(self, monster_feats):
        """
        取最近怪物的关键信息 [is_in_view, dir_x, dir_z, speed_norm, dist_norm]
        """
        nearest_feat = self.get_nearest_monster_feat(monster_feats)
        if nearest_feat is None:
            return 1.0, 0.0, 0.0

        return float(nearest_feat[4]), float(nearest_feat[3]), float(nearest_feat[0])

    def calc_base_danger_score(self, monster_feats) -> float:
        """
        1. 距离越近，危险度越高
        2. 怪物越快，危险度越高
        3. 怪物已经进入视野时，再额外增加危险度
        """
        min_dist_norm, nearest_speed_norm, nearest_in_view = self.get_nearest_monster_stats(monster_feats)
        near_pressure = 1.0 - min_dist_norm
        danger_score = 0.75 * near_pressure + 0.20 * nearest_speed_norm + 0.05 * nearest_in_view
        return float(np.clip(danger_score, 0.0, 1.0))

    def calc_danger_score(self, monster_feats, wall_pressure=0.0, corner_pressure=0.0) -> float:
        base_danger = self.calc_base_danger_score(monster_feats)
        danger_score = base_danger + 0.15 * float(wall_pressure) + 0.10 * float(corner_pressure)
        return float(np.clip(danger_score, 0.0, 1.0))

    def should_allow_flash(self, danger_score) -> bool:
        # 只有在足够危险时，才开放闪现动作给策略选择
        return float(danger_score) >= FLASH_DANGER_THRESHOLD

    def mask_legal_action(self, legal_action, danger_score):
        # 根据危险度对合法动作做二次过滤，如果当前不危险，就把 8~15 的闪现动作全部屏蔽，避免模型在安全期乱交技能。
        if self.should_allow_flash(danger_score):
            return list(legal_action)

        masked_action = list(legal_action)
        for idx in range(FLASH_ACTION_START, min(FLASH_ACTION_END, len(masked_action))):
            masked_action[idx] = 0
        return masked_action

    def calc_reward(self, last_action, danger_score) -> float:
        prev_danger_score = self.last_danger_score
        flash_reward = 0.0

        if (
            last_action is not None
            and FLASH_ACTION_START <= int(last_action) < FLASH_ACTION_END
            and prev_danger_score < FLASH_DANGER_THRESHOLD
        ):
            # 如果上一帧不危险却使用了闪现，就给惩罚
            safe_margin = FLASH_DANGER_THRESHOLD - prev_danger_score
            flash_reward = -SAFE_FLASH_PENALTY_SCALE * (0.5 + safe_margin)
        self.last_danger_score = float(danger_score)
        return flash_reward
