import numpy as np


# 闪现动作在离散动作空间中的下标范围：[8, 15]
FLASH_ACTION_START = 8
FLASH_ACTION_END = 16
# 只有危险度达到该阈值，才允许使用闪现
FLASH_DANGER_THRESHOLD = 0.50
SPEEDUP_FLASH_DANGER_THRESHOLD = 0.42
STRATEGIC_FLASH_DEAD_END_THRESHOLD = 0.45
STRATEGIC_FLASH_SCORE_THRESHOLD = 0.68
SPEEDUP_STRATEGIC_FLASH_DEAD_END_THRESHOLD = 0.32
SPEEDUP_STRATEGIC_FLASH_SCORE_THRESHOLD = 0.58
SPEEDUP_STRATEGIC_READINESS_THRESHOLD = 0.52
# 在安全状态下误用闪现时的惩罚强度
SAFE_FLASH_PENALTY_SCALE = 0.22
FLASH_ESCAPE_REWARD_SCALE = 0.45
FLASH_TRAP_ESCAPE_REWARD_SCALE = 0.25
FLASH_DISTANCE_GAIN_REWARD_SCALE = 0.18
FAILED_FLASH_PENALTY_SCALE = 0.16
PREDICTION_APPROACH_DANGER_SCALE = 0.18
PREDICTION_GAIN_DANGER_SCALE = 0.22


class FlashProcessor:
    def __init__(self):
        self.reset()

    def reset(self):
        # 缓存上一帧的危险度，用来判断上一帧做出的闪现动作是否属于误用
        self.last_danger_score = 0.0
        self.last_trap_risk = 0.0
        self.last_dead_end_risk = 0.0
        self.last_best_flash_score = 0.0
        self.last_min_dist_norm = None
        self.last_max_monster_speed = 1

    def get_nearest_monster_feat(self, monster_feats):
        # 直接取最近怪物的整条特征，供多个逻辑复用
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
        active_feats = [
            feat
            for feat in monster_feats
            if float(feat[0]) > 0.0 or float(feat[3]) > 0.0 or float(feat[4]) < 0.999
        ]
        max_speed_norm = max((float(feat[3]) for feat in active_feats), default=nearest_speed_norm)
        monster_count_pressure = min(max(len(active_feats) - 1, 0), 1)
        fast_pressure = max(0.0, (max_speed_norm - 0.25) / 0.75)
        near_pressure = 1.0 - min_dist_norm
        danger_score = (
            0.58 * near_pressure
            + 0.28 * max_speed_norm
            + 0.06 * nearest_in_view
            + 0.10 * fast_pressure
            + 0.05 * monster_count_pressure
        )
        return float(np.clip(danger_score, 0.0, 1.0))

    def calc_danger_score(self, monster_feats, terrain_stats=None, prediction_feat=None) -> float:
        # 在基础怪物危险度上叠加“贴墙/死角”困境，避免等到贴脸才开始逃
        base_danger = self.calc_base_danger_score(monster_feats)
        min_dist_norm, _, _ = self.get_nearest_monster_stats(monster_feats)
        active_feats = [
            feat
            for feat in monster_feats
            if float(feat[0]) > 0.0 or float(feat[3]) > 0.0 or float(feat[4]) < 0.999
        ]
        max_speed_norm = max((float(feat[3]) for feat in active_feats), default=0.0)
        fast_pressure = max(0.0, (max_speed_norm - 0.25) / 0.75)
        near_pressure = 1.0 - min_dist_norm
        terrain_stats = terrain_stats or {}
        trap_risk = float(terrain_stats.get("trap_risk", 0.0))
        readiness_score = float(terrain_stats.get("readiness_score", 1.0))
        readiness_gap = max(0.0, 0.62 - readiness_score) / 0.62
        danger_score = base_danger + trap_risk * (0.20 + 0.45 * (1.0 - min_dist_norm))
        danger_score += fast_pressure * (0.08 + 0.15 * trap_risk + 0.12 * readiness_gap)
        if prediction_feat is not None and len(prediction_feat) >= 6:
            approach_pressure = float(np.clip(prediction_feat[1], 0.0, 1.0))
            worst_pred_danger = float(np.clip(prediction_feat[5], 0.0, 1.0))
            pred_danger_gain = max(0.0, worst_pred_danger - near_pressure)
            danger_score += PREDICTION_APPROACH_DANGER_SCALE * approach_pressure
            danger_score += PREDICTION_GAIN_DANGER_SCALE * pred_danger_gain * (0.60 + 0.40 * fast_pressure)
        return float(np.clip(danger_score, 0.0, 1.0))

    def should_allow_flash(self, danger_score, terrain_stats=None, max_monster_speed=1) -> bool:
        # 只有在足够危险时，才开放闪现动作给策略选择
        if float(danger_score) >= FLASH_DANGER_THRESHOLD:
            return True

        terrain_stats = terrain_stats or {}
        dead_end_risk = float(terrain_stats.get("dead_end_risk", 0.0))
        readiness_score = float(terrain_stats.get("readiness_score", 1.0))
        best_flash_score = float(terrain_stats.get("best_flash_score", 0.0))
        if (
            int(max_monster_speed) > 1
            and best_flash_score >= SPEEDUP_STRATEGIC_FLASH_SCORE_THRESHOLD
            and (
                float(danger_score) >= SPEEDUP_FLASH_DANGER_THRESHOLD
                or dead_end_risk >= SPEEDUP_STRATEGIC_FLASH_DEAD_END_THRESHOLD
                or readiness_score <= SPEEDUP_STRATEGIC_READINESS_THRESHOLD
            )
        ):
            return True
        if (
            int(max_monster_speed) > 1
            and dead_end_risk >= STRATEGIC_FLASH_DEAD_END_THRESHOLD
            and best_flash_score >= STRATEGIC_FLASH_SCORE_THRESHOLD
        ):
            return True
        return False

    def mask_legal_action(self, legal_action, danger_score, terrain_stats=None, max_monster_speed=1):
        # 根据危险度对合法动作做二次过滤，如果当前不危险，就把 8~15 的闪现动作全部屏蔽，避免模型在安全期乱交技能。
        if self.should_allow_flash(
            danger_score=danger_score,
            terrain_stats=terrain_stats,
            max_monster_speed=max_monster_speed,
        ):
            return list(legal_action)

        masked_action = list(legal_action)
        for idx in range(FLASH_ACTION_START, min(FLASH_ACTION_END, len(masked_action))):
            masked_action[idx] = 0
        return masked_action

    def calc_reward(self, last_action, danger_score, monster_feats=None, terrain_stats=None, max_monster_speed=1) -> float:
        prev_danger_score = self.last_danger_score
        prev_trap_risk = self.last_trap_risk
        prev_dead_end_risk = self.last_dead_end_risk
        prev_best_flash_score = self.last_best_flash_score
        prev_min_dist_norm = self.last_min_dist_norm
        prev_max_monster_speed = self.last_max_monster_speed
        cur_min_dist_norm, _, _ = self.get_nearest_monster_stats(monster_feats or [])
        terrain_stats = terrain_stats or {}
        cur_trap_risk = float(terrain_stats.get("trap_risk", 0.0))
        cur_dead_end_risk = float(terrain_stats.get("dead_end_risk", 0.0))
        cur_best_flash_score = float(terrain_stats.get("best_flash_score", 0.0))
        flash_reward = 0.0

        if (
            last_action is not None
            and FLASH_ACTION_START <= int(last_action) < FLASH_ACTION_END  # 上一帧使用闪现
        ):
            strategic_flash = (
                prev_dead_end_risk >= STRATEGIC_FLASH_DEAD_END_THRESHOLD
                and prev_best_flash_score >= STRATEGIC_FLASH_SCORE_THRESHOLD
            ) or (
                int(prev_max_monster_speed) > 1
                and prev_best_flash_score >= SPEEDUP_STRATEGIC_FLASH_SCORE_THRESHOLD
                and (
                    prev_danger_score >= SPEEDUP_FLASH_DANGER_THRESHOLD
                    or prev_dead_end_risk >= SPEEDUP_STRATEGIC_FLASH_DEAD_END_THRESHOLD
                )
            )
            if prev_danger_score < FLASH_DANGER_THRESHOLD and not strategic_flash:
                # 如果上一帧不危险却使用了闪现，就给惩罚
                safe_margin = FLASH_DANGER_THRESHOLD - prev_danger_score
                flash_reward = -SAFE_FLASH_PENALTY_SCALE * (0.5 + safe_margin)
            else:
                danger_drop = max(0.0, prev_danger_score - float(danger_score))
                trap_drop = max(0.0, prev_trap_risk - cur_trap_risk)
                dist_gain = 0.0
                if prev_min_dist_norm is not None:
                    dist_gain = max(0.0, cur_min_dist_norm - prev_min_dist_norm)
                flash_reward += (
                    FLASH_ESCAPE_REWARD_SCALE * danger_drop
                    + FLASH_TRAP_ESCAPE_REWARD_SCALE * trap_drop
                    + FLASH_DISTANCE_GAIN_REWARD_SCALE * dist_gain
                )
                if danger_drop <= 0.02 and float(danger_score) > prev_danger_score:
                    flash_reward -= FAILED_FLASH_PENALTY_SCALE * (float(danger_score) - prev_danger_score)

        self.last_danger_score = float(danger_score)
        self.last_trap_risk = cur_trap_risk
        self.last_dead_end_risk = cur_dead_end_risk
        self.last_best_flash_score = cur_best_flash_score
        self.last_min_dist_norm = cur_min_dist_norm
        self.last_max_monster_speed = int(max_monster_speed)
        return flash_reward
