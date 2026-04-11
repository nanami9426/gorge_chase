MOVE_ACTION_NUM = 8
BLOCKED_MOVE_PENALTY = -0.03


class MoveProcessor:
    def __init__(self):
        self.reset()

    def reset(self):
        # 记录上一帧 8 个移动方向是否真的可走，用于结算上一帧动作是否撞墙
        self.last_move_mask = [1] * MOVE_ACTION_NUM

    def is_cell_passable(self, map_info, row, col) -> bool:
        # 判断局部地图中的某个格子是否可通行
        if map_info is None:
            return True
        if row < 0 or row >= len(map_info):
            return False
        if col < 0 or col >= len(map_info[0]):
            return False
        return bool(map_info[row][col] != 0)

    def build_move_mask(self, map_info):
        """
        用于避免模型选到明显撞墙的动作
        1. 直线方向：目标格可通行即可移动
        2. 斜向方向：目标格可通行，且两条相邻边至少一条可通行
        """
        if map_info is None or not map_info or not map_info[0]:
            return [1] * MOVE_ACTION_NUM

        center_row = len(map_info) // 2
        center_col = len(map_info[0]) // 2

        east = self.is_cell_passable(map_info, center_row, center_col + 1)
        north = self.is_cell_passable(map_info, center_row - 1, center_col)
        west = self.is_cell_passable(map_info, center_row, center_col - 1)
        south = self.is_cell_passable(map_info, center_row + 1, center_col)

        north_east = self.is_cell_passable(map_info, center_row - 1, center_col + 1) and (north or east)
        north_west = self.is_cell_passable(map_info, center_row - 1, center_col - 1) and (north or west)
        south_west = self.is_cell_passable(map_info, center_row + 1, center_col - 1) and (south or west)
        south_east = self.is_cell_passable(map_info, center_row + 1, center_col + 1) and (south or east)

        return [
            int(east),
            int(north_east),
            int(north),
            int(north_west),
            int(west),
            int(south_west),
            int(south),
            int(south_east),
        ]

    def mask_legal_action(self, legal_action, map_info):
        # 把明显会撞墙的移动动作从合法动作中剔除
        move_mask = self.build_move_mask(map_info)
        masked_action = list(legal_action)
        for idx in range(min(MOVE_ACTION_NUM, len(masked_action))):
            masked_action[idx] = int(masked_action[idx] and move_mask[idx])

        # 理论上不会 8 个方向全不可走；如果真的发生，退回原始动作，避免策略彻底无路可选
        if sum(masked_action[:MOVE_ACTION_NUM]) == 0:
            return list(legal_action), move_mask
        return masked_action, move_mask

    def calc_reward(self, last_action, move_mask) -> float:
        move_reward = 0.0
        if last_action is not None:
            action_idx = int(last_action)
            if 0 <= action_idx < MOVE_ACTION_NUM and self.last_move_mask[action_idx] == 0:
                # 如果上一帧是移动，并且那个方向走不通，那就是撞墙了
                move_reward = BLOCKED_MOVE_PENALTY

        self.last_move_mask = list(move_mask)
        return move_reward
