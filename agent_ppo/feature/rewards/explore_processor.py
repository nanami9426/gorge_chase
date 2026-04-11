import math
from collections import deque


EXPLORE_GRID_REWARD = 0.03
MOVE_REWARD_SCALE = 0.004
MOVE_REWARD_CAP = 2.0
REVISIT_GRID_PENALTY = -0.0015
STALL_PENALTY_SCALE = 0.003
STALL_PENALTY_CAP = 0.12
RECENT_GRID_WINDOW = 8


class ExploreProcessor:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.visited_grids = set() # 访问过的格子
        self.grid_size = 8 # 地图128*128，把它切成16*16的格子，每格长8
        self.last_hero_pos = None
        self.stall_steps = 0
        self.recent_grids = deque(maxlen=RECENT_GRID_WINDOW)
    
    def calc_reward(self, hero_pos) -> float:
        # 探索新的区域
        grid_x = hero_pos["x"] // self.grid_size
        grid_z = hero_pos["z"] // self.grid_size
        grid = (grid_x, grid_z)
        if grid not in self.visited_grids:
            explore_reward = EXPLORE_GRID_REWARD
            self.visited_grids.add(grid)
        else:
            explore_reward = 0.0
            
        # 奖励真实位移，抑制撞墙或小范围抖动
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

        if moved:
            self.stall_steps = 0
        else:
            self.stall_steps += 1
        stall_penalty = -min(STALL_PENALTY_SCALE * self.stall_steps, STALL_PENALTY_CAP)

        self.last_hero_pos = cur_pos
        self.recent_grids.append(grid)
        return explore_reward + revisit_penalty + move_reward + stall_penalty
    
