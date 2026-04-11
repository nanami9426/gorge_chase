class ExploreProcessor:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.visited_grids = set() # 访问过的格子
        self.grid_size = 8 # 地图128*128，把它切成16*16的格子，每格长8
        self.last_hero_pos = None
        self.stall_steps = 0
    
    def calc_reward(self, hero_pos) -> float:
        # 探索新的区域
        grid_x = hero_pos["x"] // self.grid_size
        grid_z = hero_pos["z"] // self.grid_size
        grid = (grid_x, grid_z)
        if grid not in self.visited_grids:
            explore_reward = 0.02
            self.visited_grids.add(grid)
        else:
            explore_reward = 0.0

        # 防止原地逗留
        cur_pos = (hero_pos["x"], hero_pos["z"])
        if self.last_hero_pos is None:
            moved = True
        else:
            moved = cur_pos != self.last_hero_pos
        if moved:
            self.stall_steps = 0
        else:
            self.stall_steps += 1
        stall_penalty = -min(0.002 * self.stall_steps, 0.2)
        self.last_hero_pos = cur_pos
        return explore_reward + stall_penalty
    