import numpy as np

from agent_ppo.conf.conf import Config


class SpatialFeatureEncoder:
    """Encode local walls, monsters, treasures, and buffs into a spatial tensor."""

    CHANNEL_WALL = 0
    CHANNEL_MONSTER = 1
    CHANNEL_TREASURE = 2
    CHANNEL_BUFF = 3

    def __init__(self):
        self.map_size = Config.SPATIAL_MAP_SIZE
        self.map_radius = self.map_size // 2
        self.channel_num = Config.SPATIAL_CHANNELS

    def normalize_map_info(self, map_info):
        """Center-crop or pad local map to a fixed passability grid."""
        if map_info is None or not map_info or not map_info[0]:
            return None

        src_rows = len(map_info)
        src_cols = len(map_info[0])
        src_center_row = src_rows // 2
        src_center_col = src_cols // 2
        dst_center = self.map_radius
        normalized = np.zeros((self.map_size, self.map_size), dtype=np.int32)

        for dst_row in range(self.map_size):
            for dst_col in range(self.map_size):
                src_row = src_center_row + (dst_row - dst_center)
                src_col = src_center_col + (dst_col - dst_center)
                if 0 <= src_row < src_rows and 0 <= src_col < src_cols:
                    normalized[dst_row, dst_col] = int(map_info[src_row][src_col] != 0)

        return normalized.tolist()

    def project_pos_to_local_cell(self, obj_pos, hero_pos):
        """Project world position to the centered local grid."""
        if not isinstance(obj_pos, dict) or "x" not in obj_pos or "z" not in obj_pos:
            return None

        dx = int(round(float(obj_pos["x"]) - float(hero_pos["x"])))
        dz = int(round(float(obj_pos["z"]) - float(hero_pos["z"])))
        if abs(dx) > self.map_radius or abs(dz) > self.map_radius:
            return None

        center = self.map_radius
        row = center - dz
        col = center + dx
        if not (0 <= row < self.map_size and 0 <= col < self.map_size):
            return None
        return row, col

    def build_spatial_feature(self, normalized_map_info, monsters, organs, hero_pos):
        """Build CxHxW tensor for walls, monsters, treasures, and buffs."""
        spatial_feat = np.zeros(
            (self.channel_num, self.map_size, self.map_size),
            dtype=np.float32,
        )

        if normalized_map_info is not None:
            passable = np.array(normalized_map_info, dtype=np.float32)
            spatial_feat[self.CHANNEL_WALL] = 1.0 - passable

        for monster in monsters:
            cell = self.project_pos_to_local_cell(monster.get("pos"), hero_pos)
            if cell is None:
                continue
            spatial_feat[self.CHANNEL_MONSTER, cell[0], cell[1]] = 1.0

        for organ in organs:
            if int(organ.get("status", 0)) != 1:
                continue

            sub_type = int(organ.get("sub_type", 0))
            if sub_type == 1:
                channel = self.CHANNEL_TREASURE
            elif sub_type == 2:
                channel = self.CHANNEL_BUFF
            else:
                continue

            cell = self.project_pos_to_local_cell(organ.get("pos"), hero_pos)
            if cell is None:
                continue
            spatial_feat[channel, cell[0], cell[1]] = 1.0

        return spatial_feat

    def encode(self, map_info, monsters, organs, hero_pos):
        """Return normalized map for logic and the spatial tensor for CNN input."""
        normalized_map_info = self.normalize_map_info(map_info)
        spatial_feat = self.build_spatial_feature(
            normalized_map_info=normalized_map_info,
            monsters=monsters,
            organs=organs,
            hero_pos=hero_pos,
        )
        return normalized_map_info, spatial_feat
