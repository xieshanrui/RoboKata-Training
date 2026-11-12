"""Task-specific fixed visual surfaces for RoboICL scene experiments."""
import re
from pathlib import Path

import sapien

from .roboicl_select_curobo import roboicl_select
from .utils import create_box, create_table


class roboicl_select_scene(roboicl_select):
    def setup_demo(self, **kwargs):
        preset = kwargs.get('roboicl_scene_preset_config')
        if preset is not None:
            randomization = kwargs['domain_randomization']
            if any(randomization.get(key) for key in (
                    'random_background', 'random_light', 'cluttered_table',
                    'random_table_height', 'random_head_camera_dis')):
                raise ValueError('Scene texture preset requires fixed lighting, camera and geometry')
            for field in ('wall_texture', 'table_texture'):
                texture = preset[field]
                if texture is not None and (not re.fullmatch(r'(seen|unseen)/\d+', texture)
                                            or not (Path('assets/background_texture') / f'{texture}.png').is_file()):
                    raise ValueError(f'Invalid {field} in scene preset')
        self.scene_preset = preset
        return super().setup_demo(**kwargs)

    def create_table_and_wall(self, table_xy_bias=[0, 0], table_height=0.74):
        if self.scene_preset is None:
            return super().create_table_and_wall(table_xy_bias, table_height)
        self.table_xy_bias = table_xy_bias
        self.wall_texture = self.scene_preset['wall_texture']
        self.table_texture = self.scene_preset['table_texture']
        table_height += self.table_z_bias
        self.wall = create_box(self.scene, sapien.Pose(p=[0, 1, 1.5]),
            half_size=[3, 0.6, 1.5], color=(1, 0.9, 0.9), name='wall',
            texture_id=self.wall_texture, is_static=True)
        self.table = create_table(self.scene,
            sapien.Pose(p=[table_xy_bias[0], table_xy_bias[1], table_height]),
            length=1.2, width=0.7, height=table_height, thickness=0.05,
            is_static=True, texture_id=self.table_texture)
