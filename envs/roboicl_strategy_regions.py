"""Single-object route experiment; virtual regions never alter rendered RGB."""
import numpy as np

from .roboicl_select_curobo import roboicl_select_curobo


class roboicl_strategy_regions(roboicl_select_curobo):
    policy_budget_steps = 6000
    recording_substeps = 5
    task_name = 'roboicl_strategy_regions'

    def validate_object_specs(self, specs):
        if len(specs) != 1:
            raise ValueError('The first strategy experiment requires exactly one object')

    def setup_demo(self, **kwargs):
        self.strategy = kwargs['strategy']
        regions = self.strategy['regions']
        if not regions or len({r['id'] for r in regions}) != len(regions):
            raise ValueError('Route regions must have unique IDs')
        for region in regions:
            center = np.asarray(region['center_m'], dtype=float)
            if center.shape != (3,) or not np.isfinite(center).all():
                raise ValueError('Expected finite3D region center')
            if region.get('shape', 'box') == 'sphere':
                radius = float(region['radius_m'])
                if not np.isfinite(radius) or radius <= 0:
                    raise ValueError('Expected positive finite sphere radius')
            else:
                half = np.asarray(region['half_size_m'], dtype=float)
                if half.shape != (3,) or not np.isfinite(half).all() or np.any(half <= 0):
                    raise ValueError('Expected positive finite box half sizes')
        super().setup_demo(**kwargs)
