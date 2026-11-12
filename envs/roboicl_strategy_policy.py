"""Single-object strategy scene for policy execution without an expert planner."""
from .roboicl_select_curobo import roboicl_select


class roboicl_strategy_policy(roboicl_select):
    def validate_object_specs(self, specs):
        if len(specs) != 1:
            raise ValueError('This policy benchmark requires one object')

    def setup_demo(self, **kwargs):
        super().setup_demo(**kwargs)
        if self.continuous_pick_place:
            for joint in self.robot.left_arm_joints:
                joint.set_drive_property(stiffness=5000., damping=500.)
