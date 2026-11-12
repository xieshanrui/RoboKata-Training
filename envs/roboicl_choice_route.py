"""Bounded dual-arm feasibility task; native RoboTwin motion primitives.

The scene manifest is independent of robot initialization RNG. This is a new
task namespace, not a replacement of the existing single-arm dataset expert.
"""
import numpy as np
import sapien
import json
import re
from pathlib import Path

from ._base_task import Base_Task
from .utils import ArmTag, create_actor, create_box, create_table


class roboicl_choice_route(Base_Task):
    def setup_demo(self, **kwargs):
        self.case = kwargs.pop("case_manifest")
        self.scene_preset = self.case.get('scene_preset')
        if self.scene_preset is not None:
            if any(kwargs['domain_randomization'].get(k) for k in (
                    'random_background', 'random_light', 'cluttered_table',
                    'random_table_height', 'random_head_camera_dis')):
                raise ValueError('Explicit visual profiles require disabled implicit randomization')
            for key in ('wall_texture', 'table_texture'):
                value = self.scene_preset[key]
                if value is not None and (not re.fullmatch(r'(seen|unseen)/\d+', value)
                        or not (Path('assets/background_texture') / f'{value}.png').is_file()):
                    raise ValueError(f'Invalid scene texture: {key}')
        self.stage = "initial"
        self.waypoint_results = []
        super()._init_task_env_(**kwargs)
        self.initial_poses = [a.get_pose() for a in self.objects]
        self.target_actor = self.objects[self.case["intent_index"]]
        self.arm = ArmTag(self.case.get("arm", "left" if self.target_actor.get_pose().p[0] < 0 else "right"))
        self.lifted = False

    def create_table_and_wall(self, table_xy_bias=[0, 0], table_height=.74):
        if self.scene_preset is None:
            return super().create_table_and_wall(table_xy_bias, table_height)
        self.table_xy_bias = table_xy_bias
        self.wall_texture = self.scene_preset['wall_texture']
        self.table_texture = self.scene_preset['table_texture']
        table_height += self.table_z_bias
        self.wall = create_box(self.scene, sapien.Pose(p=[0, 1, 1.5]),
            half_size=[3, .6, 1.5], color=(1, .9, .9), name='wall',
            texture_id=self.wall_texture, is_static=True)
        self.table = create_table(self.scene,
            sapien.Pose(p=[*table_xy_bias, table_height]), length=1.2, width=.7,
            height=table_height, thickness=.05, is_static=True,
            texture_id=self.table_texture)

    def load_actors(self):
        self.objects = []
        for spec in self.case["objects"]:
            actor = create_actor(self, sapien.Pose(spec["position"], spec["quaternion"]),
                                 spec["asset"], model_id=spec["model_id"], convex=True)
            if actor.config is None:
                metadata = Path("assets/objects") / spec["asset"] / f"model_data{spec['model_id']}.json"
                actor.config = json.loads(metadata.read_text(encoding="utf-8"))
                actor.config.setdefault("scale", [1., 1., 1.])
            actor.config.setdefault("contact_points_pose", [])
            actor.set_mass(spec["mass_kg"])
            self.objects.append(actor)
        if self.case.get("placement_marker"):
            builder = self.scene.create_actor_builder()
            builder.add_box_visual(pose=sapien.Pose([*self.case["placement_xy"], .7402]),
                                   half_size=[.075, .075, .0002], material=[.25, .65, .30])
            builder.build_static(name="fixed_placement_marker")

    def hold(self, seconds):
        # Task waypoint dwell, without a planner or PD change.
        n = round(seconds / self.scene.get_timestep())
        seq = {"left_gripper": None, "right_gripper": None}
        for side in ["left", "right"]:
            joints = getattr(self.robot, side + "_arm_joints")
            q = np.array([j.get_drive_target()[0] for j in joints])
            seq[side + "_arm"] = {"position": np.tile(q, (n, 1)),
                                   "velocity": np.zeros((n, len(q)))}
        self.take_dense_action(seq, save_freq=None)

    def require_plan(self):
        if not self.plan_success:
            raise RuntimeError("Native motion planning failed at " + self.stage)

    def pause_at_waypoint(self, point):
        """Count the task dwell from actual motion, with a finite settling limit."""
        dt = self.scene.get_timestep()
        previous = self.target_actor.get_pose().p.copy()
        stable = 0
        for _ in range(round(.6 / dt)):
            self.hold(dt)
            position = self.target_actor.get_pose().p.copy()
            speed = np.linalg.norm(position - previous) / dt
            stable = stable + 1 if np.linalg.norm(position - point) <= .031 and speed <= .03 else 0
            previous = position
            if stable >= round(.2 / dt):
                return
        raise RuntimeError("Waypoint dwell not achieved within 600ms")

    def move_object_to_waypoint(self, point):
        delta = np.asarray(point) - self.target_actor.get_pose().p
        self.move(self.move_by_displacement(self.arm, x=delta[0], y=delta[1], z=delta[2]))

    def transport_after_grasp(self):
        self.stage = "lift"
        self.move(self.move_by_displacement(self.arm, z=0.10))
        self.require_plan()
        actor = self.target_actor
        self.lifted = bool(actor.get_pose().p[2] > self.initial_poses[self.case["intent_index"]].p[2] + .04)
        if not self.lifted:
            raise RuntimeError("Target was not lifted")
        if self.case.get("transport_entry") is not None:
            self.stage = "transport_entry"
            self.move_object_to_waypoint(self.case["transport_entry"])
            self.require_plan()

    def play_once(self):
        actor = self.target_actor
        self.stage = "grasp"
        spec = self.case["objects"][self.case["intent_index"]]
        grasp = spec.get("grasp", {})
        ids = grasp.get("official_contact_ids")
        if grasp.get("contact_frame_asset_units") is not None:
            actor.config["contact_points_pose"].append(grasp["contact_frame_asset_units"])
            ids = [len(actor.config["contact_points_pose"]) - 1]
        # Initialization opens both grippers. Keep that aperture throughout
        # approach; close only after reaching the grasp endpoint.
        grasp_distance = grasp.get("axial_retreat_m_by_robot", {}).get(
            self.case["robot"], grasp.get("axial_retreat_m", .02))
        self.move(self.grasp_actor(actor, self.arm, pre_grasp_dis=grasp.get("pre_grasp_dis", .10),
                                  grasp_dis=grasp_distance, contact_point_id=ids))
        self.require_plan()
        self.transport_after_grasp()
        for index, point in enumerate(self.case["waypoints"]):
            self.stage = "waypoint_" + str(index + 1)
            # Translate native EE target by measured actor-origin displacement;
            # native planner supplies the whole path, no Cartesian interpolation.
            self.move_object_to_waypoint(point)
            self.require_plan()
            self.pause_at_waypoint(np.asarray(point))
            self.waypoint_results.append(dict(point=point, reached=actor.get_pose().p.tolist()))
        self.stage = "place"
        spec = self.case["objects"][self.case["intent_index"]]
        pose = [*self.case["placement_xy"], spec["rest_z"], 0, 0, 0, 1]
        # The task fixes the destination, not an extra yaw. Native free placement
        # restores the asset's upright axis without imposing a wrist rotation.
        self.move(self.place_actor(actor, self.arm, target_pose=pose, constrain="free",
                                   pre_dis=.07, dis=.002, pre_dis_axis=[0.0, 0.0, -1.0]))
        self.require_plan()
        self.stage = "settle"
        self.hold(.4)
        self.stage = "complete"
        return self.info

    def check_success(self):
        p = self.target_actor.get_pose().p
        spec = self.case["objects"][self.case["intent_index"]]
        return bool(self.lifted and np.linalg.norm(p[:2] - self.case["placement_xy"]) <= .03
                    and abs(p[2] - spec["rest_z"]) <= .012
                    and self.robot.is_left_gripper_open() and self.robot.is_right_gripper_open())

