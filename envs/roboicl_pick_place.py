"""Small RGB-Demo collection task using RoboTwin's native expert primitives."""
import numpy as np
import sapien

from ._base_task import Base_Task
from .utils import ArmTag, create_actor, create_entity_box


class roboicl_pick_place(Base_Task):
    LAYOUTS = ((0, 1, 2), (1, 2, 0), (2, 0, 1), (0, 2, 1), (1, 0, 2), (2, 1, 0))
    OBJECTS = (('073_rubikscube', 0), ('075_bread', 0), ('113_coffee-box', 0))
    POSITIONS = ((-.48, -.16), (-.30, -.16), (-.12, -.16))
    DESTINATION = (-.20, .08)
    INITIAL_XY_LIMIT = .01
    # Rounded assets such as the apple naturally settle by about 29 degrees.
    INITIAL_ROTATION_LIMIT_DEG = 35.

    def setup_demo(self, **kwargs):
        self.record_control = False
        self.physics_step = 0
        self.control_trace = []
        self._dense_context = None
        self._dense_segment = 0
        self.slot_positions = np.asarray(kwargs.get('slot_positions', self.POSITIONS), dtype=float)
        self.destination = np.asarray(kwargs.get('destination', self.DESTINATION), dtype=float)
        if (self.slot_positions.shape != (3, 2) or self.destination.shape != (2,)
                or not np.isfinite(self.slot_positions).all() or not np.isfinite(self.destination).all()):
            raise ValueError('Expected three finite XY slots and one destination')
        self.position_offsets = np.asarray(kwargs.get("position_offsets", [[0., 0.]] * 3), dtype=float)
        self.yaw_offsets = np.asarray(kwargs.get("yaw_offsets", [0.] * 3), dtype=float)
        if self.position_offsets.shape != (3, 2) or self.yaw_offsets.shape != (3,):
            raise ValueError("Expected three XY offsets and three yaw offsets")
        if not np.isfinite(self.position_offsets).all() or not np.isfinite(self.yaw_offsets).all():
            raise ValueError("Initial-state offsets must be finite")
        if np.abs(self.position_offsets).max() > .01 or np.abs(self.yaw_offsets).max() > np.deg2rad(5) + 1e-9:
            raise ValueError("Initial-state offsets exceed the frozen 1 cm / 5 degree range")
        self.layout = int(kwargs.get('layout', 0))
        self.target_index = int(kwargs.get('target_index', 0))
        self.expert_contact_id = kwargs.get('expert_contact_id')
        self.expert_place_constraint = kwargs.get('expert_place_constraint', 'auto')
        self.expert_grasp_distance = float(kwargs.get('expert_grasp_distance', 0.))
        self.common_clearance = kwargs.get('common_clearance')
        if not 0 <= self.expert_grasp_distance <= .03:
            raise ValueError('Expert grasp retraction must be between zero and 3 cm')
        self.object_specs = kwargs.get('object_specs', [dict(asset=a, model_id=m, contact_id=1) for a,m in self.OBJECTS])
        if len(self.object_specs) != 3 or len({(s['asset'],s['model_id']) for s in self.object_specs}) != 3:
            raise ValueError('Exactly three different object specifications are required')
        if self.layout not in range(6) or self.target_index not in range(3):
            raise ValueError('Expected one of six layouts and three target indices')
        self.target_lifted = False
        self.audit_frames = []
        self.stages = {}
        self.initialization_quality = None
        self._checking_initial_layout = False
        super()._init_task_env_(**kwargs)
        self.initial_positions = np.array([a.get_pose().p for a in self.objects])
        self.initial_robot_qpos = self.robot.left_entity.get_qpos().tolist()
        self.physics_step = 0
        self.record_control = bool(kwargs.get("record_control", False) and self.save_data)

    def load_actors(self):
        self.arm_tag = ArmTag('left')
        self.objects = []
        for identity, spec in enumerate(self.object_specs):
            name, model_id = spec['asset'], spec['model_id']
            x, y = self.slot_positions[self.LAYOUTS[self.layout][identity]]
            x, y = np.array([x, y]) + self.position_offsets[identity]
            half_yaw = self.yaw_offsets[identity] / 2
            rotation = sapien.Pose(q=[np.cos(half_yaw), 0, 0, np.sin(half_yaw)])
            pose = sapien.Pose([x, y, .741]) * rotation * sapien.Pose(q=[.5, .5, .5, .5])
            actor = create_actor(self, pose,
                                 modelname=name, model_id=model_id, convex=True)
            actor.set_mass(.05)
            self.objects.append(actor)
            self.add_prohibit_area(actor, padding=.05)
        self.object_bodies = [a.actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent) for a in self.objects]
        self.spawn_object_poses = [a.get_pose().p.tolist() + a.get_pose().q.tolist() for a in self.objects]
        self.target = self.objects[self.target_index]
        # Thin fixed marker uses the official actor helper, not a moving receptacle.
        self.marker = create_entity_box(self, sapien.Pose([*self.destination, .7405]),
            half_size=[.075, .075, .0005], color=[.25, .65, .30], is_static=True,
            name='fixed_destination')

    def check_stable(self):
        self._initial_contacts = set()
        self._checking_initial_layout = True
        try:
            stable, unstable = super().check_stable()
        finally:
            self._checking_initial_layout = False
        before = np.asarray(self.spawn_object_poses)
        after = np.asarray([a.get_pose().p.tolist() + a.get_pose().q.tolist() for a in self.objects])
        xy_error = np.linalg.norm(after[:, :2] - before[:, :2], axis=1)
        q0 = before[:, 3:] / np.linalg.norm(before[:, 3:], axis=1, keepdims=True)
        q1 = after[:, 3:] / np.linalg.norm(after[:, 3:], axis=1, keepdims=True)
        rotation = np.rad2deg(2 * np.arccos(np.clip(np.abs(np.sum(q0 * q1, axis=1)), 0, 1)))
        layout_ok = (xy_error <= self.INITIAL_XY_LIMIT) & (rotation <= self.INITIAL_ROTATION_LIMIT_DEG)
        self.initialization_quality = dict(
            version=1, passed=bool(stable and layout_ok.all() and not self._initial_contacts),
            native_stable=bool(stable), xy_limit_m=self.INITIAL_XY_LIMIT,
            rotation_limit_deg=self.INITIAL_ROTATION_LIMIT_DEG,
            spawn_object_poses=before.tolist(), settled_object_poses=after.tolist(),
            xy_error_m=xy_error.tolist(), rotation_error_deg=rotation.tolist(),
            non_table_contact_pairs=[list(pair) for pair in sorted(self._initial_contacts)],
            contact_impulse_threshold_ns=1e-6)
        invalid = [spec['asset'] for spec, ok in zip(self.object_specs, layout_ok) if not ok]
        invalid += ['initial_contact:' + '/'.join(pair) for pair in sorted(self._initial_contacts)]
        return self.initialization_quality['passed'], list(dict.fromkeys(unstable + invalid))

    def take_dense_action(self, control_seq, save_freq=-1):
        self._dense_segment += 1
        self._dense_context = dict(sequence=control_seq, step=0, segment=self._dense_segment)
        try:
            return super().take_dense_action(control_seq, save_freq=save_freq)
        finally:
            self._dense_context = None

    def _step_scene(self):
        if self.record_control:
            joints = self.robot.left_entity.get_active_joints()
            calls = {}
            context = self._dense_context
            for name in ('left_arm', 'left_gripper', 'right_arm', 'right_gripper'):
                sequence = context['sequence'][name] if context else None
                length = (len(sequence['position']) if name.endswith('arm') else sequence['num_step']) if sequence is not None else 0
                calls[name] = bool(context and context['step'] < length)
            self.control_trace.append(dict(
                physics_step=self.physics_step,
                mode='dense' if context else 'hold',
                setter_calls=calls,
                segment=context['segment'] if context else None,
                drive_position=[float(j.get_drive_target()[0]) for j in joints],
                drive_velocity=[float(j.get_drive_velocity_target()[0]) for j in joints],
                measured_qpos=self.robot.left_entity.get_qpos().tolist(),
                measured_qvel=self.robot.left_entity.get_qvel().tolist(),
                applied_qf=self.robot.left_entity.get_qf().tolist(),
                object_poses=[a.get_pose().p.tolist()+a.get_pose().q.tolist() for a in self.objects],
                object_linear_velocity=[a.linear_velocity.tolist() for a in self.object_bodies],
                object_angular_velocity=[a.angular_velocity.tolist() for a in self.object_bodies],
                gripper_command=float(self.robot.get_left_gripper_val())))
        super()._step_scene()
        if getattr(self, '_checking_initial_layout', False):
            names = {spec['asset'] for spec in self.object_specs}
            for contact in self.scene.get_contacts():
                pair = tuple(sorted(body.entity.name for body in contact.bodies))
                if names.intersection(pair) and 'table' not in pair:
                    if any(np.linalg.norm(point.impulse) > 1e-6 for point in contact.points):
                        self._initial_contacts.add(pair)
        if self._dense_context is not None:
            self._dense_context['step'] += 1
        self.physics_step += 1

    def _take_picture(self):
        super()._take_picture()
        if self.save_data and hasattr(self, 'initial_positions'):
            self.audit_frames.append(self.object_state())

    def object_state(self):
        return dict(physics_step=self.physics_step,
                    timestamp_s=self.physics_step * self.scene.get_timestep(),
                    positions=[a.get_pose().p.tolist() for a in self.objects],
                    object_poses=[a.get_pose().p.tolist()+a.get_pose().q.tolist() for a in self.objects],
                    object_linear_velocity=[a.linear_velocity.tolist() for a in self.object_bodies],
                    object_angular_velocity=[a.angular_velocity.tolist() for a in self.object_bodies],
                    frame_index=max(0, self.FRAME_IDX-1),
                    plan_success=bool(self.plan_success),
                    hand_pose=self.robot.left_entity.find_link_by_name('panda_hand').pose.to_transformation_matrix().tolist(),
                    left_tcp=self.get_arm_pose('left'),
                    left_qpos=self.robot.left_entity.get_qpos().tolist(),
                    left_qvel=self.robot.left_entity.get_qvel().tolist(),
                    left_drive_position=[float(j.get_drive_target()[0]) for j in self.robot.left_entity.get_active_joints()],
                    left_drive_velocity=[float(j.get_drive_velocity_target()[0]) for j in self.robot.left_entity.get_active_joints()],
                    left_qf=self.robot.left_entity.get_qf().tolist(),
                    gripper_command=float(self.robot.get_left_gripper_val()),
                    right_qpos=self.robot.right_entity.get_qpos().tolist())

    def play_once(self):
        self._take_picture()
        # Clear the row before moving laterally above a different target.
        if self.common_clearance is None:
            self.move(self.move_by_displacement(arm_tag=self.arm_tag, z=.20))
        else:
            # A frozen native path establishes an identical decision state for
            # every target in an initial-state group. It is separate from the
            # task-specific planner path list in both planning and replay.
            path = self.common_clearance
            position = np.asarray(path['position'], dtype=float)
            velocity = np.asarray(path['velocity'], dtype=float)
            if position.ndim != 2 or position.shape[1] != 7 or velocity.shape != position.shape:
                raise ValueError('Common clearance requires matched seven-joint controls')
            if not np.isfinite(position).all() or not np.isfinite(velocity).all():
                raise ValueError('Non-finite clearance controls')
            if not np.allclose(self.robot.left_entity.get_qpos(), path['initial_qpos'], atol=1e-5, rtol=0):
                raise ValueError('Common clearance initial robot state mismatch')
            if abs(self.scene.get_timestep() - path['timestep_s']) > 1e-9:
                raise ValueError('Common clearance physics clock mismatch')
            limits = self.robot.left_entity.get_qlimits()[:7]
            if np.any(position < limits[:, 0]) or np.any(position > limits[:, 1]):
                raise ValueError('Common clearance violates joint limits')
            self.take_dense_action(dict(left_arm=dict(position=position, velocity=velocity),
                                        left_gripper=None, right_arm=None, right_gripper=None))
        self.stages['clearance'] = self.object_state()
        grasp = self.grasp_actor(self.target, arm_tag=self.arm_tag,
                                 contact_point_id=self.expert_contact_id,
                                 grasp_dis=self.expert_grasp_distance)
        self.move(grasp)
        self.stages['grasp'] = self.object_state()
        self.move(self.move_by_displacement(arm_tag=self.arm_tag, z=.12))
        self.stages['lift'] = self.object_state()
        self.target_lifted = bool(self.target.get_pose().p[2] - self.initial_positions[self.target_index, 2] > .06)
        self.move(self.place_actor(self.target, arm_tag=self.arm_tag,
                                  target_pose=[*self.destination, .742],
                                  constrain=self.expert_place_constraint))
        self.stages['release'] = self.object_state()
        self.move(self.move_by_displacement(arm_tag=self.arm_tag, z=.10))
        # Observe released stability; no extra planner/controller is introduced.
        for step in range(250):
            self._step_scene()
            if step % 10 == 0:
                self._take_picture()
        self.info['info'] = dict(layout=self.layout, target_index=self.target_index,
                                target_asset=self.object_specs[self.target_index])
        return self.info

    def check_success(self):
        if not hasattr(self, 'initial_positions'):
            return False
        positions = np.array([a.get_pose().p for a in self.objects])
        non_targets = [i for i in range(3) if i != self.target_index]
        return bool(self.plan_success and self.target_lifted
            and np.linalg.norm(positions[self.target_index, :2] - self.destination) < .065
            and abs(positions[self.target_index, 2] - self.initial_positions[self.target_index, 2]) < .035
            and np.max(np.linalg.norm(positions[non_targets] - self.initial_positions[non_targets], axis=1)) < .03
            and self.robot.is_left_gripper_open())
