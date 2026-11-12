"""RoboICL world-frame Franka hand commands and constrained 25 Hz Pink IK."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

CONTROL_RECIPE = dict(control_hz=25, integration_steps=10, integration_dt=.004,
    position_cost=1., orientation_cost=.1, lm_damping=.001, gain=1.,
    position_limit_gain=1., solver='quadprog', state_source='drive_command',
    collision_check=False, physics_output='each_integration_position_and_velocity',
    target_interpolation='linear_translation_shortest_SO3_and_linear_gripper')


def rigid_transform(value):
    pose = np.asarray(value, dtype=float)
    if (pose.shape != (4, 4) or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-7, rtol=0)
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-6)):
        raise ValueError('Expected a finite proper rigid transform')
    return pose


def encode10(pose, gripper):
    pose = rigid_transform(pose)
    if not np.isfinite(gripper) or not 0 <= gripper <= 1:
        raise ValueError('Native gripper must be in [0,1]')
    return np.r_[pose[:3, 3], pose[:3, 0], pose[:3, 1], 2 * gripper - 1]


def decode10(action):
    value = np.asarray(action, dtype=float)
    if value.shape != (10,) or not np.isfinite(value).all():
        raise ValueError('Expected finite 10D action')
    a, b = value[3:6], value[6:9]
    if np.linalg.norm(a) < 1e-8:
        raise ValueError('Degenerate first rotation column')
    x = a / np.linalg.norm(a)
    y = b - x * (x @ b)
    if np.linalg.norm(y) < 1e-8:
        raise ValueError('Degenerate second rotation column')
    y /= np.linalg.norm(y)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack((x, y, np.cross(x, y)))
    pose[:3, 3] = value[:3]
    return pose, float(np.clip((value[9] + 1) / 2, 0, 1))


class FrankaKinematics:
    def __init__(self, description):
        import pinocchio as pin
        self.pin = pin
        path = Path(description).resolve()
        self.description = json.loads(path.read_text(encoding='utf-8'))
        self.urdf = (path.parent / self.description['urdf']).resolve()
        if hashlib.sha256(self.urdf.read_bytes()).hexdigest() != self.description['urdf_sha256']:
            raise ValueError('URDF differs from frozen geometry')
        if self.description['frame'] != 'world' or self.description['state_source'] != 'drive_command':
            raise ValueError('Expected world drive-command contract')
        self.world_from_base = rigid_transform(self.description['world_from_base'])
        self.joint_names = self.description['arm_joint_names']
        self.tcp = self.description['tcp']
        original = pin.buildModelFromUrdf(str(self.urdf))
        locked = [i for i, name in enumerate(original.names) if i and name not in self.joint_names]
        self.model = pin.buildReducedModel(original, locked, pin.neutral(original))
        if list(self.model.names)[1:] != self.joint_names or not self.model.existFrame(self.tcp):
            raise ValueError('Geometry joint order or TCP mismatch')
        self.data = self.model.createData()
        self.frame_id = self.model.getFrameId(self.tcp)

    def pose(self, command):
        command = np.asarray(command, dtype=float)
        if command.shape != (8,) or not np.isfinite(command).all():
            raise ValueError('Expected seven joints plus native gripper')
        self.pin.forwardKinematics(self.model, self.data, command[:7])
        self.pin.updateFramePlacements(self.model, self.data)
        return self.world_from_base @ self.data.oMf[self.frame_id].homogeneous

    def state10(self, command):
        return encode10(self.pose(command), command[-1])

    def validate_robot(self, robot):
        actual = robot.left_entity.get_root_pose().to_transformation_matrix()
        if not np.allclose(actual, self.world_from_base, atol=1e-6, rtol=0):
            raise ValueError('Robot root differs from frozen geometry')
        if [j.name for j in robot.left_arm_joints] != self.joint_names:
            raise ValueError('Robot joint order differs from frozen geometry')
        command = np.r_[robot.left_entity.get_qpos()[:7], 1.]
        hand = robot.left_entity.find_link_by_name(self.tcp).pose.to_transformation_matrix()
        error = float(np.max(np.abs(self.pose(command) - hand)))
        if error > 1e-6:
            raise ValueError(f'FK differs from measured simulator link: {error}')
        return error


class PinkController:
    def __init__(self, kinematics, command, *, control_hz=25):
        import pink
        from pink.tasks import FrameTask
        from pink.limits import ConfigurationLimit, VelocityLimit
        self.pink, self.pin = pink, kinematics.pin
        if control_hz not in (25, 50):
            raise ValueError('Supported action clocks are25Hz and50Hz')
        self.control_hz = control_hz
        self.integration_steps = 250 // control_hz
        self.model = kinematics.model
        self.tcp = kinematics.tcp
        self.base_from_world = np.linalg.inv(kinematics.world_from_base)
        self.configuration = pink.Configuration(self.model, self.model.createData(), np.asarray(command[:7], dtype=float))
        self.task = FrameTask(self.tcp, **{k: CONTROL_RECIPE[k] for k in ('position_cost', 'orientation_cost', 'lm_damping', 'gain')})
        self.limits = [ConfigurationLimit(self.model, config_limit_gain=1.), VelocityLimit(self.model)]
        self.reset(command)

    def reset(self, command):
        q = np.asarray(command, dtype=float)
        if q.shape != (8,) or not np.isfinite(q).all():
            raise ValueError('Invalid command-space IK seed')
        arm = q[:7]
        clipped = np.clip(arm, self.model.lowerPositionLimit, self.model.upperPositionLimit)
        outside = (arm < self.model.lowerPositionLimit - 1e-7) | (arm > self.model.upperPositionLimit + 1e-7)
        if np.any(outside & (arm != clipped.astype(np.float32).astype(float))):
            raise ValueError('IK seed outside joint limits')
        self.configuration.update(np.where(outside, clipped, arm))
        self.gripper = float(q[-1])

    def command(self, action):
        pose, gripper = decode10(action)
        target = self.base_from_world @ pose
        start = self.configuration.get_transform_frame_to_world(self.tcp)
        start_position = start.translation.copy()
        start_rotation = start.rotation.copy()
        rotation_delta = self.pin.log3(start_rotation.T @ target[:3, :3])
        positions, velocities = [], []
        dt = CONTROL_RECIPE['integration_dt']
        count = self.integration_steps
        grippers = np.linspace(self.gripper, gripper, count + 1)[1:]
        for index in range(count):
            fraction = (index + 1) / count
            intermediate = self.pin.SE3(start_rotation @ self.pin.exp3(fraction * rotation_delta),
                start_position + fraction * (target[:3, 3] - start_position))
            self.task.set_target(intermediate)
            velocity = self.pink.solve_ik(self.configuration, [self.task], dt, solver='quadprog', limits=self.limits)
            self.configuration.integrate_inplace(velocity, dt)
            q = self.configuration.q
            if (not np.isfinite(q).all() or not np.isfinite(velocity).all()
                    or np.any(q < self.model.lowerPositionLimit - 1e-7)
                    or np.any(q > self.model.upperPositionLimit + 1e-7)
                    or np.any(np.abs(velocity) > self.model.velocityLimit + 1e-6)):
                raise RuntimeError('Invalid constrained Pink output')
            positions.append(q.copy())
            velocities.append(velocity.copy())
        actual = self.configuration.get_transform_frame_to_world(self.tcp)
        audit = dict(position_error_m=float(np.linalg.norm(actual.translation - target[:3, 3])),
            rotation_error_rad=float(np.linalg.norm(self.pin.log3(actual.rotation.T @ target[:3, :3]))))
        self.gripper = gripper
        return np.asarray(positions), np.asarray(velocities), grippers, audit
