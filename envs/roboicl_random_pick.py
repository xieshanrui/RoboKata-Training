"""Independent continuous-layout pure-pick scene; no placement or slot contract."""
from collections import deque
import numpy as np
import sapien

from ._base_task import Base_Task
from .roboicl_pick_place import roboicl_pick_place
from .utils import ArmTag, create_actor

SPAWN_ADMISSION = dict(version=2, center_x_bounds_m=[-.55,-.05], center_y_bounds_m=[-.30,.10],
    center_bound_tolerance_m=1e-6,
    table_x_bounds_m=[-.6,.6], table_y_bounds_m=[-.35,.35], table_height_m=.74,
    table_geometry_tolerance_m=.001, stable_tail_steps=125,
    stable_tail_position_tolerance_m=.001, stable_tail_rotation_tolerance_deg=1.,
    contact_impulse_threshold_ns=1e-6, spawn_rotation_change='diagnostic_only')


class roboicl_random_pick(roboicl_pick_place):
    def setup_demo(self, **kwargs):
        self.record_control = False
        self.physics_step = 0
        self.control_trace = []
        self._dense_context = None
        self._dense_segment = 0
        self.object_specs = kwargs['object_specs']
        count = len(self.object_specs)
        self.object_xy = np.asarray(kwargs['object_xy'], dtype=float)
        self.object_yaw = np.asarray(kwargs['object_yaw'], dtype=float)
        if count < 1 or self.object_xy.shape != (count, 2) or self.object_yaw.shape != (count,):
            raise ValueError('Expected one continuous XY and yaw per object')
        if not np.isfinite(self.object_xy).all() or not np.isfinite(self.object_yaw).all():
            raise ValueError('Non-finite layout')
        self.target_index = int(kwargs.get('target_index', 0))
        if not 0 <= self.target_index < count:
            raise ValueError('Invalid generation target')
        self.target_lifted = False
        self.audit_frames, self.stages = [], {}
        self.initialization_quality = None
        self._checking_initial_layout = False
        self.contact_events = []
        Base_Task._init_task_env_(self, **kwargs)
        self.initial_positions = np.asarray([a.get_pose().p for a in self.objects])
        self.initial_robot_qpos = self.robot.left_entity.get_qpos().tolist()
        self.object_collision_vertices = [self._collision_vertices(body) for body in self.object_bodies]
        self.finger_links = [self.robot.left_entity.find_link_by_name(name)
            for name in ('panda_leftfinger', 'panda_rightfinger')]
        self.finger_collision_vertices = [self._collision_vertices(link) for link in self.finger_links]
        # In hand coordinates, sweep the physical collision geometry across the
        # full legal finger travel. Vertex height is affine in finger opening,
        # so the two endpoints bound every intermediate opening.
        hand = self.robot.left_entity.find_link_by_name('panda_hand')
        world_to_hand = np.linalg.inv(hand.entity_pose.to_transformation_matrix())
        envelope = [self._collision_vertices(hand)]
        joints = self.robot.left_entity.get_active_joints()
        qpos = dict(zip((joint.name for joint in joints), self.robot.left_entity.get_qpos()))
        for index, (link, vertices) in enumerate(zip(self.finger_links, self.finger_collision_vertices)):
            transform = world_to_hand @ link.entity_pose.to_transformation_matrix()
            local = vertices @ transform[:3,:3].T + transform[:3,3]
            axis = np.array([0., 1. if index == 0 else -1., 0.])
            for opening in (0., .04):
                envelope.append(local + (opening-qpos[f'panda_finger_joint{index+1}'])*axis)
        self.gripper_collision_envelope = np.concatenate(envelope)
        self.physics_step = 0
        self.contact_events = []
    @staticmethod
    def _collision_vertices(body):
        vertices = []
        for shape in body.collision_shapes:
            if hasattr(shape, 'vertices'):
                points = np.asarray(shape.vertices)*np.asarray(shape.scale)
            elif hasattr(shape, 'half_size'):
                points = np.asarray([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])*shape.half_size
            else:
                raise RuntimeError('Unsupported collision geometry for minimum-height audit')
            pose = shape.local_pose.to_transformation_matrix()
            vertices.append(points @ pose[:3,:3].T+pose[:3,3])
        return np.concatenate(vertices)

    def gripper_bottom_at_hand_pose(self, hand_pose):
        points = self.gripper_collision_envelope
        return float(np.min(points @ hand_pose[2,:3] + hand_pose[2,3]))

    def object_state(self):
        state = super().object_state()
        if hasattr(self, 'object_collision_vertices'):
            state['object_bottom_z'] = [float((vertices @ actor.get_pose().to_transformation_matrix()[:3,:3].T
                +actor.get_pose().p)[:,2].min()) for actor,vertices in zip(self.objects,self.object_collision_vertices)]
            state['finger_bottom_z'] = [float((vertices @ link.entity_pose.to_transformation_matrix()[:3,:3].T
                +link.entity_pose.p)[:,2].min()) for link,vertices in zip(self.finger_links,self.finger_collision_vertices)]
        return state

    def load_actors(self):
        self.arm_tag = ArmTag('left')
        self.objects = []
        for spec, xy, yaw in zip(self.object_specs, self.object_xy, self.object_yaw):
            initial_quaternion = np.asarray(spec.get('initial_quaternion', [.5, .5, .5, .5]), dtype=float)
            if (initial_quaternion.shape != (4,) or not np.isfinite(initial_quaternion).all()
                    or abs(np.linalg.norm(initial_quaternion)-1.) > 1e-5):
                raise ValueError('Object initial_quaternion must be a finite unit wxyz quaternion')
            pose = (sapien.Pose([*xy, .741])
                    * sapien.Pose(q=[np.cos(yaw/2), 0, 0, np.sin(yaw/2)])
                    * sapien.Pose(q=initial_quaternion))
            actor = create_actor(self, pose, modelname=spec['asset'], model_id=spec['model_id'], convex=True)
            actor.set_mass(.05)
            self.objects.append(actor)
            self.add_prohibit_area(actor, padding=.05)
        self.object_bodies = [a.actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent) for a in self.objects]
        self.spawn_object_poses = [a.get_pose().p.tolist()+a.get_pose().q.tolist() for a in self.objects]
        self.target = self.objects[self.target_index]

    def check_stable(self):
        self._initial_contacts = set()
        self._settle_tail = deque(maxlen=SPAWN_ADMISSION['stable_tail_steps']+1)
        self._checking_initial_layout = True
        try:
            native_stable, unstable = Base_Task.check_stable(self)
        finally:
            self._checking_initial_layout = False
        poses = np.asarray(self._settle_tail)
        last = poses[-1]
        translation = np.linalg.norm(poses[:,:,:3]-last[None,:,:3],axis=2).max(axis=0)
        q = poses[:,:,3:]/np.linalg.norm(poses[:,:,3:],axis=2,keepdims=True)
        rotation = np.degrees(2*np.arccos(np.clip(np.abs(np.sum(q*q[-1][None],axis=2)),0,1))).max(axis=0)
        bounds = []
        for actor, body in zip(self.objects,self.object_bodies):
            vertices = self._collision_vertices(body)
            transform = actor.get_pose().to_transformation_matrix()
            world = vertices @ transform[:3,:3].T+transform[:3,3]
            bounds.append([world.min(axis=0).tolist(),world.max(axis=0).tolist()])
        bounds = np.asarray(bounds)
        eps = SPAWN_ADMISSION['table_geometry_tolerance_m']
        center_eps = SPAWN_ADMISSION['center_bound_tolerance_m']
        center_ok = ((last[:,0] >= -.55-center_eps) & (last[:,0] <= -.05+center_eps)
            & (last[:,1] >= -.30-center_eps) & (last[:,1] <= .10+center_eps))
        tabletop_ok = ((bounds[:,0,0] >= -.6-eps) & (bounds[:,1,0] <= .6+eps)
            & (bounds[:,0,1] >= -.35-eps) & (bounds[:,1,1] <= .35+eps)
            & (bounds[:,0,2] >= .74-eps) & (bounds[:,0,2] <= .74+eps))
        tail_ok = ((translation <= SPAWN_ADMISSION['stable_tail_position_tolerance_m'])
            & (rotation <= SPAWN_ADMISSION['stable_tail_rotation_tolerance_deg']))
        spawn = np.asarray(self.spawn_object_poses)
        q0 = spawn[:,3:]/np.linalg.norm(spawn[:,3:],axis=1,keepdims=True)
        spawn_rotation = np.degrees(2*np.arccos(np.clip(np.abs(np.sum(q0*q[-1],axis=1)),0,1)))
        valid = center_ok & tabletop_ok & tail_ok
        self.initialization_quality = dict(version=2, admission=SPAWN_ADMISSION,
            passed=bool(native_stable and valid.all() and not self._initial_contacts),
            native_stable=bool(native_stable), settled_centers_in_workspace=center_ok.tolist(),
            collision_geometry_on_table=tabletop_ok.tolist(), settled_collision_bounds_m=bounds.tolist(),
            stable_tail_translation_m=translation.tolist(), stable_tail_rotation_deg=rotation.tolist(),
            spawn_object_poses=spawn.tolist(), settled_object_poses=last.tolist(),
            spawn_to_settle_xy_m=np.linalg.norm(last[:,:2]-spawn[:,:2],axis=1).tolist(),
            spawn_to_settle_rotation_deg_diagnostic=spawn_rotation.tolist(),
            non_table_contact_pairs=[list(pair) for pair in sorted(self._initial_contacts)])
        invalid = [spec['asset'] for spec,ok in zip(self.object_specs,valid) if not ok]
        invalid += ['initial_contact:'+'/'.join(pair) for pair in sorted(self._initial_contacts)]
        return self.initialization_quality['passed'], list(dict.fromkeys(unstable+invalid))

    def _step_scene(self):
        super()._step_scene()
        if self._checking_initial_layout:
            self._settle_tail.append([a.get_pose().p.tolist()+a.get_pose().q.tolist() for a in self.objects])
        if hasattr(self, 'initial_positions'):
            for contact in self.scene.get_contacts():
                names = sorted(body.entity.name for body in contact.bodies)
                impulse = sum(float(np.linalg.norm(point.impulse)) for point in contact.points)
                if impulse > 1e-6:
                    self.contact_events.append(dict(physics_step=self.physics_step, bodies=names, impulse_ns=impulse))

    def check_success(self):
        if not hasattr(self, 'initial_positions'):
            return False
        lift = np.asarray([a.get_pose().p[2] for a in self.objects])-self.initial_positions[:, 2]
        return bool(np.any(lift >= .10) and self.robot.get_left_gripper_val() < .2)

    def play_once(self):
        raise RuntimeError('Use the independent EE pure-pick collector')
