"""Opt-in V2 expert for object-position waypoints; native tasks stay on V1."""
from copy import copy
from pathlib import Path
import numpy as np
import torch
import hppfcl as fcl
from curobo.types import ToolPoseCriteria

from .native_v2_task import NativeV2Task
from .roboicl_choice_route import roboicl_choice_route
from .robot.native_v2_planner import NativeV2Planner, matrix, pose_list, shapes
from .robot.roboicl_mesh_sweep import convex, transform
from .utils import Action


class ChoiceRouteV2Planner(NativeV2Planner):
    # Match the mature fixed-world V2 planner's native feasibility settings.
    cspace_overrides = {"acceleration_scale": .9}
    planner_options = {"optimizer_collision_activation_distance": .03}
    # A resting mouse has measured 25um support penetration; 10um left the
    # native graph start invalid. Keep this tolerance local to Choice and
    # retain the actual mesh sweep, including initial-support handling.
    payload_sphere_numerical_inset_m = 1e-4

    def __init__(self, task, side, **kwargs):
        self.cspace_overrides = dict(self.cspace_overrides)
        for key in ('max_acceleration', 'max_jerk'):
            if key in task.case.get('planner_limits', {}):
                self.cspace_overrides[key] = float(task.case['planner_limits'][key])
        super().__init__(task, side, **kwargs)
        self.log('planner_limits', **self.cspace_overrides)

    def grasp(self, actor, poses, approach_distance):
        # Parallel jaws admit the same contact with the fingers exchanged.
        # Keep both native tool orientations; the camera/palm and full physical
        # geometry still have to clear the scene for the selected trajectory.
        exchanged = []
        for pose in poses:
            frame = matrix(pose)
            frame[:3, :3] = frame[:3, :3] @ np.diag([1., -1., -1.])
            exchanged.append(pose_list(frame))
        grasp = self.task.case['objects'][self.task.case['intent_index']]['grasp']
        canonical, alternate = (exchanged, poses) if grasp.get('exchange_fingers', False) else (poses, exchanged)
        branches = [0] * len(canonical) + [1] * len(alternate)
        poses = canonical + alternate
        max_tilt = self.task.case.get("grasp_max_tilt_deg")
        if max_tilt is not None:
            tilt = [float(np.degrees(np.arccos(np.clip(-matrix(p)[2, 0], -1., 1.))))
                    for p in poses]
            branches = [b for b, angle in zip(branches, tilt) if angle <= max_tilt]
            poses = [p for p, angle in zip(poses, tilt) if angle <= max_tilt]
            self.log('grasp_orientation_candidates', max_tilt_deg=max_tilt,
                     candidate_tilts_deg=tilt, retained=len(poses))
            if not poses:
                raise RuntimeError('No native grasp candidate matches the grasp orientation definition')
        # Match Franka: try the configured object-relative jaw orientation first.
        # Within each branch, prefer the smallest rotation from the current pose.
        current = matrix(getattr(self.task.robot, 'get_'+self.side+'_ee_pose')())[:3, :3]
        angles = [float(np.degrees(np.arccos(np.clip(
            (np.trace(current.T @ matrix(p)[:3, :3])-1.)/2., -1., 1.)))) for p in poses]
        order = sorted(range(len(poses)), key=lambda i: (branches[i], angles[i]))
        self.log('grasp_rotation_priority', candidate_rotation_deg=angles,
                 candidate_order=order, candidate_branches=branches,
                 reference='configured grasp orientation, then rotation from start')
        failures = []
        for index in order:
            self.queue.clear()
            try:
                # Retain native grasp-contact handling. The complete open-jaw
                # path must still clear the actual scene and self geometry.
                result = super().grasp(actor, [poses[index]], approach_distance)
                limits = self.task.case.get('demo_quality_guard')
                if limits:
                    from .robot.roboicl_demo_quality import grasp_rotation_quality
                    positions = np.concatenate([self.entity.get_qpos()[self.indices][None],
                                                *[p['position'] for p in self.queue]])
                    check = grasp_rotation_quality(self.world_tool_poses(positions), positions,
                                                   limits['rotation_travel_deg'])
                    self.log('demo_quality_grasp_plan', **check)
                    if not check['passed']:
                        raise RuntimeError('Demo quality rejected: '+','.join(check['reasons']))
            except RuntimeError as error:
                reason = str(error)
                if not reason.startswith(('Native V2 plan_grasp failed:',
                                          'No native grasp candidate clears the physical open gripper',
                                          'Native V2 arm intersects physical self geometry',
                                          'Native V2 open gripper intersects physical scene geometry',
                                          'Demo quality rejected:')):
                    raise
                failures.append(reason)
                self.log('native_grasp_search', attempt=index, accepted=False, reason=reason,
                         rotation_from_start_deg=angles[index])
                continue
            self.log('native_grasp_search', attempt=index, accepted=True,
                     canonical_branch=branches[index] == 0,
                     rotation_from_start_deg=angles[index])
            return result
        raise RuntimeError('Native V2 plan_grasp failed: all existing candidates exhausted; '
                           + '; '.join(dict.fromkeys(failures)))

    def set_tool_transform(self, local):
        # Update the existing native tool frame without rebuilding the solver.
        for kin in [self.planner.ik_solver.kinematics, self.planner.trajopt_solver.kinematics]:
            params = copy(kin.config.kinematics_config)
            params.fixed_transforms = params.fixed_transforms.clone()
            index = params.link_name_to_idx_map[self.tool_frame]
            params.fixed_transforms[index].copy_(torch.as_tensor(
                local[:3], device='cuda', dtype=params.fixed_transforms.dtype))
            kin.update_kinematics_config(params)

    def plan_object_waypoint(self, point):
        last_error = None
        for attempt in range(3):
            try:
                result = self._plan_object_waypoint_once(point)
            except RuntimeError as error:
                if not str(error).startswith(('Demo quality rejected:',
                        'Native V2 arm intersects physical self geometry',
                        'Carried object sweep intersects scene geometry')):
                    raise
                last_error = error
                self.log('object_waypoint_candidate', attempt=attempt + 1,
                         accepted=False, reason=str(error), position=list(point))
                continue
            if result.get('status') == 'Fail':
                last_error = None
                self.log('object_waypoint_candidate', attempt=attempt + 1,
                         accepted=False, reason='native planning failed', position=list(point))
                continue
            self.log('object_waypoint_candidate', attempt=attempt + 1,
                     accepted=True, position=list(point))
            return result
        if last_error is not None:
            raise last_error
        return dict(status='Fail')

    def _plan_object_waypoint_once(self, point):
        if self.held is None:
            raise RuntimeError("An object waypoint requires a held object")
        self.sync()
        parent = matrix(self.entity.find_link_by_name(self.tool_link).entity_pose)
        ee = matrix(getattr(self.task.robot, 'get_'+self.side+'_ee_pose')())
        actor = matrix(self.held.get_pose())
        local_ee = np.linalg.inv(parent) @ ee
        local_object = np.linalg.inv(parent) @ actor
        self.set_tool_transform(local_object)
        try:
            predicted = self.world_tool_poses(self.state().position)[0]
            error = float(np.linalg.norm(predicted[:3, 3]-actor[:3, 3]))
            if error > 1e-4:
                raise RuntimeError('Object tool frame differs from measured held origin')
            target = actor.copy(); target[:3, 3] = point
            criteria = ToolPoseCriteria(
                terminal_pose_axes_weight_factor=[1, 1, 1, 0, 0, 0],
                # Native goal region, safely inside the task's 31mm region.
                # Grasp/place poses and physical acceptance stay unchanged.
                terminal_pose_convergence_tolerance=[.01, 0.],
                non_terminal_pose_axes_weight_factor=[0, 0, 0, .001, .001, .001],
                project_distance_to_goal=False)
            self.planner.update_tool_pose_criteria({self.tool_frame: criteria})
            self.log('object_waypoint_goal', position=list(point),
                     tool_from_object=local_object.tolist(), object_frame_error_m=error,
                     terminal_orientation_constrained=False,
                     terminal_position_tolerance_m=.01,
                     non_terminal_orientation_weight=.001)
            result = self.planner.plan_pose(self.goal([pose_list(target)]), self.state(), max_attempts=10)
            if result is None or not bool(result.success.all()):
                self.log('object_waypoint_failure', position=list(point))
                return dict(status='Fail')
            trajectory = self.trajectory(result.get_interpolated_plan())
            limits = self.task.case.get('demo_quality_guard')
            if limits:
                from .robot.roboicl_demo_quality import path_quality
                # The native tool represents the measured object origin here.
                check = path_quality(self.world_tool_poses(trajectory['position']),
                                     self.task.cameras.get_config()['head_camera'], limits)
                self.log('demo_quality_plan', **check)
                if not check['passed']:
                    raise RuntimeError('Demo quality rejected: '+','.join(check['reasons']))
        finally:
            self.set_tool_transform(local_ee)
            self.planner.update_tool_pose_criteria({self.tool_frame: ToolPoseCriteria()})
        self.check_carried_sweep(trajectory['position'])
        return trajectory

    def check_carried_sweep(self, positions):
        if self.held is None:
            return
        ee = matrix(getattr(self.task.robot, 'get_'+self.side+'_ee_pose')())
        relative = np.linalg.inv(ee) @ matrix(self.held.get_pose())
        parts = [convex(self.local_mesh(self.held.actor, i, shape)[0])
                 for i, shape in enumerate(shapes(self.held.actor))]
        obstacles = []
        for entity in self.world_entities():
            if entity.per_scene_id == self.held.actor.per_scene_id:
                continue
            for i, shape in enumerate(shapes(entity)):
                if 'Plane' not in type(shape).__name__:
                    obstacles.append((entity.name, convex(self.local_mesh(entity, i, shape)[0]), transform(matrix(entity.pose))))
        request = fcl.DistanceRequest(); request.enable_signed_distance = True
        initial_object = transform(matrix(self.held.get_pose()))
        initial_distances = [min(float(fcl.distance(part, initial_object, obstacle, obstacle_pose, request, fcl.DistanceResult()))
                                 for part in parts) for _, obstacle, obstacle_pose in obstacles]
        support = [name == 'table' and distance < 0 for (name, _, _), distance in zip(obstacles, initial_distances)]
        minimum = float('inf'); witness = None; violation = None
        for index, pose in enumerate(self.world_tool_poses(positions)):
            obj = transform(pose @ relative)
            for part in parts:
                for obstacle_index, (name, obstacle, obstacle_pose) in enumerate(obstacles):
                    distance = float(fcl.distance(part, obj, obstacle, obstacle_pose, request, fcl.DistanceResult()))
                    if distance < minimum:
                        minimum = distance; witness = dict(sample=index, object=name)
                    floor = min(-1e-5, initial_distances[obstacle_index]-1e-5) if support[obstacle_index] else -1e-5
                    if distance < floor and violation is None:
                        violation = dict(sample=index, object=name, distance_m=distance, allowed_m=floor)
            # Initial support penetration may only persist until the payload
            # clears that table shape; a later return is a new intersection.
            for obstacle_index, (_, obstacle, obstacle_pose) in enumerate(obstacles):
                if support[obstacle_index] and min(float(fcl.distance(part, obj, obstacle, obstacle_pose, request, fcl.DistanceResult())) for part in parts) >= 0:
                    support[obstacle_index] = False
        self.log('carried_mesh_sweep', minimum_m=minimum, witness=witness, violation=violation, samples=len(positions))
        if violation is not None:
            raise RuntimeError('Carried object sweep intersects scene geometry')

    def plan_pose(self, *args, **kwargs):
        result = super().plan_pose(*args, **kwargs)
        if result['status'] == 'Success':
            self.check_carried_sweep(result['position'])
        return result

    def trajectory(self, *args, **kwargs):
        result = super().trajectory(*args, **kwargs)
        directory = getattr(self.task, 'v2_plan_directory', None)
        if directory is not None:
            directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
            index = getattr(self, '_saved_plan_index', 0)
            self._saved_plan_index = index + 1
            name = f'{index:03d}_{self.side}.npz'
            np.savez_compressed(directory/name, **{k:result[k] for k in ['position', 'velocity', 'acceleration']})
            self.log('saved_plan', path=name, joint_names=self.names)
        return result


class roboicl_choice_route_v2(NativeV2Task, roboicl_choice_route):
    def place_actor(self, actor, arm_tag, target_pose, dis=.002, **kwargs):
        # This task needs one release endpoint, not an intermediate placement
        # stop. V2 supplies the complete trajectory and endpoint deceleration.
        if not self.plan_success:
            return None, []
        # Native free placement aligns a functional frame's Z axis. Derive
        # that frame from this asset's stable spawn orientation, rather than
        # assuming every asset stands on its local Y axis (a lying hammer
        # does not). Its origin remains the recorded object/task origin.
        spec = self.case['objects'][self.objects.index(actor)]
        support_frame = np.eye(4)
        support_frame[:3,:3] = matrix([0.,0.,0.,*spec['quaternion']])[:3,:3].T
        frames = actor.config.setdefault('functional_matrix', [])
        frames.append(support_frame.tolist())
        pose = self.get_place_pose(actor, arm_tag, target_pose,
            constrain=kwargs.get('constrain', 'free'), pre_dis=dis,
            pre_dis_axis=[0., 0., -1.], functional_point_id=len(frames)-1)
        self.v2_planner(str(arm_tag)).log('direct_release_goal', target=pose,
            release_gap_m=dis, explicit_preplacement=False,
            asset_support_frame=support_frame.tolist())
        return arm_tag, [Action(arm_tag, 'move', target_pose=pose),
                         Action(arm_tag, 'open', target_gripper_pos=1.)]

    def setup_demo(self, **kwargs):
        # Dataset optics include a fixed, embodiment-specific optical offset.
        # The D435 default clips near-table wrist observations at 10 cm.
        case = kwargs['case_manifest']
        if case.get('robot') == 'aloha-agilex':
            kwargs['camera'] = dict(kwargs['camera'], wrist_camera_type=case.get(
                'wrist_camera_type', 'ROBOKATA_ALOHA_WRIST_224_V1'))
        from copy import deepcopy
        collision_urdf = kwargs['case_manifest'].get('robot_collision_urdf')
        if collision_urdf is not None:
            # Task-frozen collision geometry retains the original articulation.
            import hashlib
            path = Path(collision_urdf['path']).resolve()
            if hashlib.sha256(path.read_bytes()).hexdigest() != collision_urdf['sha256']:
                raise ValueError('Robot collision URDF does not match its frozen SHA256')
            for side in ('left', 'right'):
                cfg = deepcopy(kwargs[side + '_embodiment_config'])
                cfg['urdf_path'] = str(path)
                kwargs[side + '_embodiment_config'] = cfg
        home = kwargs['case_manifest'].get('left_homestate')
        if home is not None:
            cfg = deepcopy(kwargs['left_embodiment_config'])
            cfg['homestate'][0] = list(home)
            kwargs['left_embodiment_config'] = cfg
        camera = kwargs['case_manifest'].get('external_camera')
        if camera is not None:
            cfg = deepcopy(kwargs['left_embodiment_config'])
            cfg['static_camera_list'] = [dict(c, **camera) if c['name']=='head_camera' else c
                                         for c in cfg['static_camera_list']]
            kwargs['left_embodiment_config'] = cfg
            kwargs['camera'] = dict(kwargs['camera'], head_camera_type=camera['type'])
        super().setup_demo(**kwargs)
        if str(self.arm) != 'left':
            raise ValueError("This collection version executes with the left arm only")

    def transport_after_grasp(self):
        # Match Franka's departure: raise the object bottom above the distractor.
        self.stage = 'transport_entry'
        planner = self.v2_planner(str(self.arm))
        bounds = []
        for actor in self.objects:
            pose = matrix(actor.get_pose())
            local = np.concatenate([np.asarray(planner.local_mesh(actor.actor, i, s)[0])
                                    for i, s in enumerate(shapes(actor.actor))])
            world = local @ pose[:3, :3].T + pose[:3, 3]
            bounds.append((float(world[:, 2].min()), float(world[:, 2].max())))
        target = self.case['intent_index']
        highest = max(top for i, (_, top) in enumerate(bounds) if i != target)
        rise = max(0., highest + .025 - bounds[target][0])
        start = self.target_actor.get_pose().p.copy()
        if rise > .001:
            action = self.move_by_displacement(self.arm, z=rise)
            action[1][0].args['v2_linear_world_z'] = True
            self.move(action)
        self.require_plan()
        actual = self.target_actor.get_pose().p
        self.lifted = bool(actual[2] > self.initial_poses[self.case['intent_index']].p[2] + .04)
        planner.log('vertical_departure', start=start.tolist(), actual=actual.tolist(),
                    rise_m=rise, non_target_top_m=highest, requested_clearance_m=.025,
                    lateral_drift_m=float(np.linalg.norm(actual[:2]-start[:2])))
        if not self.lifted:
            raise RuntimeError('Target was not lifted')

    def pause_at_waypoint(self, point):
        # Production3000 uses a 20ms backward difference, sampled every 4ms.
        # This changes neither physics timing nor the task's 200ms dwell.
        from collections import deque
        dt = self.scene.get_timestep()
        width = round(.020 / dt)
        positions = deque([self.target_actor.get_pose().p.copy()], maxlen=width+1)
        stable = 0
        for _ in range(round(1.0 / dt)):
            self.hold(dt)
            current = self.target_actor.get_pose().p.copy()
            positions.append(current)
            if len(positions) < width+1:
                continue
            speed = np.linalg.norm(current-positions[0]) / (width*dt)
            stable = stable+1 if np.linalg.norm(current-point) <= .031 and speed <= .05 else 0
            if stable >= 51:
                return
        raise RuntimeError("Waypoint dwell not achieved within 1000ms")

    def v2_planner(self, side):
        side = str(side)
        if side not in self.v2_planners:
            self.v2_planners[side] = ChoiceRouteV2Planner(
                self, side, collision_spheres_path=self.case.get('collision_spheres_profile'),
                exact_self_links=self.case.get('exact_self_links', ()))
        return self.v2_planners[side]

    def move_object_to_waypoint(self, point):
        side = str(self.arm)
        result = self.v2_planner(side).plan_object_waypoint(point)
        if result['status'] != 'Success':
            self.plan_success = False
            return
        controls = dict(left_arm=None, right_arm=None, left_gripper=None, right_gripper=None)
        controls[side+'_arm'] = result
        self.take_dense_action(controls, save_freq=None)
