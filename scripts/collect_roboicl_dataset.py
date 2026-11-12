"""Collect configured grasps with one object-independent RoboICL control profile.

Run from the RoboTwin directory with --batch and --geometry. Batch entries
contain id, config, and output paths. Geometry, initial states and requested
routes vary by record; physics and control do not. Every attempt is retained. This profile includes RoboICL controller overrides
and is not a claim of unmodified official RoboTwin compatibility.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import transforms3d as t3d

from collect_roboicl_open_routes import RecordingOpenRoutesTask
from collect_roboicl_curobo_select import RecordingSelectTask, run_episode
from collect_roboicl_curobo_pick import read, write, sha
from score_roboicl_open_routes import score_open_route
from envs.roboicl_select_scene import roboicl_select_scene
from envs.robot.roboicl_ee import FrankaKinematics
from envs._base_task import Base_Task


REPOSITORY = Path(__file__).resolve().parents[1]
CONTROL_PROFILE = dict(
    version='roboicl_finger_friction_v2', physics_dt_s=.004, rgb_hz=50,
    gripper_stiffness=1000., gripper_damping=200.,
    gripper_force_limit='native_uncapped', close_steps=[200, 100],
    close_profile='official linear command plus hold',
    release_steps=[200, 100], release_profile='official linear command plus hold',
    pregrasp_standoff_m=.1, release_gap_m=.002, preplace_height_m=.05,
    motion_clock='native_curobo_v2', additional_endpoint_holds=False,
    contact_geometry_and_materials='native_geometry; dataset finger friction2.0; other materials unchanged',
    finger_static_friction=2., finger_dynamic_friction=2.,
    finger_material_reference='ManiSkill Panda urdf_config gripper material',
    panda_finger_mimic=False, minimum_solver_iterations=None,
    max_policy_physics_steps=6000,
    arm_stiffness=1000., arm_damping=200., acceleration_feedforward=False,
    official_robotwin_commit='ea8b21121ebb3cd201ff5b3fe361944ac94eda3f',
    engine_version='recorded at runtime; not inferred from the controller profile',
    object_mass_kg=.05,
    scope='Official arm/gripper control and default solver; task-specific geometry, mass and planning constraints')
CONTACT_COMPLETION = dict(
    version='native_contact_completion_v1', native_contact_legs=1,
    start_state='measured_current_joints', target='unchanged_requested_contact_pose',
    preclose_hold_steps=0, terminal_position='last_native_contact_endpoint',
    terminal_drive_velocity='zero_during_native_close', gripper_steps=[200, 100])

RETIRED_OPTIONS = {
    'qualification_repair', 'pilot_repair', 'stable_placement',
    'qualification_snap_collision_bottom', 'cartesian_time_dilation_factor',
    'native_contact_refinement', 'native_contact_closure_hold', 'native_endpoint_arrival',
    'linear_placement_descent', 'unconstrained_placement_descent',
    'allow_placement_descent_backtracking',
}


def validate_config(config):
    """Reject old overrides instead of silently running different controllers."""
    retired = RETIRED_OPTIONS.intersection(config)
    if retired:
        raise ValueError('Remove retired per-record control options: ' + ', '.join(sorted(retired)))
    if type(config.get('initialization_keep_awake', False)) is not bool:
        raise ValueError('Initialization keep-awake must be a boolean')
    time_factor = config.get('placement_time_dilation_factor', 1.)
    if time_factor != 1.:
        raise ValueError('Unified V2 uses native timing with placement factor1.0')
    contract = config['gallery_contract']
    if set(contract) != {'grasp', 'placement'}:
        raise ValueError('The geometry contract must contain only grasp and placement')
    grasp_keys = {'id', 'official_contact_ids', 'contact_frame_asset_units',
                  'axial_retreat_m', 'downward_angle_deg', 'exchange_fingers', 'semantic_region',
                  'symmetric_jaw_fallback'}
    if set(contract['grasp']) - grasp_keys:
        raise ValueError('Unknown grasp geometry field')
    symmetric = contract['grasp'].get('symmetric_jaw_fallback', False)
    if type(symmetric) is not bool:
        raise ValueError('Symmetric jaw fallback must be a boolean')
    if symmetric and (contract['grasp'].get('contact_frame_asset_units') is None
            or type(contract['grasp'].get('exchange_fingers', False)) is not bool):
        raise ValueError('Symmetric jaw fallback requires a custom contact frame and boolean finger exchange')
    placement_keys = {'base_center_local_m', 'yaw_offset_rad'}
    if set(contract['placement']) - placement_keys:
        raise ValueError('Placement config may contain only the support reference and goal yaw')
    required = dict(official_robotwin_control=True, measured_segment_chaining=True,
                    curobo_num_trajopt_seeds=4, panda_finger_mimic=False,
                    continuous_pick_place=True, cartesian_pick_paths=True, fast_gripper=False,
                    direct_pick_approach=True,
                    parallel_jaw_flip=False, prefer_native_jaw_orientation=True)
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f'{key} must be {expected!r} for uniform collection')
    if type(config.get('return_to_origin')) is not bool:
        raise ValueError('return_to_origin must be a boolean')
    strategy = config['strategy']
    pass_through = strategy.get('sequence_rule') == 'requested_projection_exact'
    pregrasp_distance = config.get('native_pregrasp_distance_m')
    if pregrasp_distance is not None and not .01 <= float(pregrasp_distance) <= .1:
        raise ValueError('Task pregrasp distance requires measured official control and10–100mm')
    fixed_strategy = dict(motion_profile='native_curobo_v2', duration_scale=1.,
                          closed_loop=False, straight_lateral_segments=False,
                          ordered_route_guard=False, allow_natural_xy_curves=True,
                          allow_region_revisit=False)
    if pass_through:
        fixed_strategy['region_dwell_steps'] = 0
        if 'stop_visit' in strategy or 'expert_stop_minimum_samples' in strategy:
            raise ValueError('Pass-through routes must not carry stop criteria')
    else:
        fixed_strategy['sequence_rule'] = 'ordered_region_stops'
        dwell = strategy.get('region_dwell_steps')
        minimum = strategy.get('expert_stop_minimum_samples')
        if type(dwell) is not int or not 1 <= dwell <= 250:
            raise ValueError('Requested region dwell must be 1–250 physics steps')
        if type(minimum) is not int or not strategy['stop_visit']['minimum_samples'] <= minimum <= 250:
            raise ValueError('Expert stop window must cover the frozen stop criterion, within 250 steps')
    for key, expected in fixed_strategy.items():
        if strategy.get(key) != expected:
            raise ValueError(f'Uniform strategy requires {key}={expected!r}')
    reference = strategy.get('reference_point', 'object_origin')
    if reference not in ('object_origin', 'panda_hand'):
        raise ValueError('Unknown route reference point')
    offsets = np.asarray(strategy.get('waypoint_offsets_m', np.zeros((len(strategy['sequence']), 3))))
    if offsets.shape != (len(strategy['sequence']), 3) or not np.isfinite(offsets).all():
        raise ValueError('Expected one frozen finite offset per visit')
    namespace = config.get('task_namespace')
    if namespace == 'FIXED_WORLD_OBJECT_V2' and reference != 'object_origin':
        raise ValueError('Object-reference tasks must use the measured target actor origin')
    if reference == 'panda_hand':
        if namespace not in ('FIXED_WORLD_EE_V1', 'FIXED_WORLD_EE_V2'):
            raise ValueError('EE route semantics require a separate task namespace')
    if reference == 'panda_hand' or namespace == 'FIXED_WORLD_OBJECT_V2':
        if np.any(np.linalg.norm(offsets, axis=1) > .005 + 1e-9):
            raise ValueError('Frozen route target perturbations must stay within 5mm')
    elif np.any(offsets):
        raise ValueError('Historical object-reference routes use the fixed centers')


class DatasetCollectionTask(RecordingOpenRoutesTask):
    """Configured grasp geometry; identical dynamics and state-based placement."""

    policy_budget_steps = CONTROL_PROFILE['max_policy_physics_steps']

    def __init__(self, geometry):
        super().__init__()
        self.geometry_path = Path(geometry).resolve()

    def load_robot(self, **kwargs):
        super().load_robot(**kwargs)
        # Match ManiSkill Panda's finger friction without changing its other
        # contact parameters or mutating a material shared with palm/objects.
        for name in ('panda_leftfinger', 'panda_rightfinger'):
            body = self.robot.left_entity.find_link_by_name(name)
            for shape in body.get_collision_shapes():
                original = shape.get_physical_material()
                material = self.scene.create_physical_material(
                    CONTROL_PROFILE['finger_static_friction'],
                    CONTROL_PROFILE['finger_dynamic_friction'],
                    original.get_restitution())
                shape.set_physical_material(material)

    def check_stable(self):
        # Sleeping can conceal residual table-settling motion until the hand
        # approaches. This optional initialization policy keeps every object
        # awake during the existing settling interval, then restores its native
        # sleep threshold before initial RGB or any recorded task action.
        awake = getattr(self, 'initialization_keep_awake', False)
        thresholds = [body.get_sleep_threshold() for body in self.object_bodies] if awake else []
        try:
            if awake:
                for body in self.object_bodies:
                    body.set_sleep_threshold(0.)
                    body.wake_up()
            result = super().check_stable()
        finally:
            if awake:
                for body, threshold in zip(self.object_bodies, thresholds):
                    body.set_sleep_threshold(threshold)
        if awake:
            restored = [body.get_sleep_threshold() for body in self.object_bodies]
            if restored != thresholds:
                raise RuntimeError('Object sleep thresholds were not restored after initialization')
            quality = self.initialization_quality
            tail_ok = (np.asarray(quality['stable_tail_translation_m']) <= .0001) & (
                np.asarray(quality['stable_tail_rotation_deg']) <= .1)
            quality['awake_initialization'] = dict(
                enabled=True, scope='existing initialization interval only; all scene objects',
                original_sleep_thresholds=thresholds, restored_sleep_thresholds=restored,
                sleep_thresholds_restored=True, stable_tail_position_tolerance_m=.0001,
                stable_tail_rotation_tolerance_deg=.1, stable_tail_passed=tail_ok.tolist(),
                final_linear_speed_m_s=[float(np.linalg.norm(body.get_linear_velocity()))
                                       for body in self.object_bodies],
                final_angular_speed_deg_s=[float(np.degrees(np.linalg.norm(body.get_angular_velocity())))
                                          for body in self.object_bodies])
            quality['awake_initialization']['enforced'] = False
            # The tighter awake-tail guide is diagnostic. Native spawn
            # collision and the original settling admission remain authoritative.
        # Keep the admission evidence even when setup fails before recording.
        write(self.output/'initialization_quality.json', self.initialization_quality)
        if not result[0]:
            self._update_render()
            self.cameras.update_picture()
            rgb = self.cameras.get_rgb()
            np.savez_compressed(self.output/'initial_rgb.npz',
                head=rgb['head_camera']['rgb'], wrist=rgb['left_camera']['rgb'])
        return result

    def setup_demo(self, **kwargs):
        validate_config(kwargs)
        self.placement_time_dilation_factor = float(kwargs.get('placement_time_dilation_factor', 1.))
        self.retreat_linear_scale = float(kwargs.get('retreat_linear_scale', 1.))
        if not np.isfinite(self.retreat_linear_scale) or not 1. <= self.retreat_linear_scale <= 10.:
            raise ValueError('Retreat linear scale must be between 1 and 10')
        self.initialization_keep_awake = kwargs.get('initialization_keep_awake', False)
        self.official_robotwin_control = True
        self.panda_finger_mimic = []
        self.route_pass_through = kwargs['strategy']['sequence_rule'] == 'requested_projection_exact'
        self.measured_segment_chaining = True
        self.contact_completion_contract = copy.deepcopy(CONTACT_COMPLETION)
        self._contact_closure_joint_goal = None
        self.control_profile = CONTROL_PROFILE
        self.contract = kwargs['gallery_contract']
        self.scene_preset = kwargs['roboicl_scene_preset_config']
        self.output = Path(kwargs['save_path'])
        self.placement_trace = []
        self.route_execution = []
        self.planned_approach_arrays = {}
        self.wrist_calibration = None
        self.contact_diagnostics = []
        self.arrival_trace = []
        self.departure_trace = []
        self.pregrasp_distance = kwargs.get('native_pregrasp_distance_m')
        self.contact_path_clearance = kwargs.get('native_contact_path_clearance_m')
        if self.contact_path_clearance is not None and not .001 <= float(self.contact_path_clearance) <= .01:
            raise ValueError('Native contact-path clearance must be between1and10mm')
        self.departure_clearance = kwargs.get('carried_departure_clearance_m')
        if self.departure_clearance is not None and not .01 <= float(self.departure_clearance) <= .05:
            raise ValueError('Carried departure clearance must be between10and50mm')
        self.control_profile = dict(self.control_profile,
            pregrasp_standoff_m=(float(self.pregrasp_distance) if self.pregrasp_distance is not None
                                 else self.control_profile['pregrasp_standoff_m']),
            pregrasp_distance_reference=('clearance-adjusted grasp' if self.pregrasp_distance is not None
                                         else 'original native grasp'),
            additional_endpoint_holds=False,
            additional_contact_arrival_hold=False,
            native_contact_closure_hold=True,
            native_endpoint_arrival=None,
            route_sequence_rule=kwargs['strategy']['sequence_rule'],
            route_dwell_steps=kwargs['strategy']['region_dwell_steps'],
            measured_segment_chaining=self.measured_segment_chaining,
            native_contact_refinement=False,
            curobo_num_trajopt_seeds=4,
            linear_placement_descent=True, unconstrained_placement_descent=False,
            grasp_motion='official_v2_plan_grasp_approach_then_grasp',
            carried_collision='native_v2_attachment_manager_64_spheres_plus_existing_sweep',
            placement_motion='native_plan_pose_preplace_then_world_z_descent',
            placement_time_dilation_factor=self.placement_time_dilation_factor,
            grasp_approach_active_knot_fraction=None,
            empirical_motion_shape_policy='diagnostic_only',
            carried_departure_clearance_m=self.departure_clearance)
        self.control_profile['native_contact_refinement_contract'] = copy.deepcopy(self.contact_completion_contract)
        if self.initialization_keep_awake:
            self.control_profile['initialization_policy'] = 'all_objects_awake_during_native_settling'
        self.kin = FrankaKinematics(self.geometry_path)
        result = super().setup_demo(**kwargs)
        self.robot.left_planner.enforce_path_shape = False
        # Dataset uses native V2 planning with measured-start execution.
        self.robot.left_planner.continuous_pick_place = False
        self.robot.left_planner.native_contact_approach_distance_m = self.control_profile['pregrasp_standoff_m']
        self.robot.left_planner.native_contact_path_clearance_m = self.contact_path_clearance
        if self.panda_finger_mimic or self.execution_dynamics is not None:
            raise RuntimeError('Non-official coupling or acceleration feedforward remains enabled')
        for joint in self.robot.left_entity.get_active_joints():
            if joint.stiffness != 1000. or joint.damping != 200.:
                raise RuntimeError('Official controller requires K1000/D200 for every active joint')
        write(self.output/'grip_drive_runtime.json', dict(joints=[dict(
            stiffness=j.stiffness, damping=j.damping, force_limit=j.force_limit,
            mode=str(j.drive_mode)) for j in self.robot.left_entity.get_active_joints()[7:9]]))
        write(self.output/'collection_control_profile.json', self.control_profile)
        import importlib.metadata
        write(self.output/'runtime_versions.json', {
            name: importlib.metadata.version(name) for name in ('sapien', 'torch', 'numpy')})
        write(self.output/'coupling_solver_runtime.json', dict(
            left_robot=dict(position=self.robot.left_entity.get_solver_position_iterations(),
                            velocity=self.robot.left_entity.get_solver_velocity_iterations()),
            objects=[dict(position=b.get_solver_position_iterations(),
                          velocity=b.get_solver_velocity_iterations()) for b in self.object_bodies],
            coupling=self.panda_finger_mimic))
        physics = {}
        for name in ('panda_hand', 'panda_leftfinger', 'panda_rightfinger'):
            body = self.robot.left_entity.find_link_by_name(name)
            physics[name] = [dict(contact_offset_m=s.get_contact_offset(),
                rest_offset_m=s.get_rest_offset(), min_patch_radius_m=s.get_min_patch_radius(),
                patch_radius_m=s.get_patch_radius(),
                static_friction=s.get_physical_material().get_static_friction(),
                dynamic_friction=s.get_physical_material().get_dynamic_friction())
                for s in body.get_collision_shapes()]
        write(self.output/'gripper_contact_runtime.json', physics)
        write(self.output/'object_material_runtime.json', [dict(
            mass_kg=float(body.mass), shapes=[dict(
                static_friction=s.get_physical_material().get_static_friction(),
                dynamic_friction=s.get_physical_material().get_dynamic_friction(),
                restitution=s.get_physical_material().get_restitution())
                for s in body.get_collision_shapes()]) for body in self.object_bodies])
        return result

    def validate_object_specs(self, specs):
        identities = [(s['asset'], s['model_id']) for s in specs]
        if not 1 <= len(identities) <= 3 or len(set(identities)) != len(identities):
            raise ValueError('Dataset scenes require one to three distinct object identities')

    def create_table_and_wall(self, table_xy_bias=None, table_height=.74):
        return roboicl_select_scene.create_table_and_wall(
            self, [0, 0] if table_xy_bias is None else table_xy_bias, table_height)

    def add_prohibit_area(self, actor, *args, **kwargs):
        # Native actors without metadata still need their geometric contact list.
        if actor.config is None:
            # Several task objects can use different models of the same asset.
            # Bind metadata to the actor's slot, not the first matching asset name.
            asset = self.object_specs[self.objects.index(actor)]
            actor.config = read(REPOSITORY/'assets/objects'/asset['asset']/f"model_data{asset['model_id']}.json")
            actor.config = copy.deepcopy(actor.config)
            actor.config.setdefault('scale', [1., 1., 1.])
        actor.config.setdefault('contact_points_pose', [])
        return super().add_prohibit_area(actor, *args, **kwargs)

    def get_grasp_pose(self, actor, arm_tag, contact_point_id=0, pre_dis=0.):
        grasp = self.contract['grasp']
        retreat = float(grasp.get('axial_retreat_m', 0.))
        if not 0. <= retreat <= .03:
            raise ValueError('Grasp axial retreat must be within 0–30 mm')
        pre_dis += retreat
        if grasp.get('contact_frame_asset_units') is None:
            return super().get_grasp_pose(actor, arm_tag, contact_point_id, pre_dis)
        frame = actor.get_contact_point(contact_point_id, 'matrix') @ np.array(
            [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        rotation = frame[:3, :3]
        if getattr(self, '_candidate_exchange_fingers', grasp.get('exchange_fingers', False)):
            rotation = rotation @ np.diag([1., -1., -1.])
        return ((frame[:3, 3] + rotation @ np.array([-.12-pre_dis, 0, 0])).tolist()
                + t3d.quaternions.mat2quat(rotation).tolist())

    def _apply_grasp_clearance(self, result, actor, mesh_margin, sphere_margin, palm_margin):
        result = super()._apply_grasp_clearance(result, actor, mesh_margin, sphere_margin, palm_margin)
        if self.pregrasp_distance is not None:
            # Change only the task waypoint; the parent plans both native paths
            # after this method, including its ordinary collision checks.
            moves = [action for action in result[1] if action.action == 'move']
            pose = np.asarray(moves[-1].target_pose, dtype=float).copy()
            original = list(moves[0].target_pose)
            pose[:3] -= float(self.pregrasp_distance) * t3d.quaternions.quat2mat(pose[3:])[:, 0]
            moves[0].target_pose = pose.tolist()
            self.grasp_clearance_audit.update(native_pregrasp_distance_m=float(self.pregrasp_distance),
                original_pregrasp_pose=original, task_pregrasp_pose=pose.tolist())
        return result

    def clearance_grasp_actor(self, actor, arm_tag, mesh_margin=.003,
                             sphere_margin=.001, palm_margin=.001, contact_point_ids=None):
        grasp = self.contract['grasp']
        self.approach_start_q = np.asarray(self.robot.left_entity.get_qpos())[:7].copy()
        symmetric = grasp.get('symmetric_jaw_fallback', False)
        if symmetric and (self.physics_step != 0 or self.recorder.executions):
            raise RuntimeError('Symmetric jaw fallback must precede all recorded action')
        configured_exchange = grasp.get('exchange_fingers', False)
        selection = (dict(version='native_symmetric_jaw_fallback_v1',
            configured_exchange_fingers=configured_exchange,
            candidate_order=[configured_exchange, not configured_exchange],
            selected_index=None, chosen_exchange_fingers=None, branches=[],
            grasp_contract_sha256=hashlib.sha256(json.dumps(grasp, sort_keys=True,
                separators=(',', ':')).encode('utf-8')).hexdigest()) if symmetric else None)
        missing = object()
        previous_exchange = getattr(self, '_candidate_exchange_fingers', missing)
        original = actor.config
        try:
            frame = grasp.get('contact_frame_asset_units')
            if frame is not None:
                actor.config = copy.deepcopy(original)
                actor.config['contact_points_pose'].append(frame)
                allowed = [len(actor.config['contact_points_pose']) - 1]
            else:
                allowed = grasp['official_contact_ids']
            for index, exchange in enumerate(selection['candidate_order'] if symmetric else [configured_exchange]):
                branch = None
                if symmetric:
                    if arm_tag != 'left':
                        raise ValueError('Symmetric jaw fallback requires the recorded left arm')
                    self._candidate_exchange_fingers = exchange
                    for label in ('direct', 'pregrasp', 'grasp'):
                        setattr(self, 'qualified_' + label + '_path', None)
                        for key in ('position', 'velocity'):
                            self.planned_approach_arrays.pop(label + '_' + key, None)
                    self.grasp_clearance_audit = {}
                    branch = dict(index=index, exchange_fingers=exchange, status='pending', reason=None,
                        before_physics_step=self.physics_step,
                        before_qpos=self.robot.left_entity.get_qpos().tolist(),
                        before_execution_count=len(self.recorder.executions),
                        plan_log_begin=len(self.recorder.plans), native_paths={})
                    selection['branches'].append(branch)
                    original_plan = self.robot.left_plan_path
                    plan_override = self.robot.__dict__.get('left_plan_path', missing)
                    planner = self.robot.left_planner
                    original_cartesian = planner.cartesian_segment

                    def record_candidate_path(target_pose, *args, **kwargs):
                        label = 'grasp' if kwargs.get('last_qpos') is not None else 'pregrasp'
                        assert label not in branch['native_paths'], 'Repeated path in one symmetric jaw candidate'
                        start = kwargs.get('last_qpos')
                        item = dict(status='pending', requested_pose=list(target_pose),
                            constraint_pose=copy.deepcopy(kwargs.get('constraint_pose')),
                            start_qpos=np.asarray(self.robot.left_entity.get_qpos() if start is None else start).tolist(),
                            plan_log_begin=len(self.recorder.plans),
                            array_prefix=f'branch_{index}_{label}')
                        branch['native_paths'][label] = item
                        try:
                            path = original_plan(target_pose, *args, **kwargs)
                            item['status'] = path['status']
                            for key in ('position', 'velocity'):
                                if key in path:
                                    values = np.asarray(path[key]).copy()
                                    self.planned_approach_arrays[item['array_prefix']+'_'+key] = values
                                    item[key+'_shape'] = list(values.shape)
                                    item[key+'_sha256'] = hashlib.sha256(values.tobytes()).hexdigest()
                            return path
                        finally:
                            item['plan_log_end'] = len(self.recorder.plans)
                            if item['status'] == 'pending':
                                item['status'] = 'exception'

                    self.robot.left_plan_path = record_candidate_path
                try:
                    result = super().clearance_grasp_actor(actor, arm_tag, mesh_margin,
                        sphere_margin, palm_margin, contact_point_ids=allowed)
                    angle = float(np.degrees(np.arcsin(np.clip(
                        -self.grasp_clearance_audit['approach_z'], -1., 1.))))
                    low, high = grasp['downward_angle_deg']
                    if not low <= angle <= high:
                        raise RuntimeError('Selected grasp outside requested approach class')
                    segments = ([self.qualified_direct_path] if self.qualified_direct_path is not None
                                else [self.qualified_pregrasp_path, self.qualified_grasp_path])
                    self.grasp_clearance_audit.update(
                        requested_grasp=grasp['id'], allowed_contact_ids=allowed,
                        approach_downward_degrees=angle,
                        requested_axial_retreat_m=grasp.get('axial_retreat_m', 0.))
                    self.record_approach_quality(segments)
                    if symmetric:
                        branch['status'] = 'accepted'
                        selection.update(selected_index=index, chosen_exchange_fingers=exchange)
                    return result
                except RuntimeError as error:
                    if not symmetric:
                        raise
                    reason = str(error)
                    branch.update(status='error', reason=reason)
                    if reason == 'No native contact has feasible pregrasp and clearance-adjusted approach':
                        geometric = {
                            'Native grasp axial retreat cannot satisfy table clearance',
                            'Native grasp frame retreat does not separate the palm from the target',
                            'Required grasp retreat exceeds native pregrasp distance',
                            'No physical palm-clear grasp within the native pregrasp distance'}
                        errors = [attempt['error'] for attempt in self.grasp_clearance_audit['candidate_attempts']
                                  if 'error' in attempt]
                        if any(message not in geometric for message in errors):
                            raise  # Parent-caught programming/runtime errors are not fallback admission.
                        branch['status'] = 'geometry_rejected' if errors else 'native_rejected'
                    else:
                        raise
                finally:
                    if symmetric:
                        if plan_override is missing:
                            del self.robot.left_plan_path
                        else:
                            self.robot.left_plan_path = plan_override
                        planner.cartesian_segment = original_cartesian
                        branch.update(plan_log_end=len(self.recorder.plans),
                            after_physics_step=self.physics_step,
                            after_qpos=self.robot.left_entity.get_qpos().tolist(),
                            after_execution_count=len(self.recorder.executions),
                            approach_quality=copy.deepcopy(self.grasp_clearance_audit.get('approach_quality')),
                            clearance=copy.deepcopy(self.grasp_clearance_audit))
                        if branch['status'] == 'pending':
                            branch.update(status='error', reason='Unexpected exception before candidate admission')
                        self.grasp_clearance_audit['symmetric_jaw_selection'] = selection
                        self.set_pregrasp_obstacles(True)
                        assert branch['before_physics_step'] == branch['after_physics_step']
                        assert branch['before_qpos'] == branch['after_qpos']
                        assert branch['before_execution_count'] == branch['after_execution_count']
                        if index:
                            first = selection['branches'][0]
                            assert branch['before_physics_step'] == first['before_physics_step']
                            assert branch['before_qpos'] == first['before_qpos']
            raise RuntimeError('No symmetric jaw orientation passed native planning and collision admission')
        finally:
            actor.config = original
            if previous_exchange is missing:
                if hasattr(self, '_candidate_exchange_fingers'):
                    del self._candidate_exchange_fingers
            else:
                self._candidate_exchange_fingers = previous_exchange
            for label in ('direct', 'pregrasp', 'grasp'):
                if symmetric and selection['selected_index'] is None:
                    setattr(self, 'qualified_' + label + '_path', None)
                    for key in ('position', 'velocity'):
                        self.planned_approach_arrays.pop(label + '_' + key, None)
                    continue
                path = getattr(self, 'qualified_' + label + '_path', None)
                if isinstance(path, dict):
                    for key in ('position', 'velocity'):
                        if key in path:
                            self.planned_approach_arrays[label + '_' + key] = np.asarray(path[key]).copy()

    def record_approach_quality(self, segments):
        q = np.concatenate([self.approach_start_q[None]]
                           + [np.asarray(p['position'])[:, :7] for p in segments])
        xyz = np.stack([self.kin.pose(np.r_[row, 1.])[:3, 3] for row in q])
        length = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
        chord = float(np.linalg.norm(xyz[-1] - xyz[0]))
        span = float(np.degrees(np.ptp(q, axis=0)).max())
        ratio = length / max(chord, 1e-9)
        quality = dict(ee_length_m=length, chord_m=chord, chord_ratio=ratio,
            max_joint_span_deg=span, passed=bool(ratio <= 1.75 and span <= 120),
            enforced=False, role='diagnostic_only',
            reference_limits=dict(chord_ratio=1.75, max_joint_span_deg=120))
        self.grasp_clearance_audit['approach_quality'] = quality

    def execute_arm_path(self, path):
        # Execute the official V2 approach and grasp arrays without retiming.
        # The native segment boundary is retained; no extra hold is inserted.
        contact = (path is getattr(self, 'qualified_direct_path', None)
                   and self.carried_vertices_in_hand is None)
        if not contact:
            return super().execute_arm_path(path)
        q = self.robot.left_entity.get_qpos().copy()
        trace = dict(segment='grasp', begin_step=self.physics_step,
            start_qpos=q.tolist(), constraint_pose=None,
            planner_method='native_plan_grasp_two_phases',
            requested_pose=self.grasp_clearance_audit['corrected_high_pose'],
            attempts=[dict(status=path['status'],
                physical_target_sweep=path['cartesian_path']['target_mesh_sweep'])],
            status=path['status'], before_hand_pose=self.object_state()['hand_pose'],
            source='measured current joints; official native approach then grasp')
        self.grasp_clearance_audit.update(
            execution_approach='official_v2_approach_then_grasp', contact_replans=[trace],
            contact_replan=trace, official_plan_grasp_phases=path['cartesian_path']['native_phases'])
        for key in ('position', 'velocity'):
            self.planned_approach_arrays['grasp_' + key] = np.asarray(path[key]).copy()
        try:
            result = super().execute_arm_path(path)
        finally:
            trace.update(end_step=self.physics_step,
                after_hand_pose=self.object_state()['hand_pose'],
                native_steps=len(path['position']))
        self._contact_closure_joint_goal = np.asarray(path['position'][-1, :7], dtype=np.float32).copy()
        self.grasp_clearance_audit['contact_completion'] = dict(
            contract=copy.deepcopy(self.contact_completion_contract), native_contact_end_step=self.physics_step,
            native_joint_goal=self._contact_closure_joint_goal.tolist(),
            requested_pose=self.grasp_clearance_audit['corrected_high_pose'],
            measured_hand_before_close=self.object_state()['hand_pose'])
        return result

    def take_dense_action(self, sequence, save_freq=-1):
        goal = getattr(self, '_contact_closure_joint_goal', None)
        gripper = sequence.get('left_gripper')
        closing = (goal is not None and gripper is not None and
                   float(gripper['result'][-1]) < self.robot.get_left_gripper_val())
        if not closing:
            return super().take_dense_action(sequence, save_freq)
        if sequence.get('left_arm') is not None or sequence.get('right_arm') is not None or sequence.get('right_gripper') is not None:
            raise RuntimeError('Contact closure must follow its native arm path as a left-gripper action')
        # The native300-step gripper action is unchanged. Set the arm endpoint's
        # velocity target to zero in those SAME steps, without a preclose wait
        # or any reset of the measured articulation position/velocity.
        steps = int(gripper['num_step'])
        combined = dict(sequence)
        combined['left_arm'] = dict(status='Success',
            position=np.repeat(goal[None], steps, axis=0),
            velocity=np.zeros((steps, 7), dtype=np.float32),
            native_contact_closure_hold=True)
        completion = self.grasp_clearance_audit['contact_completion']
        completion.update(close_begin_step=self.physics_step, planned_close_steps=steps)
        try:
            return super().take_dense_action(combined, save_freq)
        finally:
            completion.update(close_end_step=self.physics_step,
                actual_close_steps=self.physics_step-completion['close_begin_step'])
            self._contact_closure_joint_goal = None


    def set_gripper(self, set_tag='together', left_pos=None, right_pos=None):
        return Base_Task.set_gripper(self, set_tag, left_pos, right_pos)

    def placement_hand(self, bottom):
        hand = np.asarray(self.object_state()['hand_pose'])
        actor = self.target.get_pose().to_transformation_matrix()
        attachment = np.linalg.inv(hand) @ actor
        desired = np.eye(4)
        initial = self.recorder.rows[0]['object_poses'][self.target_index]
        desired[:3, :3] = t3d.quaternions.quat2mat(np.asarray(initial)[3:])
        placement = self.contract['placement']
        yaw = float(placement.get('yaw_offset_rad', 0.))
        desired[:3, :3] = t3d.axangles.axangle2mat([0, 0, 1], yaw) @ desired[:3, :3]
        reference = desired[:3, :3] @ np.asarray(placement['base_center_local_m'])
        desired[:2, 3] = self.destination - reference[:2]
        desired[2, 3] = bottom - np.min(self.object_collision_vertices[self.target_index] @ desired[2, :3])
        return desired @ np.linalg.inv(attachment)

    def move_from_drive(self, stage, displacement=None, target_hand=None):
        planner = self.robot.left_planner
        if stage == 'lift' and self.carried_vertices_in_hand is None and planner.payload_attached:
            # Exit the released contact along negative tool Z before restoring
            # the target's buffered obstacle and planning a joint-space return.
            hand = np.asarray(self.object_state()['hand_pose'])
            goal = hand.copy()
            goal[:3,3] -= .1*hand[:3,2]
            error = RecordingSelectTask.move_from_drive(self, 'retreat', target_hand=goal)
            world = self.set_pregrasp_obstacles(True)
            planner.attachment_events.append(dict(action='restore_released_obstacle',
                physics_step=self.physics_step, released_world=world))
            planner.save_evidence('v2_attachments.json', planner.attachment_events)
            return error
        if stage != 'place' or self.carried_vertices_in_hand is None:
            return super().move_from_drive(stage, displacement, target_hand)
        planner = self.robot.left_planner
        keys = ('continuous_pick_place', 'place_hold_orientation', 'allow_natural_route_curves', 'free_place_motion', 'placement_time_dilation_factor')
        previous = {key: getattr(planner, key, 1. if key == 'placement_time_dilation_factor' else False)
                    for key in keys}
        previous_line_axis = getattr(planner, 'route_world_line_axis', None)
        planner.continuous_pick_place = False
        planner.allow_natural_route_curves = True
        planner.place_hold_orientation = True
        try:
            if self.departure_clearance is not None:
                # Clear surrounding objects vertically before the first requested
                # region. Requested A/B centers and visitation order stay fixed.
                state = self.object_state()
                highest = max((float((vertices @ actor.get_pose().to_transformation_matrix()[:3, :3].T
                               + actor.get_pose().p)[:, 2].max())
                    for i, (actor, vertices) in enumerate(zip(self.objects, self.object_collision_vertices))
                    if i != self.target_index), default=.74)
                bottom = float(state['object_bottom_z'][self.target_index])
                rise = max(0., highest+float(self.departure_clearance)-bottom)
                if rise > .001:
                    begin = self.physics_step
                    RecordingSelectTask.move_from_drive(self, 'lift', displacement=[0., 0., rise])
                    self.departure_trace.append(dict(begin_step=begin, end_step=self.physics_step,
                        rise_m=rise, non_target_top_m=highest, requested_clearance_m=self.departure_clearance,
                        measured_bottom_m=self.object_state()['object_bottom_z'][self.target_index]))
            regions = {r['id']: r for r in self.strategy['regions']}
            offsets = self.strategy.get('waypoint_offsets_m', np.zeros((len(self.strategy['sequence']), 3)))
            for name, offset in zip(self.strategy['sequence'], offsets):
                self.move_reference(name, np.array(regions[name]['center_m']) + np.asarray(offset),
                                    dwell=self.strategy['region_dwell_steps'])
            # Finish lateral transfer and orientation above the support, then
            # use native linear-motion criteria only for the short descent.
            # Refresh the measured attachment at each planning boundary.
            planner.place_hold_orientation = False
            planner.free_place_motion = True
            planner.placement_time_dilation_factor = self.placement_time_dilation_factor
            planner.route_world_line_axis = None
            release_bottom = self.destination_top_z_m + self.control_profile['release_gap_m']
            for phase, motion_stage, bottom in (
                    ('preplace', 'place', release_bottom + self.control_profile['preplace_height_m']),
                    ('place', 'place_descent', release_bottom)):
                goal = self.placement_hand(bottom)
                hand = np.asarray(self.object_state()['hand_pose'])
                angle = float(np.degrees(np.arccos(np.clip(
                    (np.trace(hand[:3, :3].T @ goal[:3, :3])-1.)/2., -1., 1.))))
                begin = self.physics_step
                RecordingSelectTask.move_from_drive(self, motion_stage, target_hand=goal)
                linear = motion_stage == 'place_descent'
                self.placement_trace.append(dict(phase=phase, begin_step=begin,
                    end_step=self.physics_step, requested_bottom_m=bottom,
                    requested_orientation_change_deg=angle, hold_orientation=linear,
                    time_dilation_factor=self.placement_time_dilation_factor,
                    intermediate_pose_constraint=linear, world_line_axis='z' if linear else None,
                    requested_hand_goal_world=goal.tolist()))
        finally:
            planner.route_world_line_axis = previous_line_axis
            for key, value in previous.items():
                setattr(planner, key, value)

    def _step_scene(self):
        result = super()._step_scene()
        if self.recorder is not None:
            body = self.object_bodies[self.target_index]
            contacts = [dict(bodies=[b.entity.name for b in c.bodies], points=[dict(
                position=p.position.tolist(), normal=p.normal.tolist(),
                separation_m=float(p.separation), impulse_ns=p.impulse.tolist()) for p in c.points])
                for c in self.scene.get_contacts() if body in c.bodies]
            self.contact_diagnostics.append(dict(step=self.physics_step,
                sleeping=bool(body.is_sleeping), sleep_threshold=float(body.get_sleep_threshold()),
                contacts=contacts))
        return result

    def close_env(self, clear_cache=False):
        if hasattr(self, 'cameras') and hasattr(self, 'robot'):
            self._update_render()
            camera = self.cameras.get_config().get('left_camera')
            if camera is not None:
                hand = np.asarray(self.object_state()['hand_pose'])
                self.wrist_calibration = dict(camera,
                    hand_pose_world_at_capture=hand,
                    camera_gl_in_hand=np.linalg.inv(hand) @ np.asarray(camera['cam2world_gl']),
                    capture='Native camera configuration before shutdown; no physics step.')
        if self.contact_diagnostics:
            write(self.output/'sleep_contact_diagnostics.json', self.contact_diagnostics)
        return super().close_env(clear_cache)


def collect(task, config_path, output):
    config = read(config_path)
    validate_config(config)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Retained attempt already exists: ' + str(output))
    started = time.perf_counter()
    receipt = run_episode(task, config_path, output, task.geometry_path)
    receipt['initialization_quality'] = task.initialization_quality
    if task.grasp_clearance_audit:
        receipt['execution_approach'] = task.grasp_clearance_audit.get('execution_approach')
    if 'placement_goal' in receipt:
        receipt['superseded_parent_placement_goal'] = receipt.pop('placement_goal')
    receipt['placement_goal'] = dict(
        method='configured support reference using the measured object-in-hand transform',
        completed_phase_goals=task.placement_trace,
        authoritative_planner_inputs='planner_calls.json: target for each recorded plan_path call')
    write(output/'receipt.json', receipt)
    for name, value in [('head_camera_calibration', getattr(task, 'head_calibration', None)),
                        ('api_frame', getattr(task, 'initial_api_frame', None)),
                        ('route_execution', task.route_execution),
                        ('placement_trace', task.placement_trace),
                        ('endpoint_arrival', task.arrival_trace),
                        ('departure_clearance', task.departure_trace),
                        ('left_camera_mount_calibration', task.wrist_calibration)]:
        write(output/(name + '.json'), value)
    if task.planned_approach_arrays:
        np.savez_compressed(output/'planned_approach.npz', **task.planned_approach_arrays)
    sources = [Path(__file__), *[REPOSITORY/p for p in (
        'scripts/collect_roboicl_open_routes.py',
        'envs/robot/roboicl_v2_planner.py',
        'scripts/score_roboicl_open_routes.py', 'envs/roboicl_strategy_regions.py',
        'envs/roboicl_select_scene.py')]]
    for source in sources:
        shutil.copy2(source, output/'source'/source.name)
    write(output/'collection_source_hashes.json', {str(p.relative_to(REPOSITORY)): sha(p) for p in sources})
    recipe = read(output/'frozen_recipe.json')
    profile = task.control_profile
    recipe.update(collection_control_profile=profile,
                  gripper_ramp_and_settle_steps=profile['close_steps'],
                  release_clearance_m=profile['release_gap_m'],
                  release_object_bottom_m=getattr(task, 'destination_top_z_m', .741)+profile['release_gap_m'])
    recipe['gripper_operation_steps']['close'] = profile['close_steps']
    write(output/'frozen_recipe.json', recipe)
    result = dict(id=config['gallery_case']['id'], case=config['gallery_case'],
        status=receipt['status'], physical_success=receipt.get('task_success', False),
        data_quality_passed=receipt.get('data_quality_passed', False), error=receipt.get('error'),
        stages=receipt.get('stages', {}), grasp=task.contract['grasp']['id'],
        requested_grasp=task.contract['grasp']['id'], policy_seconds=receipt.get('policy_elapsed_s'))
    if (output/'grasp_clearance.json').exists():
        grasp = read(output/'grasp_clearance.json')
        result.update(selected_contact=grasp.get('selected_contact_id'),
            approach_z=grasp.get('approach_z'), approach_quality=grasp.get('approach_quality'),
            approach_downward_degrees=grasp.get('approach_downward_degrees'))
    if (output/'measurements.npz').exists():
        with np.load(output/'measurements.npz', allow_pickle=False) as archive:
            measured = {k: archive[k] for k in archive.files}
        result['initial_physics_sha256'] = hashlib.sha256(
            measured['qpos'][0].tobytes() + measured['object_poses'][0].tobytes()).hexdigest()
        final = measured['object_poses'][-1, config['target_index']]
        center = final[:3] + t3d.quaternions.quat2mat(final[3:]) @ np.asarray(
            task.contract['placement']['base_center_local_m'])
        result['body_center_error_m'] = float(np.linalg.norm(center[:2] - config['destination']))
        if receipt['status'] == 'completed':
            metrics = score_open_route(measured, config, read(output/'contacts.json'))
            write(output/'strategy_metrics.json', metrics)
            result.update(route_success=bool(metrics['route_success']),
                joint_success=bool(receipt['qualified'] and metrics['route_success']
                                   and result['body_center_error_m'] <= .01))
    result['wall_seconds'] = time.perf_counter() - started
    write(output/'feasibility_result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--geometry', type=Path, required=True)
    args = parser.parse_args()
    jobs = read(args.batch)
    for job in jobs:
        validate_config(read(job['config']))
        if Path(job['output']).exists():
            raise FileExistsError('Retained attempt already exists: ' + job['output'])
    task = DatasetCollectionTask(args.geometry)
    results = []
    for job in jobs:
        parent = Path(job['output']).resolve().parent
        parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(parent).free < 25*1024**3:
            raise RuntimeError('Insufficient storage reserve for another recording')
        result = collect(task, Path(job['config']), Path(job['output']))
        results.append(result)
        write(args.batch.with_suffix('.results.json'), results)
        print(json.dumps(dict(id=job['id'], completed=len(results), total=len(jobs),
            status=result['status'], physical=result['physical_success'],
            route=result.get('route_success', False), error=result['error'],
            wall_seconds=result['wall_seconds'])), flush=True)


if __name__ == '__main__':
    main()
