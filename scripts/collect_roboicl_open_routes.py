"""Native route recording support for the current Dataset collector."""
from pathlib import Path
import shutil
import sys

import numpy as np
import transforms3d as t3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envs.roboicl_strategy_regions import roboicl_strategy_regions
from collect_roboicl_curobo_select import RecordingSelectTask
from collect_roboicl_curobo_pick import command, write, sha
from score_roboicl_open_routes import STOP_SPEED_WINDOW_S, route_reference_speed

class RecordingOpenRoutesTask(RecordingSelectTask, roboicl_strategy_regions):
    supports_separate_pregrasp = True

    def setup_demo(self, **kwargs):
        strategy = kwargs['strategy']
        sequence = strategy['sequence']
        if not 1 <= len(sequence) <= len(strategy['regions']):
            raise ValueError('Requested visits must fit the configured route regions')
        if len(set(sequence)) != len(sequence) and not strategy.get('allow_region_revisit', False):
            raise ValueError('Repeated regions require allow_region_revisit')
        if any(a == b for a, b in zip(sequence, sequence[1:])):
            raise ValueError('Consecutive identical regions have no exit/re-entry')
        region_ids = {r['id'] for r in strategy['regions']}
        if len(strategy['regions']) != 5 or len(region_ids) != 5 or not set(sequence) <= region_ids:
            raise ValueError('Five unique detectors and known requested IDs are required')
        offsets = np.asarray(strategy.get('waypoint_offsets_m', np.zeros((len(sequence), 3))), dtype=float)
        if offsets.shape != (len(sequence), 3) or not np.isfinite(offsets).all():
            raise ValueError('Expected one finite XYZ offset per requested visit')
        regions = {r['id']: r for r in strategy['regions']}
        for name, offset in zip(sequence, offsets):
            if np.any(offset) and (regions[name].get('shape') != 'sphere' or
                    np.linalg.norm(offset) > regions[name]['radius_m'] - .008):
                raise ValueError('Waypoint offset must retain an 8 mm interior margin')
        if strategy.get('motion_profile') != 'native_curobo_v2':
            raise ValueError('Only native_curobo_v2 is supported; use frozen source for historical timing profiles')
        if strategy.get('duration_scale',1.)!=1.:
            raise ValueError('Native cuRobo timing must not receive a clock multiplier')
        minimum = strategy.get('expert_stop_minimum_samples')
        if minimum is not None and (not isinstance(minimum, int) or isinstance(minimum, bool)
                or strategy.get('sequence_rule') != 'ordered_region_stops'
                or minimum < strategy['stop_visit']['minimum_samples']):
            raise ValueError('Expert stop duration must be at least the scoring minimum')
        result=super().setup_demo(**kwargs)
        api = np.asarray(self.robot.get_left_ee_pose(), dtype=float)
        transform = np.eye(4)
        transform[:3, :3] = t3d.quaternions.quat2mat(api[3:7])
        transform[:3, 3] = api[:3]
        self.initial_api_frame = dict(initial_api=api.tolist(),
            api_to_hand=(np.linalg.inv(transform) @ np.asarray(self.object_state()['hand_pose'])).tolist())
        if strategy['motion_profile']=='native_curobo_v2':
            # Keep the physical scene adapter, but use plan_pose's original
            # interpolated clock for approach and every carried segment.
            self.robot.left_planner.continuous_pick_place=False
            output=Path(kwargs['save_path']);native=Path(self.robot.left_planner.yml_path)
            shutil.copyfile(native,output/'source/franka_curobo.yml')
            write(output/'native_clock_definition.json',dict(
                method='RoboTwin execution of native cuRobo V2 plan_pose clock',robot_config_sha256=sha(native),
                interpolation_dt_s=.004,extra_clock_multiplier=1.,custom_route_retiming=False,
                grasp_approach_toppra=False,task_specific_region_dwell_s=strategy['region_dwell_steps']*.004,
                note='Scene/region/grasp/placement constraints remain task-specific; no universal fixed EE speed is asserted'))
        return result

    def execute_arm_path(self, path):
        if path['status']=='Success':
            if self.carried_vertices_in_hand is not None and self.strategy.get('ordered_route_guard',False):
                blocked=self.blocked_requested_region(path['position'])
                if blocked is not None:
                    write(self.recorder.output/f'native_order_rejection_{self.physics_step}.json',blocked)
                    raise RuntimeError('Native path enters a noncurrent requested region: '+blocked['id'])
            self.robot.left_planner.validate_path(path)
            path=dict(path,cartesian_path=dict(path.get('cartesian_path',{}),native_timing=dict(
                method='Unmodified cuRobo V2 interpolated position and velocity; native robot limits',
                physics_dt_s=.004,samples=len(path['position']),duration_s=(len(path['position'])-1)*.004,
                duration_scale=1.,post_retiming=False,native_constraints_feasible=True)))
        return super().execute_arm_path(path)

    def blocked_requested_region(self, q):
        """Only protect requested identities; all other regions remain free."""
        start = self.recorder.kin.pose(np.r_[q[0,:7],0.])
        actor = np.asarray(self.object_state()['object_poses'][self.target_index][:3])
        offset = start[:3,:3].T @ (actor-start[:3,3])
        poses = np.stack([self.recorder.kin.pose(np.r_[row[:7],0.]) for row in q])
        xyz = poses[:,:3,3] + np.einsum('nij,j->ni',poses[:,:3,:3],offset)
        current = getattr(self, '_route_target', None)
        for region in self.strategy['regions']:
            if region['id'] not in self.strategy['sequence'] or region['id']==current:
                continue
            distance = np.linalg.norm(xyz-region['center_m'],axis=1)
            begin = 0
            if region['id']==getattr(self, '_last_region', None) and distance[0] <= region['radius_m']+.01:
                outside = np.flatnonzero(distance>region['radius_m']+.01)
                begin = int(outside[0]) if len(outside) else len(distance)
            hits = np.flatnonzero(distance[begin:]<region['radius_m']+.008)
            if len(hits):return dict(region,first_index=begin+int(hits[0]),
                minimum_center_distance_m=float(distance[begin:].min()))
        return None

    def route_reference_position(self, state):
        if self.strategy.get('reference_point', 'object_origin') == 'panda_hand':
            return np.asarray(state['hand_pose'])[:3, 3]
        return np.asarray(state['object_poses'][self.target_index][:3])

    def move_reference(self, name, xyz, dwell=0):
        state = self.object_state()
        hand = self.recorder.kin.pose(command(np.asarray(state['left_qpos'])))
        hand[:3, 3] += np.asarray(xyz)-self.route_reference_position(state)
        begin = self.physics_step
        planner = self.robot.left_planner
        previous_axis = getattr(planner, 'route_world_line_axis', None)
        previous_target = getattr(self, '_route_target', None)
        self._route_target = name
        delta = np.asarray(xyz)-self.route_reference_position(state)
        if self.strategy.get('straight_lateral_segments', False):
            planner.route_world_line_axis = 0 if abs(delta[0])>.05 and np.max(abs(delta[1:]))<=.003 else None
        try:
            RecordingSelectTask.move_from_drive(self, 'place', target_hand=hand)
        finally:
            planner.route_world_line_axis = previous_axis
            self._route_target = previous_target
        arrival_step = self.physics_step
        if dwell:
            self.hold_region(dwell)
        # Requested regions are intentional stops. Bound any residual settling;
        # there is no auxiliary clearance waypoint or auxiliary dwell.
        settling = 0
        # The original100-step bound includes the scoring20-sample stop.
        # A longer RGB-visible stop needs its additional measured intervals;
        # otherwise a physically settling endpoint can time out one sample
        # before the requested64-sample window is complete. Motion is untouched.
        # The baseline100-step budget includes a20-sample stop window.
        # Increasing the scoring window itself needs the same extension as
        # increasing the expert-only window.
        stop_samples=self.strategy.get('expert_stop_minimum_samples',self.strategy.get('stop_visit',{}).get('minimum_samples',0))
        extra_stop=max(0,stop_samples-20)
        pass_through = getattr(self, 'route_pass_through', False)
        settling_limit = 0 if pass_through else ((100+extra_stop+9)//10)*10
        while True:
            drive = np.array([j.get_drive_target()[0] for j in self.robot.left_entity.get_active_joints()])
            driven = self.recorder.kin.pose(command(drive))
            measured = np.asarray(self.object_state()['hand_pose'])
            error = float(np.linalg.norm(driven[:3,3]-measured[:3,3]))
            if pass_through:
                break
            stopped=True
            if self.strategy.get('sequence_rule')=='ordered_region_stops':
                criteria=self.strategy['stop_visit']
                samples=self.strategy.get('expert_stop_minimum_samples',criteria['minimum_samples'])
                window=round(STOP_SPEED_WINDOW_S/.004)
                recent=np.asarray([self.route_reference_position(row) for row in self.recorder.rows[-samples-window:]])
                stopped=bool(len(recent)==samples+window and np.all(route_reference_speed(recent)[-samples:]<=criteria['maximum_speed_m_s']))
                if stopped and 'expert_stop_minimum_samples' in self.strategy:
                    region=next(r for r in self.strategy['regions'] if r['id']==name)
                    if region.get('shape')!='sphere':
                        raise ValueError('Measured expert-stop extension requires spherical regions')
                    stopped=bool(np.all(np.linalg.norm(recent[-samples:]-region['center_m'],axis=1)<=region['radius_m']+1e-6))
            measured_chaining = getattr(self, 'measured_segment_chaining', False)
            if stopped and (measured_chaining or error <= .003):
                break
            if settling>=settling_limit:
                if not stopped:raise RuntimeError('Region did not settle within native endpoint hold')
                break
            self.hold_region(10)
            settling += 10
        self.route_execution.append(dict(waypoint=name, begin_step=begin, end_step=self.physics_step,
            arrival_step=arrival_step, requested_dwell_steps=dwell,
            requested_reference_m=list(xyz),
            expert_stop_minimum_samples=self.strategy.get('expert_stop_minimum_samples'),
            stop_speed_measurement_window_s=STOP_SPEED_WINDOW_S,
            endpoint_settle_steps=settling, endpoint_settle_limit_steps=settling_limit, endpoint_tracking_error_m=error,
            reference_point=self.strategy.get('reference_point', 'object_origin'),
            measured_reference_m=self.route_reference_position(self.object_state()).tolist()))
        self._last_region = name

    def hold_region(self,steps):
        q=np.asarray([joint.get_drive_target()[0] for joint in self.robot.left_entity.get_active_joints()])[:7]
        arm=dict(status='Success',position=np.repeat(q[None],steps,axis=0).astype(np.float32),
            velocity=np.zeros((steps,7),np.float32),native_region_hold=True)
        self.take_dense_action(dict(left_arm=arm,right_arm=None,right_gripper=None,
            left_gripper=dict(result=np.full(steps,self.robot.get_left_gripper_val()),
                num_step=steps,per_step=0.,open_route_dwell=True)))

    def close_env(self, clear_cache=False):
        if hasattr(self, 'cameras'):
            self.head_calibration = self.cameras.get_config()['head_camera']
        return super().close_env(clear_cache)
