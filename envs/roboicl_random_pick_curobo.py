"""RoboICL native cuRobo V2 expert with geometry from the actual scene.

RoboTwin joint/gripper execution remains unchanged. The task supplies scene
geometry and actual inertias, and retains planners across scene resets.
"""
from pathlib import Path
from contextlib import contextmanager
from copy import copy, deepcopy

import numpy as np
import transforms3d as t3d
from .robot.robot import Robot
from .robot.roboicl_v2_planner import SceneCuroboPlanner
from .roboicl_random_pick import roboicl_random_pick, SPAWN_ADMISSION



class roboicl_random_pick_curobo(roboicl_random_pick):
    def load_robot(self, **kwargs):
        self.cartesian_pick_paths = bool(kwargs.get('cartesian_pick_paths', False))
        self.direct_pick_approach = bool(kwargs.get('direct_pick_approach', False))
        seeds = kwargs.get('curobo_num_trajopt_seeds', 1)
        if type(seeds) is not int or seeds < 1:
            raise ValueError('cuRobo trajectory seed count must be a positive integer')
        # cuRobo already supplies timed position/velocity trajectories. This
        # separate task does not create the unused MPLib TOPP helper.
        self.need_topp = False
        reused = hasattr(self, 'robot')
        if not reused:
            self.robot = Robot(self.scene, False, **kwargs)
            self.robot.communication_flag = False
            for side in ('left', 'right'):
                entity = getattr(self.robot, side+'_entity')
                planner = SceneCuroboPlanner(entity.get_root_pose(),
                    getattr(self.robot, side+'_arm_joints_name'),
                    [joint.name for joint in entity.get_active_joints()],
                    getattr(self.robot, side+'_curobo_yml_path'), self.table,
                    getattr(self,'planner_object_capacity',len(self.object_specs))+getattr(self,'planner_extra_static_boxes',0),
                    num_trajopt_seeds=seeds)
                planner.entity = entity
                planner.task = self
                setattr(self.robot, side+'_planner', planner)
            self.robot.init_joints()
        else:
            self.robot._init_robot_(self.scene, False, **kwargs)
            self.robot.init_joints()
            for side in ('left', 'right'):
                planner = getattr(self.robot, side+'_planner')
                if not isinstance(planner, SceneCuroboPlanner):
                    raise ValueError('Native collection cannot reuse another planner backend')
                if planner.num_trajopt_seeds != seeds:
                    raise ValueError('A changed cuRobo seed count requires a fresh task process')
                planner.entity = getattr(self.robot, side+'_entity')
                planner.task = self
                planner.update_scene(self.table, planner.entity.get_root_pose())
        for entity in (self.robot.left_entity, self.robot.right_entity):
            for link in entity.get_links():
                link.set_mass(1)
        self.curobo_reused = reused

    @contextmanager
    def cartesian_motion(self, arm_tag, stage):
        if stage not in ('approach', 'lift', 'direct', 'place', 'place_descent', 'retreat'):
            raise ValueError('Unknown Cartesian pick stage')
        planner = getattr(self.robot, str(arm_tag)+'_planner')
        previous = planner.cartesian_segment
        planner.cartesian_segment = stage if self.cartesian_pick_paths else None
        try:
            yield
        finally:
            planner.cartesian_segment = previous

    def set_pregrasp_obstacles(self, enabled):
        boxes, bounds = {}, []
        if enabled:
            for index,(actor,vertices) in enumerate(zip(self.objects,self.object_collision_vertices)):
                pose = actor.get_pose().to_transformation_matrix()
                points = vertices @ pose[:3,:3].T + pose[:3,3]
                lower,upper = points.min(axis=0),points.max(axis=0)
                transform = np.eye(4)
                transform[:3,3] = (lower+upper)/2
                boxes[f'pregrasp_object_{index}'] = dict(dims=(upper-lower).tolist(),transform=transform)
                bounds.append(dict(object_index=index,world_min=lower.tolist(),world_max=upper.tolist()))
        worlds = {side:getattr(self.robot,side+'_planner').update_pregrasp_world(boxes) for side in ('left','right')}
        return dict(physics_step=self.physics_step,enabled=enabled,
            source='World AABB of all actual SAPIEN object collision vertices at switch time',
            object_bounds=bounds,worlds=worlds)

    def planner_gripper_spheres_in_hand(self):
        planner = self.robot.left_planner
        state = planner.kinematic_state(self.robot.left_entity.get_qpos()[:7])
        pose = state.tool_poses.get_link_pose('panda_hand')
        transform = np.eye(4)
        transform[:3,3] = pose.position.detach().cpu().numpy().reshape(-1,3)[0]
        transform[:3,:3] = t3d.quaternions.quat2mat(pose.quaternion.detach().cpu().numpy().reshape(-1,4)[0])
        inverse = np.linalg.inv(transform)
        indices = planner.sphere_indices(('panda_hand','panda_leftfinger','panda_rightfinger'))
        all_spheres = state.robot_spheres.reshape(-1,state.robot_spheres.shape[-2],4)
        spheres = all_spheres[0,indices].detach().cpu().numpy().astype(float)
        spheres[:,:3] = spheres[:,:3] @ inverse[:3,:3].T + inverse[:3,3]
        return spheres

    def clearance_grasp_actor(self, actor, arm_tag, mesh_margin=.003, sphere_margin=.001, palm_margin=.001,
                              contact_point_ids=None):
        """Validate native pregrasp and approach before committing to a contact."""
        candidates = ([None]+[index for index,_ in actor.iter_contact_points()]
                      if contact_point_ids is None else list(contact_point_ids))
        if not candidates:
            raise ValueError('At least one native grasp candidate is required')
        self.qualified_direct_path = None
        fallback = None
        attempts = []
        plan = self.robot.left_plan_path if arm_tag == 'left' else self.robot.right_plan_path
        for contact_id in candidates:
            self.set_pregrasp_obstacles(True)
            attempt = dict(contact_id=contact_id,selection='native_preferred' if contact_id is None else 'native_contact_fallback')
            attempts.append(attempt)
            try:
                result = super().grasp_actor(actor,arm_tag,pre_grasp_dis=.1,grasp_dis=0.,contact_point_id=contact_id)
                if (self.direct_pick_approach and getattr(self,'continuous_pick_place',False)
                        and not getattr(self, 'official_robotwin_control', False)):
                    result, direct, symmetry = self._plan_symmetric_grasp(
                        result,actor,arm_tag,mesh_margin,sphere_margin,palm_margin)
                    attempt['symmetry_candidates'] = symmetry
                    if direct is None:
                        attempt['rejected_stage'] = 'both_jaw_orientations'
                        continue
                    attempt['clearance'] = self.grasp_clearance_audit.copy()
                    self.grasp_clearance_audit.update(candidate_count=len(attempts),selected_contact_id=contact_id,
                        candidate_attempts=attempts,execution_approach='continuous')
                    self.qualified_direct_path = direct
                    return result
                result = self._apply_grasp_clearance(result,actor,mesh_margin,sphere_margin,palm_margin)
                moves = [action for action in result[1] if action.action == 'move']
                attempt['clearance'] = self.grasp_clearance_audit.copy()
                if self.direct_pick_approach:
                    self.set_pregrasp_obstacles(False)
                    with self.cartesian_motion(arm_tag, 'direct'):
                        direct = plan(moves[-1].target_pose)
                    attempt['direct_status'] = direct['status']
                    if direct['status'] == 'Success':
                        self.grasp_clearance_audit.update(candidate_count=len(attempts),selected_contact_id=contact_id,
                            candidate_attempts=attempts,execution_approach='continuous')
                        self.qualified_direct_path = direct
                        return result
                    attempt['direct_rejection'] = direct.get('cartesian_path')
                    if getattr(self,'continuous_pick_place',False):
                        continue
                    self.set_pregrasp_obstacles(True)
                pre_path = plan(moves[0].target_pose)
                attempt['pregrasp_status'] = pre_path['status']
                if pre_path['status'] != 'Success':
                    attempt['rejected_stage'] = 'pregrasp_with_object_obstacles'
                    continue
                self.set_pregrasp_obstacles(False)
                with self.cartesian_motion(arm_tag, 'approach'):
                    grasp_path = plan(moves[-1].target_pose,constraint_pose=moves[-1].args.get('constraint_pose'),
                        last_qpos=pre_path['position'][-1])
                attempt['grasp_status'] = grasp_path['status']
                if grasp_path['status'] != 'Success':
                    attempt['rejected_stage'] = 'grasp_from_pregrasp_terminal_joints'
                    continue
                if self.direct_pick_approach:
                    if fallback is None:
                        fallback = (result,pre_path,grasp_path,self.grasp_clearance_audit.copy(),contact_id)
                    continue
                self.grasp_clearance_audit.update(candidate_count=len(attempts),selected_contact_id=contact_id,
                    candidate_attempts=attempts,execution_approach='pregrasp_then_straight_approach')
                self.qualified_pregrasp_path = pre_path
                self.qualified_grasp_path = grasp_path
                self.set_pregrasp_obstacles(True)
                return result
            except (RuntimeError,TypeError,AssertionError) as error:
                if isinstance(error,AssertionError) and str(error)!='target_pose cannot be None for move action.':
                    raise
                attempt.update(rejected_stage='native_candidate_or_clearance',error=str(error))
        if fallback is not None and not getattr(self,'continuous_pick_place',False):
            result,self.qualified_pregrasp_path,self.qualified_grasp_path,clearance,contact_id = fallback
            self.grasp_clearance_audit = dict(clearance,candidate_count=len(attempts),selected_contact_id=contact_id,
                candidate_attempts=attempts,execution_approach='pregrasp_then_straight_approach')
            self.set_pregrasp_obstacles(True)
            return result
        self.set_pregrasp_obstacles(True)
        self.grasp_clearance_audit = dict(candidate_count=len(attempts),candidate_attempts=attempts)
        raise RuntimeError('No native contact has feasible pregrasp and clearance-adjusted approach')

    def _plan_symmetric_grasp(self, result, actor, arm_tag, mesh_margin, sphere_margin, palm_margin):
        """Compare two finger-swapped endpoints with the same full collision checks."""
        plan = self.robot.left_plan_path if arm_tag == 'left' else self.robot.right_plan_path
        feasible, attempts = [], []
        for flipped in (False, True):
            candidate = (result[0],[copy(action) for action in result[1]])
            for action in candidate[1]:
                action.target_pose = deepcopy(action.target_pose)
                action.args = deepcopy(action.args)
            attempt = dict(fingers_exchanged=flipped)
            attempts.append(attempt)
            if flipped:
                # The official grasp frame approaches along local X. A half
                # turn about X exchanges the jaws without moving the contact.
                for action in candidate[1]:
                    if action.action == 'move':
                        pose = action.target_pose
                        rotation = t3d.quaternions.quat2mat(pose[3:]) @ np.diag([1.,-1.,-1.])
                        action.target_pose = list(pose[:3])+t3d.quaternions.mat2quat(rotation).tolist()
            try:
                candidate = self._apply_grasp_clearance(candidate,actor,mesh_margin,sphere_margin,palm_margin)
                clearance = deepcopy(self.grasp_clearance_audit)
                self.set_pregrasp_obstacles(False)
                moves = [action for action in candidate[1] if action.action == 'move']
                with self.cartesian_motion(arm_tag,'direct'):
                    path = plan(moves[-1].target_pose)
                attempt.update(status=path['status'],path_audit=path.get('cartesian_path'))
                if path['status'] == 'Success':
                    length = float(np.linalg.norm(np.diff(np.asarray(path['position']),axis=0),axis=1).sum())
                    attempt['joint_path_length_rad'] = length
                    feasible.append((length,candidate,path,clearance,flipped))
            except RuntimeError as error:
                attempt.update(status='Fail',error=str(error))
        if not feasible:
            return None,None,attempts
        prefer_native = getattr(self, 'prefer_native_jaw_orientation', False)
        _,candidate,path,clearance,flipped = min(
            feasible,key=lambda item: (item[4],item[0]) if prefer_native else item[0])
        self.grasp_clearance_audit = dict(clearance,fingers_exchanged=flipped,
            grasp_selection=('prefer_native_jaw_orientation_then_shortest_feasible_joint_path'
                             if prefer_native else 'shortest_feasible_joint_path_among_two_jaw_orientations'))
        return candidate,path,attempts

    def _apply_grasp_clearance(self, result, actor, mesh_margin, sphere_margin, palm_margin):
        moves = [action for action in result[1] if action.action == 'move']
        if len(moves) != 2:
            raise RuntimeError('Expected native pregrasp and grasp actions')
        high = np.eye(4)
        high[:3,3] = moves[-1].target_pose[:3]
        high[:3,:3] = t3d.quaternions.quat2mat(moves[-1].target_pose[3:])
        current = np.eye(4)
        pose = self.robot.get_left_ee_pose()
        current[:3,3], current[:3,:3] = pose[:3], t3d.quaternions.quat2mat(pose[3:])
        measured = self.robot.left_entity.find_link_by_name('panda_hand').entity_pose.to_transformation_matrix()
        high_to_hand = np.linalg.inv(current) @ measured
        hand = high @ high_to_hand
        approach_z = float(high[2,0])
        spheres = self.planner_gripper_spheres_in_hand()
        mesh_z = self.gripper_bottom_at_hand_pose(hand)
        sphere_z = float(np.min(spheres[:,:3] @ hand[2,:3]+hand[2,3]-spheres[:,3]))
        table_z = SPAWN_ADMISSION['table_height_m']
        missing_height = max(0.,table_z+mesh_margin-mesh_z,table_z+sphere_margin-sphere_z)
        if missing_height > 0 and approach_z >= -1e-9:
            raise RuntimeError('Native grasp axial retreat cannot satisfy table clearance')
        table_retreat = missing_height/(-approach_z) if missing_height else 0.
        palm_vertices = self._collision_vertices(self.robot.left_entity.find_link_by_name('panda_hand'))
        actor_index = self.objects.index(actor)
        actor_pose = actor.get_pose().to_transformation_matrix()
        actor_world = self.object_collision_vertices[actor_index] @ actor_pose[:3,:3].T+actor_pose[:3,3]
        inverse_hand = np.linalg.inv(hand)
        actor_in_hand = actor_world @ inverse_hand[:3,:3].T+inverse_hand[:3,3]
        axial_gain = float((inverse_hand[:3,:3] @ high[:3,0])[2])
        if axial_gain <= 0:
            raise RuntimeError('Native grasp frame retreat does not separate the palm from the target')
        palm_front = float(palm_vertices[:,2].max())
        object_near = float(actor_in_hand[:,2].min())
        palm_retreat = max(0.,(palm_front+palm_margin-object_near)/axial_gain)
        projected_palm_retreat = palm_retreat
        physical_palm_clearance = None
        if getattr(self,'continuous_pick_place',False):
            retreat,physical_palm_clearance = self.robot.left_planner.target_mesh_sweep.grasp_retreat(
                hand,high[:3,0],table_retreat,palm_margin)
            palm_retreat = retreat if retreat > table_retreat+1e-9 else 0.
        else:
            retreat = max(table_retreat,palm_retreat)
        if retreat >= .1:
            raise RuntimeError('Required grasp retreat exceeds native pregrasp distance')
        original = list(moves[-1].target_pose)
        high[:3,3] -= retreat*high[:3,0]
        moves[-1].target_pose = high[:3,3].tolist()+t3d.quaternions.mat2quat(high[:3,:3]).tolist()
        self.grasp_clearance_audit = dict(grasp_dis_m=retreat,original_high_pose=original,
            approach_z=approach_z,missing_height_m=missing_height,
            table_retreat_m=table_retreat,palm_retreat_m=palm_retreat,palm_object_margin_m=palm_margin,
            palm_clearance_method='physical_convex_parts' if physical_palm_clearance is not None else 'axial_projection',
            physical_palm_clearance_m=physical_palm_clearance,projected_palm_retreat_m=projected_palm_retreat,
            original_object_near_hand_z_m=object_near,palm_front_hand_z_m=palm_front,axial_gain=axial_gain,
            palm_collision_vertices_in_hand=palm_vertices.tolist(),object_collision_vertices_in_original_hand=actor_in_hand.tolist(),
            object_collision_vertices_in_actor=self.object_collision_vertices[actor_index].tolist(),
            object_actor_to_world=actor_pose.tolist(),object_index=actor_index,
            corrected_high_pose=moves[-1].target_pose,high_to_hand=high_to_hand.tolist(),
            original_mesh_bottom_z_m=mesh_z,original_buffered_sphere_bottom_z_m=sphere_z,
            mesh_margin_m=mesh_margin,sphere_margin_m=sphere_margin,
            hand_collision_envelope=self.gripper_collision_envelope.tolist(),
            buffered_planner_gripper_spheres_in_hand=spheres.tolist())
        return result
