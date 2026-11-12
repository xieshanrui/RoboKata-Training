"""RoboICL scene adapter for native cuRobo V2 planning, timing and dynamics.

The planner supplies dense position/velocity commands. RoboTwin still executes
them with its existing SAPIEN drives; no torque feedforward is applied.
"""
from pathlib import Path
from copy import copy
import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np
import torch
import transforms3d as t3d
import yaml
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState, ToolPoseCriteria, Pose
from curobo._src.geom.types import SceneCfg, Mesh
from curobo._src.geom.sphere_fit.types import SphereFitType
from curobo._src.state.state_joint_trajectory_ops import trim_joint_state_trajectory


def table_world(table, origin):
    inverse = np.linalg.inv(origin.to_transformation_matrix())
    bodies = [c for c in table.components if hasattr(c, 'collision_shapes')]
    if len(bodies) != 1:
        raise ValueError('Expected one physical table body')
    boxes = {}
    for index, shape in enumerate(bodies[0].collision_shapes):
        if not hasattr(shape, 'half_size'):
            raise ValueError('Expected physical table boxes')
        matrix = inverse @ table.pose.to_transformation_matrix() @ shape.local_pose.to_transformation_matrix()
        u, _, vh = np.linalg.svd(matrix[:3,:3])
        boxes[f'table_{index}'] = dict(dims=(2*np.asarray(shape.half_size)).tolist(),
            pose=matrix[:3,3].tolist()+t3d.quaternions.mat2quat(u@vh).tolist())
    return dict(cuboid=boxes)


def body_inertia(body, transform=None):
    cm = body.cmass_local_pose.to_transformation_matrix()
    if transform is not None:
        cm = transform @ cm
    return float(body.mass), cm[:3,3], cm[:3,:3]@np.diag(body.inertia)@cm[:3,:3].T


def combine_inertia(parts):
    mass = sum(p[0] for p in parts)
    com = sum(m*c for m,c,_ in parts)/mass if mass else np.zeros(3)
    matrix = sum(i+m*((c-com)@(c-com)*np.eye(3)-np.outer(c-com,c-com)) for m,c,i in parts)
    return dict(mass=mass, com=torch.as_tensor(com,device='cuda'),
        inertia=torch.as_tensor(matrix[[0,1,2,0,0,1],[0,1,2,1,2,2]],device='cuda'))


class SceneCuroboPlanner:
    def __init__(self, origin, active_names, all_names, yml_path, table, object_count, num_trajopt_seeds=4):
        self.robot_origion_pose = origin
        self.active_joints_name, self.all_joints = active_names, all_names
        self.yml_path = str(Path(yml_path).resolve())
        self.num_trajopt_seeds = num_trajopt_seeds
        self.cartesian_segment = None
        self.task = self.entity = None
        config = yaml.safe_load(Path(self.yml_path).read_text(encoding='utf-8'))
        if not np.allclose(config['planner']['frame_bias'],0,atol=0,rtol=0):
            raise ValueError('The task requires the frozen zero-bias Franka frame')
        robot = config['robot_cfg']; k = robot['kinematics']
        k.pop('use_usd_kinematics',None)
        k['tool_frames'] = [k.pop('ee_link')]
        k['asset_root_path'] = str(Path(k['urdf_path']).parent)
        k['cspace']['default_joint_position'] = k['cspace'].pop('retract_config')
        # Leave acceleration headroom for native torque feasibility after retiming.
        # Velocity limits and the returned native trajectory remain unchanged.
        k['cspace']['acceleration_scale'] = .9
        k['format_version'] = 2.0
        # Native attachment geometry participates in every carried plan.
        k['extra_collision_spheres']['attached_object'] = 64
        if isinstance(k['collision_spheres'],str):
            k['collision_spheres'] = yaml.safe_load(Path(k['collision_spheres']).read_text(encoding='utf-8'))['collision_spheres']
        robot['load_dynamics'] = True
        self.robot_config = robot
        self.fixed_joints = {j.find('child').attrib['link']:j for j in ET.parse(k['urdf_path']).getroot().findall('joint')}
        self.scene_world = table_world(table,origin)
        cfg = MotionPlannerCfg.create(robot=robot,scene_model=self.scene_world,
            collision_cache=dict(cuboid=len(self.scene_world['cuboid'])+object_count),
            num_trajopt_seeds=num_trajopt_seeds,interpolation_dt=.004,
            optimizer_collision_activation_distance=.03,
            interpolation_buffer_size=4096,random_seed=123)
        self.planner = MotionPlanner(cfg)
        self.kinematics_config = cfg.trajopt_solver_config.robot_config.kinematics.kinematics_config
        self.limits = cfg.trajopt_solver_config.robot_config.kinematics.get_joint_limits()
        self.planner.warmup(enable_graph=True,num_warmup_iterations=1)
        self.active_world = self.scene_world
        self.payload_spheres = {}
        self.payload_attached = False
        self.attachment_events = []

    def save_evidence(self, name, value):
        output = getattr(self.task, 'output', None)
        if output is not None:
            (Path(output)/name).write_text(json.dumps(value, indent=2), encoding='utf-8')

    def sync_attachment(self, start):
        """Use measured object pose, native cached sphere fitting and attachment updates."""
        carried = (self.entity is self.task.robot.left_entity
                   and getattr(self.task, 'carried_vertices_in_hand', None) is not None)
        if carried:
            parts = self.task.object_collision_parts[self.task.target_index]
            key = hashlib.sha256(json.dumps(parts, sort_keys=True).encode('utf-8')).hexdigest()
            manager = self.planner.attachment_manager
            if key not in self.payload_spheres:
                import trimesh
                meshes = []
                for index, part in enumerate(parts):
                    hull = trimesh.convex.convex_hull(np.asarray(part['vertices_in_actor']))
                    meshes.append(Mesh(name=f'payload_{index}', pose=[0,0,0,1,0,0,0],
                        vertices=hull.vertices.tolist(), faces=hull.faces.tolist()))
                # Inscribed voxel spheres avoid inventing a protrusion through
                # the supporting surface. The existing mesh sweep still runs.
                self.payload_spheres[key] = manager.fit_spheres(meshes, num_spheres=64,
                    sphere_fit_type=SphereFitType.VOXEL)
            spheres = self.payload_spheres[key]
            transform = (np.linalg.inv(self.robot_origion_pose.to_transformation_matrix())
                         @ self.task.target.get_pose().to_transformation_matrix())
            support_contact = None
            if self.cartesian_segment == 'lift':
                # Resting PhysX contacts can penetrate by a few micrometres.
                # Permit only that measured support-contact boundary during
                # departure. Cached geometry and subsequent carried plans
                # retain their original radii; the physical mesh sweep remains.
                tolerance = 1e-5
                table = self.scene_world['cuboid']['table_0']
                rotation = t3d.quaternions.quat2mat(table['pose'][3:])
                center = np.asarray(table['pose'][:3])
                half = np.asarray(table['dims']) / 2
                raw = spheres.detach().cpu().numpy()
                local = (raw[:, :3] @ transform[:3, :3].T + transform[:3, 3] - center) @ rotation
                clearance = float(np.min(local[:, 2] - raw[:, 3] - half[2]))
                vertices = np.concatenate([p['vertices_in_actor'] for p in parts])
                mesh = (vertices @ transform[:3, :3].T + transform[:3, 3] - center) @ rotation
                mesh_clearance = float(np.min(mesh[:, 2] - half[2]))
                on_support = bool(np.all(np.abs(mesh[:, :2]) <= half[:2]))
                if (on_support and -tolerance <= clearance < 0
                        and -tolerance <= mesh_clearance <= tolerance):
                    spheres = spheres.clone()
                    spheres[:, 3] = torch.clamp(spheres[:, 3] - tolerance, min=0.)
                    support_contact = dict(tolerance_m=tolerance,
                        original_sphere_clearance_m=clearance,
                        actual_mesh_clearance_m=mesh_clearance,
                        scope='lift payload radii only; original radii restored at next plan')
            pose = Pose.from_list(transform[:3,3].tolist()+t3d.quaternions.mat2quat(transform[:3,:3]).tolist())
            manager.update(spheres, start, link_name='attached_object', world_objects_pose_offset=pose)
            self.payload_attached = True
            event = dict(action='update', physics_step=self.task.physics_step,
                geometry_sha256=key, spheres=spheres.detach().cpu().numpy().tolist(),
                object_pose_in_robot_base=transform.tolist(), start_joints=start.position.detach().cpu().numpy().tolist(),
                method='native AttachmentManager.fit_spheres(VOXEL) + update',
                target_world_obstacle=False, support_contact=support_contact)
            self.attachment_events.append(event)
            self.save_evidence('v2_attachments.json', self.attachment_events)
            return dict(attached=True, spheres=len(spheres), geometry_sha256=key)
        if self.payload_attached:
            self.planner.attachment_manager.detach()
            self.payload_attached = False
            # Keep only the released target exempt during contact withdrawal;
            # the physical mesh sweep checks it instead of buffered spheres.
            contact_exit = self.cartesian_segment == 'retreat'
            world = self.task.set_pregrasp_obstacles(not contact_exit)
            self.attachment_events.append(dict(action='detach', physics_step=self.task.physics_step,
                released_world=world, contact_exit_pending=contact_exit))
            self.save_evidence('v2_attachments.json', self.attachment_events)
        return dict(attached=False)

    def sphere_indices(self, names):
        return torch.cat([self.kinematics_config.get_sphere_index_from_link_name(n) for n in names])

    def kinematic_state(self, positions):
        q = torch.as_tensor(positions,dtype=torch.float32,device='cuda').reshape(-1,7)
        return self.planner.compute_kinematics(JointState.from_position(q,joint_names=self.active_joints_name))

    def sync_dynamics(self):
        """Match actual simulated link inertia; add a rigid held payload only after closure."""
        components = {}; merged = {}
        for link in self.entity.get_links():
            name = link.name; frame = name; transform = np.eye(4)
            while frame not in self.kinematics_config.link_name_to_idx_map:
                joint = self.fixed_joints[frame]
                if joint.attrib['type'] != 'fixed':
                    raise ValueError('Unrepresented moving link: '+frame)
                origin = joint.find('origin'); local = np.eye(4)
                if origin is not None:
                    local[:3,3] = np.fromstring(origin.attrib.get('xyz','0 0 0'),sep=' ')
                    local[:3,:3] = t3d.euler.euler2mat(*np.fromstring(origin.attrib.get('rpy','0 0 0'),sep=' '))
                transform = local@transform; frame = joint.find('parent').attrib['link']
            components.setdefault(frame,[]).append(body_inertia(link,transform))
            if frame != name: merged[name] = dict(ancestor=frame,mass_kg=float(link.mass))
        payload_mass = 0.
        if self.entity is self.task.robot.left_entity and getattr(self.task,'carried_vertices_in_hand',None) is not None:
            hand = self.entity.find_link_by_name('panda_hand').entity_pose.to_transformation_matrix()
            attachment = np.linalg.inv(hand)@self.task.target.get_pose().to_transformation_matrix()
            payload = body_inertia(self.task.object_bodies[self.task.target_index],attachment)
            payload_mass = payload[0]; components['panda_hand'].append(payload)
        self.planner.update_links_inertial({name:combine_inertia(parts) for name,parts in components.items()})
        return dict(enabled=True,source='actual SAPIEN inertia; fixed attachments merged by URDF transform',
            merged_fixed_links=merged,payload_mass_kg=payload_mass,torque_feedforward=False)

    def sync_gripper(self):
        """Update native locked-finger transforms to the measured aperture."""
        # KinematicsParams.clone() reapplies cspace scaling in V2's constructor.
        # Copy only the fields being changed so joint limits stay fixed.
        params = copy(self.kinematics_config)
        params.fixed_transforms = params.fixed_transforms.clone()
        params.lock_jointstate = params.lock_jointstate.clone()
        aperture = {}
        for name in ('panda_leftfinger', 'panda_rightfinger'):
            joint = self.fixed_joints[name]
            parent = self.entity.find_link_by_name(joint.find('parent').attrib['link'])
            child = self.entity.find_link_by_name(name)
            local = (np.linalg.inv(parent.entity_pose.to_transformation_matrix())
                     @ child.entity_pose.to_transformation_matrix())
            index = params.link_name_to_idx_map[name]
            params.fixed_transforms[index].copy_(torch.as_tensor(
                local[:3], device=params.fixed_transforms.device,
                dtype=params.fixed_transforms.dtype))
            joint_name = joint.attrib['name']
            value = float(self.entity.get_qpos()[self.all_joints.index(joint_name)])
            aperture[joint_name] = value
            params.lock_jointstate.position[..., params.lock_jointstate.joint_names.index(joint_name)] = value
        self.planner.kinematics.update_kinematics_config(params)
        return dict(method='native Kinematics.update_kinematics_config', measured_locked_joints=aperture,
                    collision_sphere_radii_unchanged=True)

    def update_scene(self, table, origin):
        self.planner.attachment_manager.detach()
        self.payload_attached = False
        self.attachment_events = []
        self.robot_origion_pose = origin; self.scene_world = table_world(table,origin)
        self.active_world = self.scene_world
        self.planner.update_world(SceneCfg.create(self.scene_world)); self.planner.reset_seed()

    def update_pregrasp_world(self, world_boxes):
        world = dict(cuboid=dict(self.scene_world['cuboid']))
        inverse = np.linalg.inv(self.robot_origion_pose.to_transformation_matrix())
        for name, box in world_boxes.items():
            matrix = inverse@box['transform']; u,_,vh = np.linalg.svd(matrix[:3,:3])
            world['cuboid'][name] = dict(dims=box['dims'],pose=matrix[:3,3].tolist()+t3d.quaternions.mat2quat(u@vh).tolist())
        if world_boxes:
            self.direct_obstacle_boxes = {n:b for n,b in world['cuboid'].items() if n not in self.scene_world['cuboid']}
        self.planner.update_world(SceneCfg.create(world)); self.active_world = world
        return world

    def plan_grasp(self, goal, start, dynamics):
        """Official approach + grasp, preserving both native trajectories and their timing."""
        links = ['panda_hand', 'panda_leftfinger', 'panda_rightfinger']
        self.save_evidence('official_grasp_world.json', self.task.set_pregrasp_obstacles(True))
        try:
            result = self.planner.plan_grasp(goal, start, grasp_approach_axis='z',
                grasp_approach_offset=-.1, grasp_approach_in_tool_frame=True,
                plan_approach_to_grasp=True, plan_grasp_to_lift=False, disable_collision_links=links)
        finally:
            self.planner.enable_link_collision(links)
            self.planner.update_tool_pose_criteria({'panda_hand':ToolPoseCriteria()})
            self.task.set_pregrasp_obstacles(False)
        audit = dict(stage='direct', planner_engine='cuRoboV2', planner_method='plan_grasp',
            dynamics=dynamics, time_dilation_factor=1., interpolation_dt_s=.004,
            shape_diagnostics=dict(exceeded=[], enforced=False), native_interface_status=result.status,
            approach_offset_m=-.1, target_in_planning_world=True, planned_lift=False,
            official_collision_disabled_links=links,
            collision_disable_scope='Globally disabled contact links during native goalset and final grasp; full collision checking during approach')
        if result.success is None or not bool(result.success.all()):
            self.save_evidence('official_grasp_plan.json', dict(audit, accepted=False))
            return dict(status='Fail', cartesian_path=dict(audit, rejected_reason='native_plan_grasp'))
        arrays, phases, offset = [], [], 0
        for name in ('approach', 'grasp'):
            trajectory = getattr(result, name+'_interpolated_trajectory')
            end = getattr(result, name+'_interpolated_last_tstep')
            if end is not None:
                trajectory = trim_joint_state_trajectory(trajectory, 0, end.reshape(-1)[0])
            trajectory = self.planner.kinematics.get_active_js(trajectory)
            values = {k:getattr(trajectory,k).detach().cpu().numpy().reshape(-1,7)
                      for k in ('position','velocity','acceleration','jerk')}
            phases.append(dict(name=name, start_sample=offset, samples=len(values['position']),
                duration_s=(len(values['position'])-1)*.004,
                position_sha256=hashlib.sha256(values['position'].tobytes()).hexdigest(),
                velocity_sha256=hashlib.sha256(values['velocity'].tobytes()).hexdigest(),
                terminal_velocity_max=float(abs(values['velocity'][-1]).max()),
                terminal_acceleration_max=float(abs(values['acceleration'][-1]).max())))
            offset += len(values['position'])
            arrays.append(values)
        combined = {k:np.concatenate([a[k] for a in arrays]) for k in arrays[0]}
        sweep = self.target_mesh_sweep(self.kinematic_state(combined['position']))
        clearance = float(getattr(self,'native_contact_path_clearance_m',None) or .001)
        accepted = bool(sweep['accepted'] and min(sweep['minimum_gripper_clearance_m'].values()) >= clearance)
        audit.update(native_constraints_feasible=True, native_phases=phases,
            concatenation='Unmodified native arrays in order, boundary retained; no blending or extra hold',
            boundary_joint_position_gap_rad=float(abs(arrays[0]['position'][-1]-arrays[1]['position'][0]).max()),
            terminal_joint_velocity_max=float(abs(combined['velocity'][-1]).max()),
            terminal_joint_acceleration_max=float(abs(combined['acceleration'][-1]).max()),
            target_mesh_sweep=sweep, required_contact_path_clearance_m=clearance)
        self.last_trajectory = None
        self.save_evidence('official_grasp_plan.json', dict(audit, accepted=accepted))
        if not accepted:
            return dict(status='Fail', cartesian_path=dict(audit, rejected_reason='physical_target_sweep'))
        return dict(status='Success', **combined, cartesian_path=audit)

    def plan_path(self, curr_joint_pos, target_gripper_pose, constraint_pose=None, arms_tag=None):
        base = self.robot_origion_pose.to_transformation_matrix()
        target = np.linalg.inv(base)@target_gripper_pose.to_transformation_matrix()
        position = torch.as_tensor(target[:3,3],dtype=torch.float32,device='cuda').reshape(1,1,1,1,3)
        quaternion = torch.as_tensor(t3d.quaternions.mat2quat(target[:3,:3]),dtype=torch.float32,device='cuda').reshape(1,1,1,1,4)
        goal = GoalToolPose(tool_frames=['panda_hand'],position=position,quaternion=quaternion)
        indices = [self.all_joints.index(n) for n in self.active_joints_name]
        q = np.round(np.asarray(curr_joint_pos)[indices],5)
        start = JointState.from_position(torch.as_tensor(q,dtype=torch.float32,device='cuda').reshape(1,7),joint_names=self.active_joints_name)
        criteria = ToolPoseCriteria()
        stage = self.cartesian_segment
        ee_waypoint = (stage == 'place' and getattr(self.task, '_route_target', None) is not None
                       and self.task.strategy.get('reference_point') == 'panda_hand')
        if ee_waypoint:
            # EE waypoints specify position only. Let native IK/trajectory
            # optimization choose a feasible orientation for the held geometry.
            criteria = ToolPoseCriteria(terminal_pose_axes_weight_factor=[1,1,1,0,0,0])
        elif stage in ('lift','approach','retreat'):
            # Wide objects can leave only millimetres of clearance after release.
            # Keep native planning and the physical sweep; request tighter tracking
            # for an explicitly configured retreat, without changing other phases.
            scale = getattr(self.task, 'retreat_linear_scale', 1.) if stage == 'retreat' else 1.
            criteria = ToolPoseCriteria.linear_motion(axis='z', non_terminal_scale=scale,
                project_distance_to_goal=stage in ('approach','retreat'))
        elif stage == 'place_descent':
            # Criteria use the robot base frame; this task's base Z is world Z.
            if not np.allclose(base[:3,2], [0.,0.,1.], atol=1e-6):
                raise ValueError('Vertical placement requires an upright robot base')
            criteria = ToolPoseCriteria.linear_motion(axis='z',project_distance_to_goal=False)
        elif stage == 'place' and not getattr(self,'free_place_motion',False) and getattr(self,'place_hold_orientation',False):
            criteria = ToolPoseCriteria(non_terminal_pose_axes_weight_factor=[0,0,0,1,1,1],project_distance_to_goal=False)
        self.planner.update_tool_pose_criteria({'panda_hand':criteria})
        gripper = self.sync_gripper()
        attachment = self.sync_attachment(start)
        dynamics = self.sync_dynamics()
        if stage == 'direct':
            return self.plan_grasp(goal, start, dynamics)
        result = self.planner.plan_pose(goal,start,max_attempts=10)
        audit = dict(stage=stage,planner_engine='cuRoboV2',planner_method='plan_pose',
            dynamics=dynamics,attachment=attachment,gripper=gripper,time_dilation_factor=1.,interpolation_dt_s=.004,
            terminal_orientation_constrained=not ee_waypoint,
            intermediate_pose_constraint=not ee_waypoint and (stage in ('lift','approach','retreat','place_descent') or (stage=='place' and not getattr(self,'free_place_motion',False))),
            world_line_axis='z' if stage == 'place_descent' else None,
            max_native_attempts=10,shape_diagnostics=dict(exceeded=[],enforced=False))
        if stage == 'retreat':
            audit['linear_non_terminal_scale'] = getattr(self.task, 'retreat_linear_scale', 1.)
        path = self.result_path(result, audit)
        if stage == 'retreat' and path['status'] == 'Success':
            from .roboicl_mesh_sweep import TargetMeshSweep
            # Snapshot the actual open fingers and actual released target.
            sweep = TargetMeshSweep(self.task, self)(self.kinematic_state(path['position']))
            audit['released_target_mesh_sweep'] = sweep
            self.save_evidence('retreat_mesh_sweep.json', sweep)
            if not sweep['accepted']:
                return dict(status='Fail', cartesian_path=dict(audit,rejected_reason='released_target_mesh_sweep'))
        return path

    def plan_home_path(self, curr_joint_pos, initial_joint_pos):
        indices = [self.all_joints.index(n) for n in self.active_joints_name]
        def state(values):
            return JointState.from_position(torch.as_tensor(np.asarray(values)[indices],
                dtype=torch.float32, device='cuda').reshape(1,7),joint_names=self.active_joints_name)
        start, goal = state(curr_joint_pos), state(initial_joint_pos)
        self.planner.update_tool_pose_criteria({'panda_hand':ToolPoseCriteria()})
        gripper = self.sync_gripper()
        attachment = self.sync_attachment(start)
        if attachment['attached']:
            raise ValueError('Return home requires the released payload')
        dynamics = self.sync_dynamics()
        # RoboTwin back_to_origin specifies the original EE pose. Do not add
        # an exact seven-joint goal that removes the arm's redundant solutions.
        pose = self.planner.compute_kinematics(goal).tool_poses.get_link_pose('panda_hand')
        home = GoalToolPose(tool_frames=['panda_hand'],
            position=pose.position.reshape(1,1,1,1,3),quaternion=pose.quaternion.reshape(1,1,1,1,4))
        result = self.planner.plan_pose(home, start, max_attempts=10)
        audit = dict(stage='home',planner_engine='cuRoboV2',planner_method='plan_pose',
            dynamics=dynamics,attachment=attachment,gripper=gripper,time_dilation_factor=1.,interpolation_dt_s=.004,
            intermediate_pose_constraint=False,max_native_attempts=10,
            shape_diagnostics=dict(exceeded=[],enforced=False),
            home_goal_kind='initial_hand_pose',
            requested_initial_hand_position=pose.position.detach().cpu().numpy().tolist(),
            requested_initial_hand_quaternion=pose.quaternion.detach().cpu().numpy().tolist(),
            initial_joint_reference=goal.position.detach().cpu().numpy().tolist())
        return self.result_path(result, audit)

    def result_path(self, result, audit):
        if result is None or not bool(result.success.all()):
            return dict(status='Fail',cartesian_path=dict(audit,rejected_reason='native_v2_planning'))
        path = self.planner.kinematics.get_active_js(result.get_interpolated_plan())
        self.last_trajectory = path
        arrays = {key:getattr(path,key).detach().cpu().numpy().reshape(-1,7) for key in ('position','velocity','acceleration','jerk')}
        states = self.kinematic_state(arrays['position'])
        pose = states.tool_poses.get_link_pose('panda_hand')
        xyz = pose.position.detach().cpu().numpy().reshape(-1,3)
        chord = float(np.linalg.norm(xyz[-1]-xyz[0])); length = float(np.linalg.norm(np.diff(xyz,axis=0),axis=1).sum())
        audit.update(native_constraints_feasible=True,endpoint_distance_m=chord,path_length_m=length,
            path_chord_ratio=length/max(chord,1e-8),terminal_joint_velocity_max=float(abs(arrays['velocity'][-1]).max()),
            terminal_joint_acceleration_max=float(abs(arrays['acceleration'][-1]).max()))
        return dict(status='Success',**arrays,cartesian_path=audit)

    def validate_path(self,path):
        if not path.get('cartesian_path',{}).get('native_constraints_feasible'):
            raise ValueError('Missing native V2 feasibility evidence')
        for field,limit in [('position',self.limits.position),('velocity',self.limits.velocity)]:
            values=np.asarray(path[field]);bounds=limit.detach().cpu().numpy()
            if not np.isfinite(values).all() or np.any(values<bounds[0]-1e-4) or np.any(values>bounds[1]+1e-4):
                raise ValueError('V2 trajectory violates '+field+' limits')

    def plan_batch(self,curr_joint_pos,target_gripper_pose_list,constraint_pose=None,arms_tag=None):
        # RoboTwin's candidate selector only needs per-candidate status. Dataset
        # supplies one fixed grasp orientation, so no extra batch planner exists.
        rows=[self.plan_path(curr_joint_pos,p,constraint_pose,arms_tag) for p in target_gripper_pose_list]
        return dict(status=np.asarray([r['status'] for r in rows],dtype=object),
            position=[r.get('position') for r in rows],velocity=[r.get('velocity') for r in rows])

    def plan_grippers(self,now_val,target_val):
        # Unchanged RoboTwin native linear command; Base_Task adds its hold.
        return dict(num_step=200,per_step=(target_val-now_val)/200,result=np.linspace(now_val,target_val,200))
