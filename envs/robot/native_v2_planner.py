"""Opt-in cuRobo V2 adapter for RoboTwin's native pose/action interface.

Frames and geometry come from the live simulator. This module does not change
the simulator, drives, task goals, camera configuration, or success predicates.
"""
from copy import copy, deepcopy
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import numpy as np
import torch
import trimesh
import transforms3d as t3d
import yaml
import hppfcl as fcl
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState, ToolPoseCriteria, Pose
from curobo._src.geom.types import SceneCfg, Mesh
from curobo._src.geom.sphere_fit.types import SphereFitType
from curobo._src.state.state_joint_trajectory_ops import trim_joint_state_trajectory
from .roboicl_mesh_sweep import convex, transform


def matrix(pose):
    if hasattr(pose, 'to_transformation_matrix'):
        return pose.to_transformation_matrix().astype(float)
    if np.asarray(pose).shape == (4,4): return np.asarray(pose,dtype=float)
    out = np.eye(4)
    out[:3, 3] = pose[:3]
    out[:3, :3] = t3d.quaternions.quat2mat(pose[3:])
    return out


def pose_list(value):
    return value[:3, 3].tolist() + t3d.quaternions.mat2quat(value[:3, :3]).tolist()


def shapes(entity):
    return [s for c in entity.components if hasattr(c, 'collision_shapes') for s in c.collision_shapes]


def vertices(shape):
    if hasattr(shape, 'vertices'):
        return np.asarray(shape.vertices) * np.asarray(shape.scale)
    if hasattr(shape, 'half_size'):
        return np.asarray(trimesh.creation.box(extents=2*np.asarray(shape.half_size)).vertices)
    if type(shape).__name__ == 'PhysxCollisionShapeCylinder':
        # PhysX cylinders run along local X. Circumscribe the round cross
        # section so polygonization never removes physical collision volume.
        angles = np.arange(64) * (2*np.pi/64)
        radius = float(shape.radius) / np.cos(np.pi/64)
        ring = np.column_stack([radius*np.cos(angles), radius*np.sin(angles)])
        return np.concatenate([np.column_stack([np.full(64, x), ring])
                               for x in [-shape.half_length, shape.half_length]])
    raise ValueError('Unsupported physical collision shape: '+type(shape).__name__)


class NativeV2Planner:
    cspace_overrides = {}
    planner_options = {}
    payload_sphere_numerical_inset_m = 1e-5

    def __init__(self, task, side, collision_spheres_path=None, exact_self_links=()):
        self.task, self.side = task, side
        robot = task.robot
        self.entity = getattr(robot, side+'_entity')
        self.names = getattr(robot, side+'_arm_joints_name')
        self.all_names = [j.name for j in self.entity.get_active_joints()]
        self.indices = [self.all_names.index(n) for n in self.names]
        config = yaml.safe_load(Path(getattr(robot, side+'_curobo_yml_path')).read_text(encoding='utf-8'))
        cfg = deepcopy(config['robot_cfg']); k = cfg['kinematics']
        self.exact_self_links = tuple(exact_self_links)
        if self.exact_self_links:
            # Keep scene padding. For these asset links, use nominal spheres
            # for self planning and physical convex parts for final clearance.
            padding = k.get('collision_sphere_buffer', 0.)
            for name in self.exact_self_links:
                if name not in k['collision_link_names']:
                    raise ValueError('Unknown exact self-collision link: '+name)
                k.setdefault('self_collision_buffer', {})[name] = -padding
            names = k['collision_link_names']
            ignore = k['self_collision_ignore']
            self.exact_self_pairs = [(a, b) for i, a in enumerate(names) for b in names[i+1:]
                if (a in self.exact_self_links or b in self.exact_self_links)
                and b not in ignore.get(a, []) and a not in ignore.get(b, [])]
        self.base_link, self.tool_link = k['base_link'], k.pop('ee_link')
        self.base = matrix(self.entity.find_link_by_name(self.base_link).entity_pose)
        self.inverse = np.linalg.inv(self.base)
        tool = matrix(self.entity.find_link_by_name(self.tool_link).entity_pose)
        ee = matrix(getattr(robot, 'get_'+side+'_ee_pose')())
        # The native EE convention is a rigid tool frame, measured without the
        # old approximate arm-base bias or robot-name-dependent rotation patch.
        self.tool_frame = 'roboicl_native_ee'
        k['extra_links'] = k.get('extra_links') or {}
        k['extra_links'][self.tool_frame] = dict(parent_link_name=self.tool_link,
            link_name=self.tool_frame, fixed_transform=pose_list(np.linalg.inv(tool)@ee),
            joint_type='FIXED', joint_name='roboicl_native_ee_joint')
        k['extra_links']['attached_object'] = dict(parent_link_name=self.tool_link,
            # AttachmentManager.update expresses fitted spheres in the first
            # tool frame. The attachment link must use that same frame.
            link_name='attached_object', fixed_transform=k['extra_links'][self.tool_frame]['fixed_transform'],
            joint_type='FIXED', joint_name='roboicl_payload_joint')
        k['tool_frames'] = [self.tool_frame]
        k.pop('use_usd_kinematics', None)
        k['format_version'] = 2.0
        k['cspace'].update(self.cspace_overrides)
        k['asset_root_path'] = str(Path(k['urdf_path']).parent)
        k['cspace']['default_joint_position'] = k['cspace'].pop('retract_config')
        self.locked = k.get('lock_joints', {})
        self.joints = {j.attrib['name']: j for j in ET.parse(k['urdf_path']).getroot().findall('joint')}
        self.parents = {j.find('child').attrib['link']: j for j in self.joints.values()}
        other_side = 'right' if side=='left' else 'left'
        other_entity = getattr(robot,other_side+'_entity')
        # A shared articulation can contain rear arms absent from both arm
        # solver configurations. Keep every outside branch as a live obstacle.
        # This arm's subtree uses self collision; its mounting ancestors are
        # excluded from world collision to avoid treating the mount as an obstacle.
        own_tree = {self.base_link}
        for _ in range(len(self.parents)):
            children = {name for name, joint in self.parents.items()
                        if joint.find('parent').attrib['link'] in own_tree}
            if children <= own_tree:
                break
            own_tree.update(children)
        ancestors = set(); ancestor = self.base_link
        while ancestor in self.parents:
            ancestor = self.parents[ancestor].find('parent').attrib['link']
            ancestors.add(ancestor)
        excluded = own_tree | ancestors if other_entity is self.entity else set()
        self.other_links = [link.entity for link in other_entity.get_links()
                            if link.name not in excluded and shapes(link.entity)]
        self.log('robot_world_obstacles', link_names=[e.name for e in self.other_links])
        self.robot_entity_ids={link.entity.per_scene_id for side_name in ['left','right']
                               for link in getattr(robot,side_name+'_entity').get_links()}
        self.contact_links = [self.tool_link]+[self.joints[n].find('child').attrib['link'] for n in self.locked]
        k['extra_collision_spheres'] = dict(k.get('extra_collision_spheres') or {}, attached_object=64)
        if 'attached_object' not in k['collision_link_names']: k['collision_link_names'].append('attached_object')
        for name in self.contact_links:
            ignores = k['self_collision_ignore'].setdefault(name, [])
            if 'attached_object' not in ignores: ignores.append('attached_object')
        if isinstance(k['collision_spheres'], str):
            k['collision_spheres'] = yaml.safe_load(Path(k['collision_spheres']).read_text(encoding='utf-8'))['collision_spheres']
        if collision_spheres_path is not None:
            k['collision_spheres'] = yaml.safe_load(Path(collision_spheres_path).read_text(encoding='utf-8'))['collision_spheres']
        cfg['load_dynamics'] = True
        self.mesh_cache, self.payload_cache, self.queue = {}, {}, []
        self.held = None
        world = self.scene_world()
        pcfg = MotionPlannerCfg.create(robot=cfg, scene_model=world,
            collision_cache=dict(cuboid=64, mesh=128), num_trajopt_seeds=4,
            interpolation_dt=.004, interpolation_buffer_size=4096, random_seed=123,
            max_goalset=max(1,max((len(list(a.iter_contact_points()))*10 for a in task.objects),default=1)),
            **self.planner_options)
        self.planner = MotionPlanner(pcfg)
        self.kconfig = pcfg.trajopt_solver_config.robot_config.kinematics.kinematics_config
        # Compile pose kernels without requiring free-space random graph seeds
        # in the occupied dual-arm scene. Planning retains native graph fallback.
        self.planner.warmup(enable_graph=False, num_warmup_iterations=1)
        self.sync_gripper()
        fk = self.world_tool_poses(self.state().position)[0]
        error = float(np.linalg.norm(fk[:3,3]-ee[:3,3]))
        angle = float(np.arccos(np.clip((np.trace(fk[:3,:3].T@ee[:3,:3])-1)/2,-1,1)))
        self.log('frame_check', base=self.base.tolist(), ee_offset=k['extra_links'][self.tool_frame],
                 position_error_m=error, rotation_error_rad=angle)
        self.log('robot_adapter', joint_names=self.names, locked_joint_names=list(self.locked),
                 native_tool_link=self.tool_link, contact_links=self.contact_links,
                 collision_spheres_profile=str(collision_spheres_path) if collision_spheres_path else None,
                 scene_sphere_padding_m=k.get('collision_sphere_buffer',0.),
                 self_collision_buffer=k.get('self_collision_buffer',{}))
        if error > 1e-4 or angle > 1e-3:
            raise RuntimeError('V2 FK does not match the physical native EE')

    def log(self, event, **values):
        self.task.v2_evidence.append(dict(side=self.side,event=event,stage=getattr(self.task,'stage',None),**values))

    def state(self, position=None):
        if position is None: position = self.entity.get_qpos()[self.indices]
        return JointState.from_position(torch.as_tensor(position, dtype=torch.float32, device='cuda').reshape(-1,len(self.names)), joint_names=self.names)

    def goal(self, poses):
        transformed = [self.inverse@matrix(p) for p in poses]
        return GoalToolPose(tool_frames=[self.tool_frame],
            position=torch.tensor([t[:3,3] for t in transformed],device='cuda',dtype=torch.float32).reshape(1,1,1,-1,3),
            quaternion=torch.tensor([t3d.quaternions.mat2quat(t[:3,:3]) for t in transformed],device='cuda',dtype=torch.float32).reshape(1,1,1,-1,4))

    def local_mesh(self, entity, index, shape):
        key = (entity.per_scene_id,index)
        if key not in self.mesh_cache:
            v = vertices(shape); local = matrix(shape.local_pose)
            mesh = trimesh.convex.convex_hull(v@local[:3,:3].T+local[:3,3])
            self.mesh_cache[key] = (mesh.vertices.tolist(),mesh.faces.tolist())
        return self.mesh_cache[key]

    def scene_world(self):
        world = dict(cuboid={},mesh={})
        for entity in self.world_entities():
            if self.held is not None and entity.per_scene_id == self.held.actor.per_scene_id: continue
            for i,shape in enumerate(shapes(entity)):
                if 'Plane' in type(shape).__name__: continue
                name = f'body_{entity.per_scene_id}_{i}'
                if hasattr(shape,'half_size'):
                    world['cuboid'][name] = dict(dims=(2*np.asarray(shape.half_size)).tolist(),
                        pose=pose_list(self.inverse@matrix(entity.pose)@matrix(shape.local_pose)))
                else:
                    v,f = self.local_mesh(entity,i,shape)
                    world['mesh'][name] = dict(vertices=v,faces=f,pose=pose_list(self.inverse@matrix(entity.pose)))
        return world

    def world_entities(self):
        # Explicitly exclude robot links if an engine wrapper exposes them.
        # Own-arm geometry belongs to self-collision, never a frozen world mesh.
        return [e for e in self.task.scene.get_all_actors() if e.per_scene_id not in self.robot_entity_ids]+self.other_links

    def sync_gripper(self):
        params = copy(self.kconfig)
        params.fixed_transforms = params.fixed_transforms.clone()
        params.lock_jointstate = params.lock_jointstate.clone()
        for name in self.locked:
            joint = self.joints[name]
            child = self.entity.find_link_by_name(joint.find('child').attrib['link'])
            parent = self.entity.find_link_by_name(joint.find('parent').attrib['link'])
            local = np.linalg.inv(matrix(parent.entity_pose))@matrix(child.entity_pose)
            index = params.link_name_to_idx_map[child.name]
            params.fixed_transforms[index].copy_(torch.as_tensor(local[:3],device='cuda',dtype=params.fixed_transforms.dtype))
            q = float(self.entity.get_qpos()[self.all_names.index(name)])
            params.lock_jointstate.position[...,params.lock_jointstate.joint_names.index(name)] = q
        self.planner.kinematics.update_kinematics_config(params)

    def sync_dynamics(self):
        parts = {}
        for link in self.entity.get_links():
            name = link.name; ancestor=name
            # Only this arm's subtree contributes to its dynamics. Merge fixed
            # children omitted by URDF reduction, retaining their real inertia.
            while ancestor not in self.kconfig.link_name_to_idx_map:
                joint=self.parents.get(ancestor)
                if joint is None or joint.attrib['type']!='fixed':break
                ancestor=joint.find('parent').attrib['link']
            if ancestor not in self.kconfig.link_name_to_idx_map:continue
            parent=self.entity.find_link_by_name(ancestor)
            cm = np.linalg.inv(matrix(parent.entity_pose))@matrix(link.entity_pose)@matrix(link.cmass_local_pose)
            parts.setdefault(ancestor,[]).append((float(link.mass),cm[:3,3],cm[:3,:3]@np.diag(link.inertia)@cm[:3,:3].T))
        if self.held is not None:
            body = next(c for c in self.held.actor.components if hasattr(c,'mass'))
            local = np.linalg.inv(matrix(self.entity.find_link_by_name(self.tool_link).entity_pose))@matrix(self.held.get_pose())@matrix(body.cmass_local_pose)
            parts[self.tool_link].append((float(body.mass),local[:3,3],local[:3,:3]@np.diag(body.inertia)@local[:3,:3].T))
        values = {}
        for name,rows in parts.items():
            mass = sum(m for m,_,_ in rows); com = sum(m*c for m,c,_ in rows)/mass
            inertia = sum(i+m*((c-com)@(c-com)*np.eye(3)-np.outer(c-com,c-com)) for m,c,i in rows)
            values[name] = dict(mass=mass,com=torch.as_tensor(com,device='cuda'),inertia=torch.as_tensor(inertia[[0,1,2,0,0,1],[0,1,2,1,2,2]],device='cuda'))
        self.planner.update_links_inertial(values)

    def sync(self):
        self.sync_gripper()
        self.planner.update_world(SceneCfg.create(self.scene_world()))
        self.planner.attachment_manager.detach()
        if self.held is not None:
            key = self.held.actor.per_scene_id
            if key not in self.payload_cache:
                meshes = []
                for i,shape in enumerate(shapes(self.held.actor)):
                    v,f = self.local_mesh(self.held.actor,i,shape)
                    meshes.append(Mesh(name=f'payload_{i}',vertices=v,faces=f,pose=[0,0,0,1,0,0,0]))
                spheres = self.planner.attachment_manager.fit_spheres(meshes,num_spheres=64,sphere_fit_type=SphereFitType.VOXEL)
                # Resting contact can leave a tiny physical penetration. The
                # task may specify a numerical inset for the planning spheres;
                # its independent physical mesh sweep remains unchanged.
                spheres = spheres.clone()
                spheres[:, 3] = torch.clamp(spheres[:, 3] - self.payload_sphere_numerical_inset_m, min=0.)
                self.payload_cache[key] = spheres
            self.planner.attachment_manager.update(self.payload_cache[key],self.state(),link_name='attached_object',
                world_objects_pose_offset=Pose.from_list(pose_list(self.inverse@matrix(self.held.get_pose()))))
            indices = self.kconfig.get_sphere_index_from_link_name('attached_object')[:len(self.payload_cache[key])]
            actual = self.planner.compute_kinematics(self.state()).robot_spheres.reshape(-1,4)[indices,:3]
            obj = self.inverse @ matrix(self.held.get_pose())
            expected = self.payload_cache[key][:,:3] @ torch.as_tensor(obj[:3,:3].T,device='cuda',dtype=actual.dtype) + torch.as_tensor(obj[:3,3],device='cuda',dtype=actual.dtype)
            error = float(torch.linalg.vector_norm(actual-expected,dim=-1).max())
            self.log('payload_frame_check',maximum_center_error_m=error,sphere_numerical_inset_m=self.payload_sphere_numerical_inset_m)
            if error > 1e-4:
                raise RuntimeError('Attached collision geometry differs from measured object frame')
        self.sync_dynamics()

    def world_tool_poses(self, positions):
        state = self.planner.compute_kinematics(self.state(positions))
        pose = state.tool_poses.get_link_pose(self.tool_frame)
        p = pose.position.detach().cpu().numpy().reshape(-1,3)
        q = pose.quaternion.detach().cpu().numpy().reshape(-1,4)
        return np.asarray([self.base@matrix([*a,*b]) for a,b in zip(p,q)])

    def trajectory(self, value, end=None):
        if end is not None: value=trim_joint_state_trajectory(value,0,end.reshape(-1)[0])
        value = self.planner.kinematics.get_active_js(value)
        arrays = {k:getattr(value,k).detach().cpu().numpy().reshape(-1,len(self.names)) for k in ['position','velocity','acceleration']}
        self.check_self_mesh(arrays['position'])
        self.log('trajectory',samples=len(arrays['position']),terminal_velocity=float(abs(arrays['velocity'][-1]).max()),terminal_acceleration=float(abs(arrays['acceleration'][-1]).max()))
        return dict(status='Success',**arrays)

    def check_self_mesh(self, positions):
        if not self.exact_self_links:
            return
        if not hasattr(self, '_self_meshes'):
            self._self_fk = self.entity.create_pinocchio_model()
            self._self_indices = {link.name: i for i, link in enumerate(self.entity.get_links())}
            self._self_meshes = {}
            for name in {n for pair in self.exact_self_pairs for n in pair}:
                link = self.entity.find_link_by_name(name)
                self._self_meshes[name] = [(convex(vertices(s)), matrix(s.local_pose))
                                           for s in link.collision_shapes]
            # Verify the independent FK against the actual simulator before
            # using it to check any planned trajectory. No physics is stepped.
            self._self_fk.compute_forward_kinematics(self.entity.get_qpos())
            root = matrix(self.entity.get_root_pose())
            for name, index in self._self_indices.items():
                if name not in self._self_meshes:
                    continue
                predicted = root @ matrix(self._self_fk.get_link_pose(index))
                actual = matrix(self.entity.find_link_by_name(name).entity_pose)
                if not np.allclose(predicted, actual, atol=1e-4, rtol=0):
                    raise RuntimeError('Physical self-collision FK mismatch: '+name)
        request = fcl.DistanceRequest(); request.enable_signed_distance = True
        full = self.entity.get_qpos().copy()
        minimum = float('inf'); witness = None
        for index, q in enumerate(positions):
            full[self.indices] = q
            self._self_fk.compute_forward_kinematics(full)
            posed = {name: [(part, transform(matrix(self._self_fk.get_link_pose(self._self_indices[name])) @ local))
                            for part, local in parts] for name, parts in self._self_meshes.items()}
            for a, b in self.exact_self_pairs:
                for pa, ta in posed[a]:
                    for pb, tb in posed[b]:
                        distance = float(fcl.distance(pa, ta, pb, tb, request, fcl.DistanceResult()))
                        if distance < minimum:
                            minimum = distance; witness = dict(sample=index, links=[a, b])
                        if distance < -1e-5:
                            self.log('physical_self_mesh_sweep', accepted=False, minimum_m=minimum, witness=witness)
                            raise RuntimeError('Native V2 arm intersects physical self geometry')
        self.log('physical_self_mesh_sweep', accepted=True, minimum_m=minimum,
                 witness=witness, samples=len(positions), dt=.004)

    def plan_pose(self, pose, constraint_pose=None, **kwargs):
        if self.queue: return self.queue.pop(0)
        if constraint_pose is not None or kwargs.get('last_qpos') is not None:
            raise NotImplementedError('Translate legacy constrained/start-state requests explicitly before enabling this task')
        self.sync()
        criteria = self.task.v2_criteria.get(self.side, ToolPoseCriteria())
        self.planner.update_tool_pose_criteria({self.tool_frame:criteria})
        result = self.planner.plan_pose(self.goal([pose]),self.state(),max_attempts=10)
        if result is None or not bool(result.success.all()):
            self.log('planning_failure',target=pose);return dict(status='Fail')
        return self.trajectory(result.get_interpolated_plan())

    def gripper_geometry(self):
        inverse = np.linalg.inv(matrix(getattr(self.task.robot,'get_'+self.side+'_ee_pose')()))
        geometries = []
        for name in self.contact_links:
            link = self.entity.find_link_by_name(name)
            for shape in link.collision_shapes:
                local = inverse@matrix(link.entity_pose)@matrix(shape.local_pose)
                v = vertices(shape); geometries.append(convex(v@local[:3,:3].T+local[:3,3]))
        return geometries

    def grasp_clearance(self, poses, margin):
        gripper = self.gripper_geometry()
        targets=[]
        for entity in self.world_entities():
            for i,s in enumerate(shapes(entity)):
                if 'Plane' in type(s).__name__:continue
                targets.append((convex(self.local_mesh(entity,i,s)[0]),transform(matrix(entity.pose))))
        request = fcl.DistanceRequest();request.enable_signed_distance=True
        valid=[];minimum=[]
        for p in poses:
            distance = min(float(fcl.distance(a,transform(matrix(p)),b,t,request,fcl.DistanceResult())) for a in gripper for b,t in targets)
            minimum.append(distance);valid.append(distance>=margin)
        return valid,minimum

    def grasp(self, actor, poses, approach_distance):
        self.sync()
        valid,distances = self.grasp_clearance(poses,.001)
        candidates = [p for p,ok in zip(poses,valid) if ok]
        self.log('grasp_candidates',total=len(poses),physical_clearance_passed=len(candidates),clearances_m=distances)
        if not candidates: raise RuntimeError('No native grasp candidate clears the physical open gripper')
        result = self.planner.plan_grasp(self.goal(candidates),self.state(),
            grasp_approach_axis='x',grasp_approach_offset=-approach_distance,
            grasp_approach_in_tool_frame=True,plan_grasp_to_lift=False,
            disable_collision_links=self.contact_links)
        self.planner.enable_link_collision(self.contact_links)
        self.planner.update_tool_pose_criteria({self.tool_frame:ToolPoseCriteria()})
        self.log('plan_grasp',status=result.status)
        if not bool(result.success.all()): raise RuntimeError('Native V2 plan_grasp failed: '+result.status)
        paths = [self.trajectory(getattr(result,n+'_interpolated_trajectory'),getattr(result,n+'_interpolated_last_tstep')) for n in ['approach','grasp']]
        positions = np.concatenate([p['position'] for p in paths])
        valid,distances = self.grasp_clearance(self.world_tool_poses(positions),-1e-5)
        self.log('gripper_scene_mesh_sweep',minimum_m=min(distances),samples=len(valid),accepted=all(valid))
        if not all(valid): raise RuntimeError('Native V2 open gripper intersects physical scene geometry')
        self.queue = paths
        return [pose_list(self.world_tool_poses(p['position'][-1:])[0]) for p in paths]

    def plan_grippers(self, start, goal):
        return dict(num_step=200,per_step=(goal-start)/200,result=np.linspace(start,goal,200))
