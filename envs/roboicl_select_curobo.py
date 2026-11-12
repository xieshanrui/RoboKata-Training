"""Three-object selection scene using the qualified continuous cuRobo motions.

Only the selected object is removed from the contact-stage planner world.
This task is separate from the single-object data and upstream defaults.
"""
import numpy as np
import sapien
import transforms3d as t3d

from .roboicl_random_pick_curobo import roboicl_random_pick_curobo
from .roboicl_random_pick import roboicl_random_pick
from .utils import create_entity_box


class roboicl_select(roboicl_random_pick):
    def setup_demo(self, **kwargs):
        self.official_robotwin_control=bool(kwargs.get('official_robotwin_control',False))
        self.continuous_pick_place=bool(kwargs.get('continuous_pick_place',False))
        self.target_part_boxes=bool(kwargs.get('target_part_boxes',False))
        self.motion_diagnostics=bool(kwargs.get('motion_diagnostics',False))
        self.validate_object_specs(kwargs['object_specs'])
        self.destination=np.asarray(kwargs['destination'],dtype=float)
        if self.destination.shape!=(2,) or not np.isfinite(self.destination).all():
            raise ValueError('Expected a finite destination XY')
        super().setup_demo(**kwargs)

    def validate_object_specs(self, specs):
        if len(specs)!=3 or len({(s['asset'],s['model_id']) for s in specs})!=3:
            raise ValueError('Selection requires three distinct asset identities')

    def load_actors(self):
        super().load_actors()
        # Distinguish contacts for different model IDs sharing an asset name.
        for index,actor in enumerate(self.objects):
            actor.actor.name=f'roboicl_object_{index}'
        self.marker=create_entity_box(self,sapien.Pose([*self.destination,.7405]),
            half_size=[.075,.075,.0005],color=[.25,.65,.30],is_static=True,name='fixed_destination')

    def _step_scene(self):
        super()._step_scene()
        if self._checking_initial_layout:
            names={f'roboicl_object_{i}' for i in range(3)}
            for contact in self.scene.get_contacts():
                pair=tuple(sorted(body.entity.name for body in contact.bodies))
                if names.intersection(pair) and 'table' not in pair:
                    if any(np.linalg.norm(point.impulse)>1e-6 for point in contact.points):
                        self._initial_contacts.add(pair)


class roboicl_select_curobo(roboicl_select,roboicl_random_pick_curobo):
    # Capacity only: every enabled box is still derived from actual geometry.
    planner_object_capacity=256
    planner_extra_static_boxes=1

    def setup_demo(self, **kwargs):
        self.native_grasp_rotation_index=kwargs.get('native_grasp_rotation_index')
        self.parallel_jaw_flip=bool(kwargs.get('parallel_jaw_flip',False))
        self.prefer_native_jaw_orientation=bool(kwargs.get('prefer_native_jaw_orientation',False))
        super().setup_demo(**kwargs)
        self.object_collision_parts=[]
        for body in self.object_bodies:
            parts=[]
            for shape in body.collision_shapes:
                if hasattr(shape,'vertices'):
                    points=np.asarray(shape.vertices)*np.asarray(shape.scale)
                elif hasattr(shape,'half_size'):
                    points=np.asarray([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])*shape.half_size
                else:
                    raise ValueError('Unsupported object collision shape')
                local=shape.local_pose.to_transformation_matrix()
                lo,hi=points.min(0),points.max(0)
                box=local.copy();box[:3,3]+=local[:3,:3] @ ((lo+hi)/2)
                parts.append(dict(vertices_in_actor=(points @ local[:3,:3].T+local[:3,3]).tolist(),
                    box_in_actor=box.tolist(),dims=(hi-lo+2e-6).tolist()))
            self.object_collision_parts.append(parts)
        if sum(map(len,self.object_collision_parts))>self.planner_object_capacity:
            raise ValueError('Object collision-part capacity exceeded')
        support=[]
        marker_pose=self.marker.pose.to_transformation_matrix()
        for body in self.marker.components:
            if hasattr(body,'collision_shapes'):
                points=self._collision_vertices(body)
                support.extend((points @ marker_pose[:3,:3].T+marker_pose[:3,3])[:,2])
        self.destination_top_z_m=float(max(support))
        planner=self.robot.left_planner
        planner.target_mesh_sweep=None
        planner.diagnostic_target_mesh_sweep=None
        self.execution_dynamics=None
        if self.continuous_pick_place:
            if not kwargs.get('official_robotwin_control', False):
                for joint in self.robot.left_arm_joints:
                    joint.set_drive_property(stiffness=5000.,damping=500.)
                self.execution_dynamics=self.robot.left_entity.create_pinocchio_model()
            from .robot.roboicl_mesh_sweep import TargetMeshSweep
            planner.target_mesh_sweep=TargetMeshSweep(self,planner)
        elif self.motion_diagnostics:
            from .robot.roboicl_mesh_sweep import TargetMeshSweep
            planner.diagnostic_target_mesh_sweep=TargetMeshSweep(self,planner)

    def prepare_execution_forces(self):
        context=self._dense_context
        if getattr(self,'execution_dynamics',None) is not None and context:
            sequence=context['sequence']['left_arm']
            step=context['step']
            if sequence is not None and 'acceleration' in sequence and step<len(sequence['acceleration']):
                entity=self.robot.left_entity
                acceleration=np.zeros(len(entity.get_qpos()))
                acceleration[:7]=sequence['acceleration'][step]
                inertia=self.execution_dynamics.compute_generalized_mass_matrix(entity.get_qpos())
                entity.set_qf(entity.get_qf()+inertia @ acceleration)

    def _step_scene(self):
        if getattr(self,'recorder',None) is None:
            self.prepare_execution_forces()
        super()._step_scene()

    def choose_best_pose(self, res_pose, center_pose, arm_tag=None):
        if self.native_grasp_rotation_index is None and not self.parallel_jaw_flip:
            return super().choose_best_pose(res_pose,center_pose,arm_tag)
        # Reuse an upstream grasp rotation, but let the caller validate its
        # corrected grasp endpoint as well as its pregrasp. The upstream
        # chooser returns the first feasible pregrasp, which can have no IK
        # solution at the final grasp of a thin object.
        poses=self.robot.create_target_pose_list(res_pose,center_pose,arm_tag)
        index=self.native_grasp_rotation_index
        if not isinstance(index,int) or not 0<=index<len(poses):
            raise ValueError('Invalid native grasp rotation index')
        if self.parallel_jaw_flip:
            # Parallel jaws admit the same contact with the fingers exchanged.
            # Keep the contact position and approach axis, avoiding a wrist
            # joint-limit branch before the object is lifted.
            pose=poses[index]
            rotation=t3d.quaternions.quat2mat(pose[3:]) @ np.diag([1.,-1.,-1.])
            poses[index]=list(pose[:3])+t3d.quaternions.mat2quat(rotation).tolist()
        plan=self.robot.left_plan_path if arm_tag=='left' else self.robot.right_plan_path
        return poses[index] if plan(poses[index])['status']=='Success' else None

    def load_robot(self,**kwargs):
        super().load_robot(**kwargs)
        for side in ('left','right'):
            # Three-object transfer needs vertical clearance. A measured pilot
            # path had ratio1.531 with no reversal and0.12mm XY overshoot; the
            # single-object1.5 cap rejected that smooth collision-free arc.
            # Keep the same reversal/XY limits and collision model.
            getattr(self.robot,side+'_planner').place_path_chord_limit=1.75
            getattr(self.robot,side+'_planner').place_hold_orientation=bool(kwargs.get('place_hold_orientation',False))
            getattr(self.robot,side+'_planner').continuous_pick_place=bool(kwargs.get('continuous_pick_place',False))

    def set_pregrasp_obstacles(self, enabled):
        boxes,bounds,target_box={},[],None
        for index,(actor,vertices) in enumerate(zip(self.objects,self.object_collision_vertices)):
            pose=actor.get_pose().to_transformation_matrix()
            points=vertices @ pose[:3,:3].T+pose[:3,3]
            lower,upper=points.min(axis=0),points.max(axis=0)
            transform=np.eye(4);transform[:3,3]=(lower+upper)/2
            box=dict(dims=(upper-lower).tolist(),transform=transform)
            if index==self.target_index:target_box=box
            include=bool(enabled or index!=self.target_index)
            names=[]
            if include:
                for part_index,part in enumerate(self.object_collision_parts[index]):
                    name=f'pregrasp_object_{index}_part_{part_index}'
                    boxes[name]=dict(dims=part['dims'],transform=pose @ np.asarray(part['box_in_actor']))
                    names.append(name)
            bounds.append(dict(object_index=index,world_min=lower.tolist(),world_max=upper.tolist(),
                collision_enabled=include,collision_parts=names))
        if self.continuous_pick_place:
            # The green destination is a real static SAPIEN collision plate.
            # Include it in planning; both variants retain identical physics.
            for body in self.marker.components:
                if not hasattr(body,'collision_shapes'):continue
                for index,shape in enumerate(body.collision_shapes):
                    boxes[f'destination_{index}']=dict(dims=(2*np.asarray(shape.half_size)).tolist(),
                        transform=self.marker.pose.to_transformation_matrix() @ shape.local_pose.to_transformation_matrix())
        worlds={}
        for side in ('left','right'):
            planner=getattr(self.robot,side+'_planner')
            worlds[side]=planner.update_pregrasp_world(boxes)
            # Native collision checking handles non-targets. The direct approach
            # sweep additionally checks the removed target until final contact.
            transform=np.linalg.inv(planner.robot_origion_pose.to_transformation_matrix()) @ target_box['transform']
            planner.direct_obstacle_boxes={f'target_{self.target_index}':dict(dims=target_box['dims'],
                pose=transform[:3,3].tolist()+t3d.quaternions.mat2quat(transform[:3,:3]).tolist())}
            if self.target_part_boxes or self.continuous_pick_place:
                inverse=np.linalg.inv(planner.robot_origion_pose.to_transformation_matrix())
                target_pose=self.target.get_pose().to_transformation_matrix()
                planner.direct_obstacle_boxes={}
                for part_index,part in enumerate(self.object_collision_parts[self.target_index]):
                    transform=inverse @ target_pose @ np.asarray(part['box_in_actor'])
                    u,_,vh=np.linalg.svd(transform[:3,:3])
                    planner.direct_obstacle_boxes[f'target_{self.target_index}_part_{part_index}']=dict(
                        dims=part['dims'],pose=transform[:3,3].tolist()+t3d.quaternions.mat2quat(u @ vh).tolist())
        return dict(physics_step=self.physics_step,enabled=enabled,target_index=self.target_index,
            source='Conservative oriented boxes for every physical convex collision part; non-targets retained throughout',
            object_bounds=bounds,worlds=worlds)
