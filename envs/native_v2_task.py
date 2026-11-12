"""Optional native motion primitives for rigid grasp/transport/place tasks.

Task scripts, functional points, action execution, and success checks remain
the owning task's responsibility. Coordinated dual-arm/contact-rich tasks need
separate validation; this mixin does not declare all RoboTwin tasks supported.
"""
import numpy as np
from curobo.types import ToolPoseCriteria
from .robot.robot import Robot
from .robot.native_v2_planner import NativeV2Planner, matrix, pose_list
from .utils import Action


class NativeV2Task:
    def load_robot(self, **kwargs):
        self.need_topp = False
        self.robot = Robot(self.scene, False, **kwargs)
        self.robot.communication_flag = False
        if self.robot.is_dual_arm:
            for side in ['left','right']:
                name=side+'_curobo_yml_path'
                setattr(self.robot,name,getattr(self.robot,name).replace('curobo.yml','curobo_'+side+'.yml'))
        self.robot.init_joints()
        # Match Base_Task.load_robot exactly; this is not a dynamics change.
        for side in ['left','right']:
            for link in getattr(self.robot,side+'_entity').get_links(): link.set_mass(1)
        self.v2_planners, self.v2_criteria, self.v2_pending, self.v2_evidence = {}, {}, {}, []
        self.v2_grasp_clearance_offset=float(kwargs.get('grasp_clearance_offset',0.))
        friction=kwargs.get('finger_friction')
        if friction is not None:
            seen=set();changes=[]
            for side in ['left','right']:
                for joint,_,_ in getattr(self.robot,side+'_gripper'):
                    link=joint.child_link
                    if link.entity.per_scene_id in seen:continue
                    seen.add(link.entity.per_scene_id)
                    for shape in link.collision_shapes:
                        old=shape.physical_material
                        before=[float(old.static_friction),float(old.dynamic_friction)]
                        material=self.scene.create_physical_material(float(friction),float(friction),float(old.restitution))
                        shape.set_physical_material(material)
                        actual=shape.physical_material
                        changes.append(dict(link=link.name,before=before,after=[float(actual.static_friction),float(actual.dynamic_friction)],restitution=float(actual.restitution)))
            self.v2_evidence.append(dict(event='finger_material',changes=changes))
        for side in ['left','right']:
            def plan(pose,*args,_side=side,**kw):return self.v2_planner(_side).plan_pose(pose,*args,**kw)
            def batch(poses,*args,_side=side,**kw):
                rows=[self.v2_planner(_side).plan_pose(p,*args,**kw) for p in poses]
                return dict(status=np.array([r['status'] for r in rows]),position=[r.get('position') for r in rows])
            setattr(self.robot,side+'_plan_path',plan)
            setattr(self.robot,side+'_plan_multi_path',batch)
            setattr(self.robot,side+'_plan_grippers',lambda start,goal:dict(num_step=200,per_step=(goal-start)/200,result=np.linspace(start,goal,200)))

    def v2_planner(self, side):
        side=str(side)
        if side not in self.v2_planners:self.v2_planners[side]=NativeV2Planner(self,side)
        return self.v2_planners[side]

    def grasp_actor(self, actor, arm_tag, pre_grasp_dis=.1, grasp_dis=0.,gripper_pos=0.,contact_point_id=None):
        grasp_dis+=self.v2_grasp_clearance_offset
        if grasp_dis>=pre_grasp_dis:raise ValueError('Grasp clearance offset must remain inside native approach distance')
        ids = [i for i,_ in actor.iter_contact_points()] if contact_point_id is None else (contact_point_id if isinstance(contact_point_id,list) else [contact_point_id])
        poses=[]
        for i in ids:
            contact=actor.get_contact_point(i,'matrix')
            grasp=contact@np.array([[0,0,1,0],[-1,0,0,0],[0,-1,0,0],[0,0,0,1]])
            grasp[:3,3]+=grasp[:3,:3]@np.array([-.12-grasp_dis,0,0])
            # For an explicit top-grasp dataset, retain the unmodified asset
            # frame: some embodiment tilt ranges do not include zero tilt.
            candidates=self.robot.create_target_pose_list(pose_list(grasp),actor.get_contact_point(i,'list'),arm_tag)
            if (getattr(self, 'case', {}).get('grasp_max_tilt_deg') is not None
                    and not any(np.allclose(matrix(p),grasp,atol=1e-6,rtol=0) for p in candidates)):
                candidates.insert(0,pose_list(grasp))
            poses.extend(candidates)
        side=str(arm_tag); planner=self.v2_planner(side)
        pre,goal=planner.grasp(actor,poses,pre_grasp_dis-grasp_dis)
        self.v2_pending[side]=actor
        return arm_tag,[Action(arm_tag,'move',target_pose=pre),Action(arm_tag,'move',target_pose=goal),Action(arm_tag,'close',target_gripper_pos=gripper_pos)]

    def place_actor(self, actor, arm_tag, *args, **kwargs):
        # Keep native functional-point and stable-orientation semantics.
        result=super().place_actor(actor,arm_tag,*args,**kwargs)
        if result[0] is not None:
            moves=[a for a in result[1] if a.action=='move']
            if len(moves)==2:
                delta=np.asarray(moves[1].target_pose[:3])-moves[0].target_pose[:3]
                if np.linalg.norm(delta[:2])<1e-5 and abs(delta[2])>1e-6:
                    moves[1].args['v2_linear_world_z']=True
        return result

    def move(self, *actions):
        # Apply world-Z criteria explicitly carried by native motion actions.
        active=[a for a in actions if a is not None and a[0] is not None]
        if len(active)>1:
            raise NotImplementedError('Coordinated dual-arm motion needs joint planning; do not serialize it')
        for action_group in actions:
            if action_group is None or action_group[0] is None:continue
            side=str(action_group[0])
            for action in action_group[1]:
                gripper_before=getattr(self.robot,'get_'+side+'_gripper_val')()
                self.v2_criteria[side]=(ToolPoseCriteria.linear_motion(axis='z',project_distance_to_goal=False)
                    if action.args.get('v2_linear_world_z') else ToolPoseCriteria())
                if action.action=='move' and side in self.v2_planners:
                    self.v2_planners[side].log('action_pose_criteria',
                        linear_world_z=bool(action.args.get('v2_linear_world_z')))
                if not super().move((action_group[0],[action])): return False
                # Native Action normalizes open/close to 'gripper'; its target
                # aperture and the preceding command determine the direction.
                if action.action=='gripper':
                    if action.target_gripper_pos>gripper_before:
                        if side in self.v2_planners:
                            planner=self.v2_planners[side]
                            planner.held=None
                            planner.log('payload_released')
                    elif side in self.v2_pending:
                        planner=self.v2_planner(side)
                        planner.held=self.v2_pending.pop(side)
                        planner.log('payload_attached',actor_id=planner.held.actor.per_scene_id)
        return True
