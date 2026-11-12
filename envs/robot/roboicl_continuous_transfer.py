"""Plan a carried transfer as one smooth, validated timed joint path.

Internal clearance waypoints are planned before execution. Arc-length timing
removes the zero-speed boundaries of the internal roadmap edges; the entire
result is checked again against native cuRobo constraints before it is used.
"""
import numpy as np
import time
import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algorithm
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig, PoseCostMetric
from curobo.geom.types import WorldConfig
import transforms3d as t3d


def plan_waypoints(planner,start,goal,waypoints,metrics,stage,knot_spacing=.15,quaternions=None):
    generator=planner.motion_gen
    quaternion=goal.quaternion[0].detach().cpu().numpy()
    paths=[];statuses=[];native_calls=[];current=start
    for edge,(point,metric) in enumerate(zip(waypoints,metrics)):
        if stage=='direct':
            world=dict(cuboid=dict(planner.active_world['cuboid']))
            if edge==0:
                world['cuboid'].update({f'contact_{name}':box for name,box in planner.direct_obstacle_boxes.items()})
            generator.update_world(WorldConfig.from_dict(world))
        target=Pose.from_list(point.tolist()+(quaternion if quaternions is None else quaternions[edge]).tolist())
        started=time.perf_counter()
        # A long geometric edge can exceed the native 0.15s knot-time cap.
        # Permit slower internal edges, retaining native smoothness refinement.
        # This is not the execution clock: TOPPRA retimes the complete route.
        # Preserve tensor identity for captured graphs and restore upstream
        # limits before returning to any other planning call.
        limits={id(s.traj_evaluator_config.max_dt):s.traj_evaluator_config.max_dt
            for s in (generator.trajopt_solver,generator.finetune_trajopt_solver)}
        saved={key:value.clone() for key,value in limits.items()}
        try:
            for value in limits.values():value.fill_(.6)
            result=generator.plan_single(current,target,MotionGenPlanConfig(max_attempts=10,
                pose_cost_metric=metric))
        finally:
            for key,value in limits.items():value.copy_(saved[key])
        native_calls.append(dict(edge=edge,attempts=int(result.attempts),trajopt_attempts=int(result.trajopt_attempts),
            wall_s=time.perf_counter()-started,solver_s=float(result.solve_time),status=str(result.status),
            native_finetune_enabled=True,internal_maximum_knot_dt_s=.6,
            execution_timing='whole-route TOPPRA'))
        statuses.append('Success' if result.success.item() else str(result.status))
        if not result.success.item():
            generator.update_world(WorldConfig.from_dict(planner.active_world))
            return dict(status='Fail',cartesian_path=dict(stage=stage,rejected_reason='internal_clearance_roadmap',
                internal_statuses=statuses,internal_planning_calls=native_calls))
        q=result.interpolated_plan.position.detach().cpu().numpy()
        paths.append(q)
        current=JointState.from_position(result.interpolated_plan.position[-1:].clone(),joint_names=planner.active_joints_name)
    generator.update_world(WorldConfig.from_dict(planner.active_world))
    route=np.concatenate(paths)
    distance=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(route,axis=0),axis=1))]
    keep=np.r_[True,np.diff(distance)>1e-6]
    route,distance=route[keep],distance[keep]
    # Retain the native geometric path at dense arc-length knots. Cubic timing
    # gives continuous position/velocity/acceleration across internal waypoints.
    indices=[0]
    for i in range(1,len(route)-1):
        if distance[i]-distance[indices[-1]]>=knot_spacing:indices.append(i)
    if len(indices)>1 and distance[-1]-distance[indices[-1]]<knot_spacing/2:
        indices.pop()
    indices.append(len(route)-1)
    path=ta.SplineInterpolator(distance[indices]/distance[-1],route[indices],bc_type='natural')
    grid=np.linspace(0,1,101)
    acceleration_constraint=constraint.JointAccelerationConstraint(np.full(7,4.))
    timing=algorithm.TOPPRA([constraint.JointVelocityConstraint(np.full(7,1.5)),
        acceleration_constraint],path,gridpoints=grid,solver_wrapper='ecos').compute_trajectory(0,0)
    if timing is None:
        return dict(status='Fail',cartesian_path=dict(stage=stage,rejected_reason='whole_path_time_parameterization'))
    steps=int(np.ceil(timing.duration/.004))+1;duration=(steps-1)*.004
    t=np.linspace(0,timing.duration,steps)
    factor=timing.duration/duration
    q=timing(t).astype(np.float32);dq=(timing(t,1)*factor).astype(np.float32);ddq=(timing(t,2)*factor**2).astype(np.float32)
    state=generator.kinematics.get_state(generator.tensor_args.to_device(q))
    xyz=state.ee_position.detach().cpu().numpy()
    speed=np.gradient(xyz,.004,axis=0);acceleration=np.gradient(speed,.004,axis=0)
    tensor=generator.tensor_args.to_device(q)
    planner.last_trajectory=JointState.from_position(tensor,joint_names=planner.active_joints_name)
    planner.last_trajectory.velocity=generator.tensor_args.to_device(dq)
    native=generator.check_constraints(JointState.from_position(tensor,joint_names=planner.active_joints_name))
    audit=dict(stage=stage,method='native clearance roadmap, continuous cubic joint path and whole-path TOPPRA timing',
        internal_waypoints=[p.tolist() for p in waypoints],internal_statuses=statuses,internal_planning_calls=native_calls,
        duration_s=duration,terminal_settle_steps=0,
        joint_acceleration_limit_rad_s2=4.,
        maximum_joint_velocity=float(np.abs(dq).max()),
        maximum_joint_acceleration=float(np.abs(ddq).max()),
        maximum_ee_speed_m_s=float(np.linalg.norm(speed,axis=1).max()),
        maximum_ee_acceleration_m_s2=float(np.linalg.norm(acceleration,axis=1).max()),
        collision_model='native table/non-target sphere constraints rechecked over full smoothed path')
    if not native.feasible.all().item():
        audit['rejected_reason']='smoothed_path_native_constraints'
        audit['infeasible_samples']=np.flatnonzero(~native.feasible.cpu().numpy().reshape(-1)).tolist()
        audit['minimum_joint_position']=q.min(0).tolist()
        audit['maximum_joint_position']=q.max(0).tolist()
        return dict(status='Fail',cartesian_path=audit)
    if stage=='direct':
        audit['target_mesh_sweep']=planner.target_mesh_sweep(state)
        if not audit['target_mesh_sweep']['accepted']:
            audit['rejected_reason']='physical_target_sweep'
            return dict(status='Fail',cartesian_path=audit)
    return dict(status='Success',position=q,velocity=dq,acceleration=ddq,cartesian_path=audit)


def plan_contact(planner,start,goal):
    endpoint=goal.position[0].detach().cpu().numpy()
    quaternion=goal.quaternion[0].detach().cpu().numpy()
    axis=t3d.quaternions.quat2mat(quaternion)[:,2]
    metric=PoseCostMetric(hold_partial_pose=True,
        hold_vec_weight=planner.motion_gen.tensor_args.to_device([1.,1.,1.,1.,1.,0.]),
        project_to_goal_frame=True)
    return plan_waypoints(planner,start,goal,[endpoint-.1*axis,endpoint-.025*axis,endpoint],
        [None,metric,metric],'direct',knot_spacing=.15)


def plan_transfer(planner,start,goal):
    generator=planner.motion_gen
    initial=generator.kinematics.get_state(start.position)
    position=initial.ee_position[0].detach().cpu().numpy()
    endpoint=goal.position[0].detach().cpu().numpy()
    height=max(position[2],endpoint[2])+planner.transfer_clearance_rise_m
    metric=PoseCostMetric(hold_partial_pose=True,
        hold_vec_weight=generator.tensor_args.to_device([1.,1.,1.,0.,0.,0.]),project_to_goal_frame=False)
    initial_quaternion=initial.ee_quaternion[0].detach().cpu().numpy()
    final_quaternion=goal.quaternion[0].detach().cpu().numpy()
    above=np.r_[endpoint[:2],height]
    # Blend the required orientation change into the transit. A duplicated
    # point above the destination forced a separate rotate-in-place edge.
    return plan_waypoints(planner,start,goal,[np.r_[position[:2],height],above,endpoint],
        [metric,None,metric],'place',knot_spacing=.05,
        quaternions=[initial_quaternion,final_quaternion,final_quaternion])
