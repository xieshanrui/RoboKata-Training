"""Check target clearance using the same convex parts as the physics engine.

cuRobo's buffered spheres remain in use for the table, non-targets and arm.
The target-contact check uses physical hand/finger convex geometry instead of
treating those spheres as the physical gripper. This is an opt-in task adapter.
"""
import hppfcl as fcl
import numpy as np
from scipy.spatial import ConvexHull
import transforms3d as t3d


def convex(points):
    points = np.asarray(points, dtype=float)
    hull = ConvexHull(points)
    vertices, triangles = fcl.StdVec_Vec3f(), fcl.StdVec_Triangle()
    for point in points:
        vertices.append(point)
    for face, equation in zip(hull.simplices, hull.equations):
        a, b, c = map(int, face)
        if np.dot(np.cross(points[b]-points[a], points[c]-points[a]), equation[:3]) < 0:
            b, c = c, b
        triangles.append(fcl.Triangle(a, b, c))
    return fcl.Convex(vertices, triangles)


def transform(matrix):
    return fcl.Transform3f(matrix[:3, :3], matrix[:3, 3])


def carried_mesh_sweep(hand_poses, object_in_hand, parts, object_poses, target):
    """AABBs only cull pairs; actual SAPIEN convex parts decide intersection."""
    carried = hand_poses @ object_in_hand
    vertices = np.concatenate([p['vertices_in_actor'] for p in parts[target]])
    points = np.einsum('nij,kj->nki', carried[:, :3, :3], vertices) + carried[:, None, :3, 3]
    lower, upper = points.min(1), points.max(1)
    geometries = [convex(p['vertices_in_actor']) for p in parts[target]]
    request = fcl.DistanceRequest()
    request.enable_signed_distance = True
    audited = []
    for index, other in enumerate(parts):
        if index == target:
            continue
        pose = object_poses[index]
        vertices = np.concatenate([p['vertices_in_actor'] for p in other])
        world = vertices @ pose[:3, :3].T + pose[:3, 3]
        candidates = np.flatnonzero(np.all((upper >= world.min(0)) & (lower <= world.max(0)), axis=1))
        obstacles = [convex(p['vertices_in_actor']) for p in other]
        minimum, overlapping = None, 0
        for step in candidates:
            distance = min(float(fcl.distance(a, transform(carried[step]), b, transform(pose),
                request, fcl.DistanceResult())) for a in geometries for b in obstacles)
            minimum = distance if minimum is None else min(minimum, distance)
            overlapping += int(distance < -1e-6)
        audited.append(dict(object_index=index, broadphase_steps=len(candidates),
            overlapping_steps=overlapping, minimum_checked_distance_m=minimum))
    return dict(method='Actual physical convex parts at every native FK; AABB broad phase only',
        geometry_allowance_m=1e-6, non_target_overlap=audited,
        minimum_object_bottom_m=float(lower[:, 2].min()),
        minimum_bottom_step=int(lower[:, 2].argmin()), path_steps=len(lower),
        start_object_bottom_m=float(lower[0, 2]), end_object_bottom_m=float(lower[-1, 2]))


class TargetMeshSweep:
    def __init__(self, task, planner):
        self.task, self.planner = task, planner
        self.request = fcl.DistanceRequest()
        self.request.enable_nearest_points = True
        self.request.enable_signed_distance = True
        hand = task.robot.left_entity.find_link_by_name('panda_hand')
        inverse = np.linalg.inv(hand.entity_pose.to_transformation_matrix())
        self.gripper = []
        self.snapshot = []
        for name in ('panda_hand', 'panda_leftfinger', 'panda_rightfinger'):
            link = task.robot.left_entity.find_link_by_name(name)
            in_hand = inverse @ link.entity_pose.to_transformation_matrix()
            for index, shape in enumerate(link.collision_shapes):
                points = np.asarray(shape.vertices)*np.asarray(shape.scale)
                local = in_hand @ shape.local_pose.to_transformation_matrix()
                points = points @ local[:3, :3].T + local[:3, 3]
                self.gripper.append((name, convex(points)))
                self.snapshot.append(dict(link=name, part=index, vertices_in_hand=points.tolist()))
        self.target = [convex(part['vertices_in_actor'])
                       for part in task.object_collision_parts[task.target_index]]
        self.gripper_sphere_indices = planner.sphere_indices(
            ('panda_hand', 'panda_leftfinger', 'panda_rightfinger')).cpu().numpy()

    def palm_clearance(self, hand_pose):
        hand = transform(np.asarray(hand_pose))
        target = transform(self.task.target.get_pose().to_transformation_matrix())
        return min(float(fcl.distance(geometry, hand, part, target, self.request, fcl.DistanceResult()))
                   for name, geometry in self.gripper if name == 'panda_hand' for part in self.target)

    def grasp_retreat(self, hand_pose, axis, minimum_retreat, margin):
        def distance(retreat):
            pose = hand_pose.copy()
            pose[:3, 3] -= retreat*axis
            return self.palm_clearance(pose)
        previous = float(minimum_retreat)
        clearance = distance(previous)
        if clearance >= margin:
            return previous, clearance
        for candidate in np.arange(previous+.001, .1, .001):
            clearance = distance(candidate)
            if clearance >= margin:
                lo, hi = previous, float(candidate)
                for _ in range(24):
                    mid = (lo+hi)/2
                    if distance(mid) >= margin:
                        hi = mid
                    else:
                        lo = mid
                return hi, distance(hi)
            previous = float(candidate)
        raise RuntimeError('No physical palm-clear grasp within the native pregrasp distance')

    def __call__(self, states):
        pose = states.tool_poses.get_link_pose('panda_hand')
        xyz = pose.position.detach().cpu().numpy().reshape(-1,3)
        quaternions = pose.quaternion.detach().cpu().numpy().reshape(-1,4)
        spheres = states.robot_spheres.detach().cpu().numpy().reshape(-1,states.robot_spheres.shape[-2],4)
        base = self.planner.robot_origion_pose.to_transformation_matrix()
        target = transform(self.task.target.get_pose().to_transformation_matrix())
        near_goal = np.linalg.norm(xyz-xyz[-1], axis=1) < .005
        minimum = {'panda_hand': float('inf'), 'panda_leftfinger': float('inf'), 'panda_rightfinger': float('inf')}
        worst, accepted = None, True
        # Every native 4 ms sample is checked. cuRobo also keeps its continuous
        # sphere collision constraints against the table and non-target parts.
        for step, (position, quaternion) in enumerate(zip(xyz, quaternions)):
            hand = np.eye(4)
            hand[:3, :3] = t3d.quaternions.quat2mat(quaternion)
            hand[:3, 3] = position
            hand = transform(base @ hand)
            for name, geometry in self.gripper:
                for part, target_geometry in enumerate(self.target):
                    result = fcl.DistanceResult()
                    distance = float(fcl.distance(geometry, hand, target_geometry, target, self.request, result))
                    minimum[name] = min(minimum[name], distance)
                    required = .001 if name == 'panda_hand' else 0.
                    # Opening fingers may touch the target at its contact pose,
                    # but an intersecting physical convex part is never admitted.
                    valid = distance >= required-1e-6
                    if not valid:
                        accepted = False
                        if worst is None or distance-required < worst['distance_m']-worst['required_m']:
                            worst = dict(step=step, link=name, target_part=part, distance_m=distance,
                                         required_m=required, within_contact_region=bool(near_goal[step]))
        arm = spheres.copy()
        arm[:, self.gripper_sphere_indices, 3] = -1
        arm_minimum = float('inf')
        for box in self.planner.direct_obstacle_boxes.values():
            rotation = t3d.quaternions.quat2mat(box['pose'][3:])
            local = (arm[:, :, :3]-np.asarray(box['pose'][:3])) @ rotation
            distance = np.linalg.norm(np.maximum(np.abs(local)-np.asarray(box['dims'])/2, 0), axis=2)-arm[:, :, 3]
            distance[arm[:, :, 3] <= 0] = np.inf
            arm_minimum = min(arm_minimum, float(distance.min()))
        accepted = accepted and arm_minimum >= 0
        return dict(accepted=bool(accepted), method='physical convex hand/finger parts; buffered arm spheres',
                    native_samples=len(xyz), minimum_gripper_clearance_m=minimum,
                    minimum_arm_sphere_clearance_m=arm_minimum, worst=worst,
                    palm_required_m=.001, finger_contact_region_m=.005)
