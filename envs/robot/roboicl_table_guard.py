"""Task-local arm/table command rejection, combined with the existing hand envelope."""
import hashlib
import itertools
import json

import numpy as np


class ArmTableGuard:
    """Check native collision shapes at the measured pose and every drive waypoint.

    This adds the eight proximal Franka links to the existing conservative
    hand/finger envelope check. It changes no target, joint command or velocity.
    """

    def __init__(self, task, kinematics, margin=.001):
        import hppfcl
        import sapien
        self.fcl, self.kin = hppfcl, kinematics
        self.data = self.kin.model.createData()
        self.margin = float(margin)
        if self.margin != .001:
            raise ValueError('Version2 retains the frozen1mm table clearance')
        self.request = hppfcl.CollisionRequest()
        self.request.security_margin = self.margin
        self.request.num_max_contacts = 1
        self.shapes, self.tables, self.geometry = [], [], {}
        metadata = []
        for name in (f'panda_link{i}' for i in range(8)):
            link = task.robot.left_entity.find_link_by_name(name)
            frame_id = self.kin.model.getFrameId(name)
            if link is None or frame_id >= len(self.kin.model.frames) or not link.collision_shapes:
                raise ValueError('Missing proximal arm collision geometry: '+name)
            for index, shape in enumerate(link.collision_shapes):
                if not isinstance(shape,sapien.physx.PhysxCollisionShapeConvexMesh):
                    raise ValueError('Expected native convex Franka arm collision shapes')
                vertices = np.asarray(shape.vertices,dtype=float)*np.asarray(shape.scale)
                triangles = np.asarray(shape.triangles,dtype=np.int32)
                points, faces = hppfcl.StdVec_Vec3f(),hppfcl.StdVec_Triangle()
                for value in vertices:
                    points.append(value)
                for face in triangles:
                    faces.append(hppfcl.Triangle(*(int(x) for x in face)))
                collision = hppfcl.Convex(points,faces)
                local = shape.local_pose.to_transformation_matrix().astype(float)
                corners = np.asarray(list(itertools.product(*zip(vertices.min(0),vertices.max(0)))))
                key = f'{name}_{index}'
                self.shapes.append((name,index,frame_id,collision,local,corners))
                self.geometry[key+'_vertices'] = vertices
                self.geometry[key+'_triangles'] = triangles
                self.geometry[key+'_local_pose'] = local
                metadata.append(dict(link=name,shape=index,vertices_sha256=hashlib.sha256(vertices.tobytes()).hexdigest(),
                    triangles_sha256=hashlib.sha256(triangles.tobytes()).hexdigest(),local_pose=local.tolist()))
        body = task.table.find_component_by_type(sapien.physx.PhysxRigidStaticComponent)
        if body is None or not body.collision_shapes:
            raise ValueError('Missing native static table collision shapes')
        table_world = body.entity_pose.to_transformation_matrix().astype(float)
        table_metadata = []
        for index,shape in enumerate(body.collision_shapes):
            if not isinstance(shape,sapien.physx.PhysxCollisionShapeBox):
                raise ValueError('Expected native tabletop and table-leg boxes')
            half = np.asarray(shape.half_size,dtype=float)
            world = table_world @ shape.local_pose.to_transformation_matrix()
            extent = np.abs(world[:3,:3]) @ half
            lower,upper = world[:3,3]-extent-self.margin,world[:3,3]+extent+self.margin
            collision = hppfcl.Box(*(2*half))
            transform = hppfcl.Transform3f(world[:3,:3],world[:3,3])
            self.tables.append((index,collision,transform,lower,upper))
            self.geometry[f'table_{index}_half_size'] = half
            self.geometry[f'table_{index}_world_pose'] = world
            table_metadata.append(dict(shape=index,half_size=half.tolist(),world_pose=world.tolist()))
        q = np.asarray(task.robot.left_entity.get_qpos()[:7],dtype=float)
        self.kin.pin.forwardKinematics(self.kin.model,self.data,q)
        self.kin.pin.updateFramePlacements(self.kin.model,self.data)
        errors = {}
        for name,_,frame_id,_,_,_ in self.shapes:
            predicted = self.kin.world_from_base @ self.data.oMf[frame_id].homogeneous
            actual = task.robot.left_entity.find_link_by_name(name).entity_pose.to_transformation_matrix()
            errors[name] = float(np.max(np.abs(predicted-actual)))
        if max(errors.values()) > 1e-5:
            raise ValueError('Arm collision FK differs from native simulator links')
        self.metadata = dict(version=2,mode='reject entire token before any of its physics steps',
            extra_links=[f'panda_link{i}' for i in range(8)],
            hand_and_fingers='Existing fixed collision envelope covers hand and full legal finger travel',
            table_shapes='Actual native tabletop and all four legs; finite geometry',
            margin_m=self.margin,geometry='Native SAPIEN convex collision vertices/triangles/local poses',
            checked_states='Current measured arm and all10 float32 drive joint waypoints',
            target_modification=False,collision_avoidance_replanning=False,
            initial_link_fk_max_abs_errors=errors,arm_shapes=metadata,table_boxes=table_metadata)

    def save_geometry(self, output):
        np.savez_compressed(output/'arm_table_guard_geometry.npz',**self.geometry)
        (output/'arm_table_guard.json').write_text(json.dumps(self.metadata,indent=2)+'\n',encoding='utf-8')

    def check(self, positions, measured=None):
        positions = np.asarray(positions,dtype=np.float32).astype(float)
        if positions.ndim != 2 or positions.shape[1] != 7 or not np.isfinite(positions).all():
            raise ValueError('Arm table guard expects finite seven-joint drive waypoints')
        samples = [('drive',index,q) for index,q in enumerate(positions)]
        if measured is not None:
            samples.insert(0,('measured',-1,np.asarray(measured,dtype=float)))
        for origin,index,q in samples:
            self.kin.pin.forwardKinematics(self.kin.model,self.data,q)
            self.kin.pin.updateFramePlacements(self.kin.model,self.data)
            for name,shape_index,frame_id,geometry,local,corners in self.shapes:
                world = self.kin.world_from_base @ self.data.oMf[frame_id].homogeneous @ local
                bounds = corners @ world[:3,:3].T + world[:3,3]
                lower,upper = bounds.min(0),bounds.max(0)
                for table_index,table,table_pose,table_lower,table_upper in self.tables:
                    if np.any(lower > table_upper) or np.any(upper < table_lower):
                        continue
                    result = self.fcl.CollisionResult()
                    self.fcl.collide(geometry,self.fcl.Transform3f(world[:3,:3],world[:3,3]),
                        table,table_pose,self.request,result)
                    if result.isCollision():
                        return dict(origin=origin,integration_index=index,link=name,link_shape=shape_index,
                            table_shape=table_index,margin_m=self.margin,arm_qpos=q.tolist())
        return None
