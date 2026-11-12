import importlib.metadata
def snapshot(task):
 def pose(p):return list(map(float,p.p))+list(map(float,p.q))
 actors=[{'name':a.get_name(),'pose':pose(a.get_pose())} for a in task.scene.get_all_actors()]
 arms=[]
 for a in task.scene.get_all_articulations():
  joints=[]
  for j in a.get_active_joints():
   joints.append({'name':j.get_name(),'stiffness':float(j.get_drive_property()[0]),'damping':float(j.get_drive_property()[1])} if hasattr(j,'get_drive_property') else {'name':j.get_name()})
  arms.append({'name':a.get_name(),'pose':pose(a.get_pose()),'qpos':a.get_qpos().tolist(),'qvel':a.get_qvel().tolist(),'joints':joints})
 c=task.cameras;cams=[('left',c.left_camera),('right',c.right_camera)]+list(zip(c.static_camera_name,c.static_camera_list))
 return {'render_device':str(task.scene.render_system.device),'sapien':importlib.metadata.version('sapien'),'actors':actors,'articulations':arms,'cameras':{n:{'pose':pose(cam.get_pose()),'intrinsic':cam.get_intrinsic_matrix().tolist()} for n,cam in cams}}
