"""Qualify recorded ICL EE labels through Pink; not a learned policy score."""
import argparse
import json
from pathlib import Path
import sys
import traceback

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from envs.roboicl_select_curobo import roboicl_select
from envs.robot.roboicl_ee import FrankaKinematics
from collect_roboicl_random_pick import Collector,RECIPE,observe,validate_pick_geometry
from qualify_roboicl_ee import read,write,sha


def replay_record(task,record,geometry,output,*,record_rgb=True):
    source=Path(record['source']);target=Path(output);target.mkdir(parents=True,exist_ok=False)
    config=read(source/'config.json');config.update(need_plan=False,save_path=str(target))
    result=dict(case=record['case'],qualified=False,source_recording_sha256=sha(source/'policy_recording.npz'),
        rgb_recording='all25hz' if record_rgb else 'initial_only',
        controller_sha256=sha(Path(__file__).parents[1]/'envs/robot/roboicl_ee.py'),
        collector_sha256=sha(Path(__file__).with_name('collect_roboicl_random_pick.py')),
        replayer_sha256=sha(Path(__file__)))
    collector=None
    try:
        task.setup_demo(**config);kin=FrankaKinematics(geometry);validate_pick_geometry(kin,task.robot,target)
        with np.load(source/'policy_recording.npz') as z:
            actions=z['action'].copy();rgb,_=observe(task);assert np.array_equal(rgb,z['rgb'][0])
            if not record_rgb:np.savez_compressed(target/'initial_query.npz',rgb=rgb)
        collector=Collector(task,kin,target,dict(RECIPE,planned_gripper_table_clearance_m=.001,
            max_policy_physics_steps=5000,write_preview_frames=False,record_rgb=record_rgb))
        for action in actions:collector.action(action)
        steps=task.physics_step;collector.hold(250)
        t=config['target_index'];others=[i for i in range(3) if i!=t]
        xyz=np.asarray([r['object_poses'] for r in collector.rows])[:,:,:3];tail=collector.rows[-251:]
        error=float(np.linalg.norm(xyz[-251:,t,:2]-np.asarray(config['destination']),axis=1).max())
        shifts=float(np.linalg.norm(xyz[:,others]-xyz[0,others],axis=2).max())
        lifted=np.flatnonzero((xyz[:,:,2]-xyz[0,:,2]).max(axis=0)>=.10).tolist()
        bottom=np.asarray([r['object_bottom_z'][t] for r in tail])
        table=[e for e in task.contact_events if 'table' in e['bodies'] and any(n.startswith('panda_') for n in e['bodies'])]
        stable=float(np.linalg.norm(np.ptp(xyz[-251:,t],axis=0)))
        result.update(status='completed',policy_steps=steps,lifted_object_indices=lifted,destination_error_m=error,
            non_target_displacement_m=shifts,robot_table_contacts=len(table),tail_range_m=stable,
            post_sequence_verification_steps=250,qualified=bool(lifted==[t] and error<.03 and shifts<.03
                and np.all(np.abs(bottom-.74)<.003) and stable<.001 and not table and task.robot.get_left_gripper_val()>.8))
    except Exception as e:result.update(status='failed',error=str(e),traceback=traceback.format_exc())
    finally:
        if collector is not None:collector.save()
        if hasattr(task,'scene'):task.close_env()
        write(target/'receipt.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('index','geometry','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();index=read(a.index);assert index['complete'] and index['audit_passed']
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'launch.json',dict(index_sha256=sha(a.index),geometry_sha256=sha(a.geometry),
        source_sha256=sha(Path(__file__)),role='Expert-label executability, not learned policy performance'))
    task=roboicl_select();results=[]
    for record in index['records']:
        result=replay_record(task,record,a.geometry,a.output/record['case']);results.append(result)
        write(a.output/'summary.json',dict(planned=len(index['records']),completed=len(results),
            successes=sum(r['qualified'] for r in results),rows=results))
        print(json.dumps({k:result.get(k) for k in ('case','qualified','destination_error_m','error')}),flush=True)


if __name__=='__main__':main()
