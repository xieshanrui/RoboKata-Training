"""Replay an accepted pick prefix, then validate blended transfer/descent and release.

This is diagnostic full-cycle evidence, not a pure-pick training recording.
The expert's terminal verification hold is not inserted between pick and place.
"""
import argparse
from pathlib import Path
import traceback

import numpy as np

from collect_roboicl_curobo_pick import RecordingTask, NativeRecording, read, write, sha
from envs.robot.roboicl_ee import FrankaKinematics
from envs.utils import ArmTag


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    original = read(args.source/'receipt.json')
    assert original['qualified']
    config = read(args.source/'config.json')
    config.update(need_plan=True, save_path=str(args.output), cartesian_pick_paths=True)
    write(args.output/'config.json',config)
    write(args.output/'provenance.json',dict(source=str(args.source),source_controls_sha256=sha(args.source/'native_controls.npz'),
        collector_sha256=sha(Path(__file__)),role='diagnostic continuation, excluded from pure-pick training'))
    task = RecordingTask()
    task.fast_gripper = True
    task.setup_demo(**config)
    recorder = NativeRecording(task,FrankaKinematics(args.geometry),args.output)
    task.recorder = recorder
    recorder.install_plan_logging()
    receipt = dict(status='started', qualified=False, destination_xy=[-.20,.08])
    try:
        count = original['policy_physics_steps']
        with np.load(args.source/'native_controls.npz') as data:
            prefix = {k:(data[k][:count] if k not in ('initial_drive_position','gripper_collision_envelope_in_hand') else data[k]) for k in data.files}
        temporary = args.output/'prefix';temporary.mkdir()
        np.savez_compressed(temporary/'native_controls.npz',**prefix)
        recorder.replay(temporary)
        with np.load(args.source/'measurements.npz') as measured:
            receipt['prefix_measured_hand_bitwise_equal'] = bool(np.array_equal(np.asarray([r['hand_pose'] for r in recorder.rows]),measured['hand_pose'][:count+1]))
            receipt['prefix_object_pose_bitwise_equal'] = bool(np.array_equal(np.asarray([r['object_poses'] for r in recorder.rows]),measured['object_poses'][:count+1]))
        assert receipt['prefix_measured_hand_bitwise_equal'] and receipt['prefix_object_pose_bitwise_equal']
        recorder.phase = 0
        receipt['transfer_start_step'] = task.physics_step
        current = task.object_state()
        q = np.asarray([j.get_drive_target()[0] for j in task.robot.left_entity.get_active_joints()])
        from collect_roboicl_curobo_pick import command
        target = recorder.kin.pose(command(q))
        target[:2,3] += np.asarray(receipt['destination_xy'])-np.asarray(current['object_poses'][0][:2])
        target[2,3] += .742-current['object_bottom_z'][0]
        # One cuRobo trajectory blends transfer into vertical final descent.
        task.move_from_drive('place',target_hand=target)
        receipt['release_start_step'] = task.physics_step
        assert task.move(task.open_gripper(ArmTag('left')),save_freq=None)
        receipt['retreat_start_step'] = task.physics_step
        task.move_from_drive('lift',displacement=[0,0,.10])
        recorder.hold((-task.physics_step)%10,phase=1)
        receipt['policy_physics_steps'] = task.physics_step
        recorder.hold(125,phase=2)
        tail = recorder.rows[-126:]
        xy = np.asarray([r['object_poses'][0][:2] for r in tail])
        bottom = np.asarray([r['object_bottom_z'][0] for r in tail])
        robot_table = [e for e in task.contact_events if 'table' in e['bodies'] and any(n.startswith('panda_') for n in e['bodies'])]
        receipt.update(status='completed', maximum_destination_error_m=float(np.linalg.norm(xy-np.asarray(receipt['destination_xy']),axis=1).max()),
            final_bottom_z_m=float(bottom[-1]), tail_position_range_m=float(np.linalg.norm(np.ptp(xy,axis=0))),
            robot_table_contacts=len(robot_table), minimum_commanded_gripper_bottom_z_m=min(recorder.commanded_bottom),
            policy_elapsed_s=receipt['policy_physics_steps']*.004)
        receipt['qualified'] = bool(receipt['maximum_destination_error_m']<.03 and np.all(np.abs(bottom-.74)<.003)
            and receipt['tail_position_range_m']<.001 and not robot_table and task.robot.get_left_gripper_val()>.8)
    except Exception as error:
        receipt.update(status='failed',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
    finally:
        task.recorder = None
        recorder.restore_plan_logging()
        recorder.save(receipt.get('policy_physics_steps',task.physics_step))
        write(args.output/'receipt.json',receipt)
        task.close_env()
    print(receipt,flush=True)


if __name__=='__main__':
    main()
