"""Render recorded expert states with visual15; preserve supervision and timing.

Run in the existing RoboICL cuRobo V2 environment, from the RoboTwin root.
This does not run expert planning or advance physics after scene initialization.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import sapien

from collect_roboicl_dataset import DatasetCollectionTask, validate_config
from roboicl_visual_profiles import apply_profile


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                              default=lambda x: x.tolist()), encoding='utf-8')


def run(args):
    source, output = args.source.resolve(), args.output.resolve()
    config = json.loads((source/'config.json').read_text(encoding='utf-8'))
    receipt = json.loads((source/'receipt.json').read_text(encoding='utf-8'))
    if receipt['status'] != 'completed':
        raise ValueError('Expected a completed source expert recording')
    config = apply_profile(config, args.view, args.background, args.seed, center=args.center)
    validate_config(config)
    with np.load(source/'policy_recording.npz', allow_pickle=False) as old:
        supervision = {key: old[key].copy() for key in old.files if key != 'rgb'}
        expected_shape = old['rgb'].shape
    trace = np.load(source/'measurements.npz', allow_pickle=False)
    steps = supervision['observation_step']
    if (len(steps) != expected_shape[0] or not np.all(np.diff(steps) == 5)
            or not np.array_equal(trace['physics_step'][steps], steps)
            or not np.array_equal(supervision['timestamp_s'], steps * .004)):
        raise ValueError('Source must have aligned 250 Hz states and 50 Hz RGB observations')
    output.mkdir(parents=True)
    (output/'source').mkdir()
    write(output/'config.json', config)
    provenance = {'source': str(source), 'derivative_only': True,
        'source_sha256': {name: sha(source/name) for name in ('config.json', 'receipt.json', 'measurements.npz', 'policy_recording.npz')},
        'physical_acceptance': 'inherited from source; no new physical trial',
        'visual_profile': config['visual_profile'], 'planning_calls_after_setup': 0,
        'physics_steps_after_setup': 0, 'frames': len(steps), 'fps': 50,
        'restoration_max_object_pose_error': 0.0}
    task = DatasetCollectionTask(args.geometry.resolve())
    try:
        runtime_config = dict(config, save_path=str(output), save_data=False, record_control=False)
        task.fast_gripper = False
        task.setup_demo(**runtime_config)
        initial_step = task.physics_step
        frame_list = []
        for i, step in enumerate(steps):
            task.robot.left_entity.set_qpos(trace['qpos'][step])
            task.robot.left_entity.set_qvel(trace['qvel'][step])
            for actor, pose in zip(task.objects, trace['object_poses'][step]):
                actor.actor.set_pose(sapien.Pose(pose[:3], pose[3:]))
            if not np.array_equal(task.robot.left_entity.get_qpos(), trace['qpos'][step]):
                raise RuntimeError('Restored robot joint positions differ')
            poses = np.array([np.r_[a.get_pose().p, a.get_pose().q] for a in task.objects])
            error = float(np.abs(poses - trace['object_poses'][step]).max())
            provenance['restoration_max_object_pose_error'] = max(error, provenance['restoration_max_object_pose_error'])
            if error > 1e-6:
                raise RuntimeError('Restored object poses differ')
            task._update_render()
            task.cameras.update_picture()
            rgb = task.cameras.get_rgb()
            frame = np.stack([np.asarray(ImageOps.pad(Image.fromarray(rgb[name]['rgb']),
                (224,224), method=Image.Resampling.BILINEAR)) for name in ('head_camera','left_camera')])
            frame_list.append(frame)
            if i == 0:
                write(output/'calibration.json', task.cameras.get_config())
                for name, value in zip(('head','wrist'),frame):
                    Image.fromarray(value).save(output/f'initial_{name}.png')
            if (i+1) % 100 == 0:
                print(f'Rendered {i+1}/{len(steps)} observations', flush=True)
        frames = np.asarray(frame_list, dtype=np.uint8)
        if frames.shape != expected_shape or task.physics_step != initial_step:
            raise RuntimeError('Frame shape or paused physics invariant failed')
        target = output/'policy_recording.npz'
        np.savez_compressed(target, rgb=frames, **supervision)
        with np.load(target, allow_pickle=False) as verified:
            if not all(np.array_equal(verified[k],v) for k,v in supervision.items()):
                raise RuntimeError('Supervision or timestamps changed')
            if not np.array_equal(verified['rgb'],frames):
                raise RuntimeError('RGB readback differs')
        provenance.update(passed=True, output_sha256=sha(target),
            supervision_arrays_identical=list(supervision), rgb_shape=list(frames.shape))
        write(output/'rerender_verified.json', provenance)
    except BaseException:
        import traceback
        write(output/'failure.json', {'traceback':traceback.format_exc(), 'source':str(source)})
        raise
    finally:
        task.close_env()
        trace.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--view', required=True)
    parser.add_argument('--background', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--center', action='store_true')
    run(parser.parse_args())
