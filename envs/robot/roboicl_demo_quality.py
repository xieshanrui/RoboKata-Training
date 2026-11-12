"""Demonstration path, visibility and grasp-stability checks."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def longest_seconds(mask, dt):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return float(max(ends - starts, default=0) * dt)


def visibility(xyz, calibration, image_size, dt):
    camera = np.asarray(calibration['extrinsic_cv'])
    intrinsic = np.asarray(calibration['intrinsic_cv'])
    points = xyz @ camera[:, :3].T + camera[:, 3]
    projected = points @ intrinsic.T
    width, height = image_size
    outside = ((points[:, 2] <= 0) | (projected[:, 0] < 0)
               | (projected[:, 0] >= width * projected[:, 2])
               | (projected[:, 1] < 0)
               | (projected[:, 1] >= height * projected[:, 2]))
    return longest_seconds(outside, dt)


def path_quality(poses, calibration, limits, dt=.004):
    poses = np.asarray(poses)
    xyz = poses[:, :3, 3]
    # Fixed 20 ms geometry sampling matches the recorded trajectory audit.
    indices = np.unique(np.r_[np.arange(0, len(poses), max(1, round(.020 / dt))), len(poses)-1])
    length = float(np.linalg.norm(np.diff(xyz[indices], axis=0), axis=1).sum())
    direct = float(np.linalg.norm(xyz[-1] - xyz[0]))
    rotation = Rotation.from_matrix(poses[indices, :3, :3])
    angle = float(np.degrees((rotation[:-1].inv() * rotation[1:]).magnitude()).sum())
    excess_height = float(xyz[:, 2].max() - max(xyz[0, 2], xyz[-1, 2]))
    outside = visibility(xyz, calibration, limits['image_size'], dt)
    reasons = []
    if length-direct > limits['path_excess_m'] and length/max(.03, direct) > limits['path_ratio']:
        reasons.append('excessive_detour')
    if angle > limits['rotation_travel_deg']:
        reasons.append('excessive_rotation')
    if excess_height > limits['excess_height_m']:
        reasons.append('excessive_height')
    if outside >= limits['outside_seconds']-1e-8:
        reasons.append('target_origin_outside_camera')
    return dict(passed=not reasons, reasons=reasons, length_m=length, direct_m=direct,
                rotation_travel_deg=angle, excess_height_m=excess_height,
                outside_longest_seconds=outside)


def grasp_rotation_quality(poses, joints, limit_deg, dt=.004):
    """Measure complete approach rotation, including joint winding."""
    indices = np.unique(np.r_[np.arange(0, len(poses), max(1, round(.020 / dt))), len(poses)-1])
    rotation = Rotation.from_matrix(np.asarray(poses)[indices, :3, :3])
    angle = float(np.degrees((rotation[:-1].inv() * rotation[1:]).magnitude()).sum())
    travel = np.degrees(np.abs(np.diff(np.asarray(joints)[indices], axis=0)).sum(axis=0))
    reasons = []
    if angle > limit_deg:
        reasons.append('excessive_grasp_rotation')
    if travel.max() > limit_deg:
        reasons.append('excessive_grasp_joint_travel')
    return dict(passed=not reasons, reasons=reasons, rotation_travel_deg=angle,
                joint_travel_deg=travel.tolist(), limit_deg=float(limit_deg))


def recorded_quality(folder, case):
    import h5py
    folder = Path(folder)
    limits = case['demo_quality_guard']
    z = np.load(folder/'trace.npz', allow_pickle=False)
    a, stages = z['samples'], z['stages']
    target = case['intent_index']
    obj = a[:, 1+7*target:8+7*target]
    rotations = Rotation.from_quat(obj[:, [4, 5, 6, 3]])
    poses = np.tile(np.eye(4), (len(a), 1, 1))
    poses[:, :3, :3], poses[:, :3, 3] = rotations.as_matrix(), obj[:, :3]
    with h5py.File(folder/'episode.hdf5', 'r') as h:
        calibration = {k: h['observation/head_camera/'+k][0] for k in ['intrinsic_cv', 'extrinsic_cv']}
        for key in calibration:
            assert np.allclose(h['observation/head_camera/'+key][:], calibration[key]), 'Head camera must be fixed'
    paths, reasons = [], []
    indices = np.flatnonzero(stages == 'grasp')
    segment = a[max(0, indices[0]-1):indices[-1]+1]
    grasp_poses = np.tile(np.eye(4), (len(segment), 1, 1))
    grasp_poses[:, :3, :3] = Rotation.from_quat(segment[:, [19, 20, 21, 18]]).as_matrix()
    grasp_check = grasp_rotation_quality(grasp_poses, segment[:, 29:35], limits['rotation_travel_deg'])
    reasons.extend('grasp:'+s for s in grasp_check['reasons'])
    for stage in dict.fromkeys(stages):
        if stage != 'transport_entry' and not str(stage).startswith('waypoint'):
            continue
        indices = np.flatnonzero(stages == stage)
        segment = poses[max(0, indices[0]-1):indices[-1]+1]
        check = path_quality(segment, calibration, limits)
        paths.append(dict(stage=str(stage), **check))
        reasons.extend(str(stage)+':'+s for s in check['reasons'])
    outside = visibility(obj[:, :3], calibration, limits['image_size'], .004)
    if outside >= limits['outside_seconds']-1e-8:
        reasons.append('recording:target_origin_outside_camera')
    carry = ((stages == 'transport_entry') | np.char.startswith(stages, 'waypoint')
             | ((stages == 'place') & (a[:, 49] < .5) & (obj[:, 2] > obj[0, 2]+.02)))
    indices = np.flatnonzero(carry)
    drift = angle = None
    no_support = 0.
    if len(indices):
        ee = Rotation.from_quat(a[:, [19, 20, 21, 18]])
        relative_position = ee.inv().apply(obj[:, :3]-a[:, 15:18])
        relative_rotation = ee.inv()*rotations
        reference = indices[:min(50, len(indices))]
        drift = float(np.linalg.norm(relative_position[indices]-np.median(relative_position[reference], axis=0), axis=1).max())
        angle = float(np.degrees((relative_rotation[reference[len(reference)//2]].inv()*relative_rotation[indices]).magnitude()).max())
        if drift > limits['relative_translation_m'] or angle > limits['relative_rotation_deg']:
            reasons.append('unstable_grasp_relative_pose')
        support = np.zeros(len(a), dtype=bool)
        contacts = json.loads((folder/'contacts.json').read_text(encoding='utf-8'))
        result = json.loads((folder/'result.json').read_text(encoding='utf-8'))
        target_ids = set(result['object_body_ids'][target]) if 'object_body_ids' in result else None
        if target_ids is None and sum(o['asset'] == case['objects'][target]['asset'] for o in case['objects']) > 1:
            raise ValueError('Ambiguous object names require recorded entity IDs for support checks')
        for event in contacts:
            target_contact = bool(target_ids.intersection(event['body_ids'])) if target_ids is not None else case['objects'][target]['asset'] in event['bodies']
            if (target_contact
                    and any(f in event['bodies'] for f in ['fl_link7', 'fl_link8'])):
                support[event['step']] = True
        no_support = longest_seconds(carry & ~support & (obj[:, 2] > obj[0, 2]+.02), .004)
        if no_support >= limits['unsupported_seconds']:
            reasons.append('unsupported_during_carry')
    return dict(passed=not reasons, reasons=reasons, paths=paths, grasp_rotation=grasp_check,
                outside_longest_seconds=outside, relative_translation_max_m=drift,
                relative_rotation_max_deg=angle, unsupported_seconds=no_support,
                scope='motion, visibility and hold stability')
