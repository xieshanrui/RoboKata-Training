"""Recorded-contact acceptance for an external grasp across a cup mouth."""
import numpy as np


def single_wall_contacts(events, held_steps, fingers, rule, finger_centers=None):
    """Require one finger inside and one outside the specified same wall."""
    held = set(map(int, held_steps))
    grouped = {}
    for e in events:
        if int(e['step']) in held and e['finger'] in fingers and e.get('impulse_ns', 1.) > 1e-6:
            grouped.setdefault(int(e['step']), {}).setdefault(e['finger'], []).append(e['object_local_m'])
    bilateral, valid, spans, centers = 0, 0, [], []
    for step, per_finger in grouped.items():
        if not all(f in per_finger for f in fingers):
            continue
        bilateral += 1
        points = [np.asarray(per_finger[f]) for f in fingers]
        med = np.array([np.median(p, axis=0) for p in points])
        center = med.mean(axis=0)
        span = float(np.linalg.norm(med[0]-med[1]))
        spans.append(span); centers.append(center)
        # Contact manifold positions can move inside a thin wall. When actual
        # fingertip geometry is recorded, use it to establish which side each
        # finger occupies; contacts still establish bilateral wall engagement.
        if finger_centers is None:
            sides = med[:, 2]-rule['wall_center_z_m']
        else:
            sides = np.asarray([finger_centers[f][step, 2] for f in fingers])-rule['wall_center_z_m']
        fractions = []
        for p in points:
            good = ((np.abs(p[:, 0]) <= rule['maximum_cross_axis_m'])
                    & (np.abs(p[:, 2]-rule['wall_center_z_m']) <= rule['wall_half_band_m'])
                    & (p[:, 1] >= rule['rim_height_m']-rule['rim_band_below_m'])
                    & (p[:, 1] <= rule['rim_height_m']+rule['rim_band_above_m']))
            fractions.append(float(good.mean()))
        valid += bool(min(fractions) >= rule['minimum_region_fraction']
                      and sides.min() <= -rule['minimum_wall_side_offset_m']
                      and sides.max() >= rule['minimum_wall_side_offset_m']
                      and rule['minimum_pair_span_m'] <= span <= rule['maximum_pair_span_m'])
    bf, vf = bilateral/max(1, len(held)), valid/max(1, bilateral)
    return dict(passed=bool(bilateral >= rule['minimum_contact_steps']
                           and bf >= rule['minimum_simultaneous_bilateral_fraction']
                           and vf >= rule['minimum_region_fraction']), revision=rule['revision'],
                held_steps=len(held), bilateral_steps=bilateral, valid_single_wall_steps=valid,
                bilateral_fraction=bf, valid_single_wall_fraction=vf,
                contact_span_median_m=float(np.median(spans)) if spans else None,
                contact_span_min_m=min(spans) if spans else None,
                contact_center_median_m=np.median(centers, axis=0).tolist() if centers else None,
                side_evidence='measured_physical_fingertip_centers' if finger_centers is not None else 'contact_positions')


def measured_franka_fingertip_centers(measurements, target_index, physical_parts):
    """Transform the recorded physical fingertip geometry into object space."""
    from scipy.spatial.transform import Rotation
    pose = measurements['object_poses'][:, target_index]
    rotation = Rotation.from_quat(pose[:, [4, 5, 6, 3]]).as_matrix()
    hand, qpos = measurements['hand_pose'], measurements['qpos']
    centers = {}
    # Same physical non-pad boundary as the independent collision auditor.
    cutoff = .0584 + .03609037026762962
    for part in physical_parts:
        name = part['link']
        if name not in ('panda_leftfinger', 'panda_rightfinger'):
            continue
        vertices = np.asarray(part['vertices_in_hand'])
        pad = vertices[vertices[:, 2] > cutoff]
        if not len(pad):
            raise ValueError('Recorded physical finger has no fingertip vertices')
        center = (pad.min(axis=0) + pad.max(axis=0)) / 2
        local = np.tile(center, (len(pose), 1))
        left = name == 'panda_leftfinger'
        joint = 7 if left else 8
        local[:, 1] += (qpos[:, joint] - qpos[0, joint]) * (1 if left else -1)
        world = np.einsum('nij,nj->ni', hand[:, :3, :3], local) + hand[:, :3, 3]
        centers[name] = np.einsum('nji,nj->ni', rotation, world - pose[:, :3])
    if set(centers) != {'panda_leftfinger', 'panda_rightfinger'}:
        raise ValueError('Both recorded physical fingers are required')
    return centers


def opposite_mouth_contacts(events, held_steps, fingers, rule):
    """Check bilateral held contacts against object-local regions in metres."""
    held = set(int(s) for s in held_steps)
    grouped = {}
    for event in events:
        step, finger = int(event['step']), event['finger']
        if step in held and finger in fingers and event.get('impulse_ns', 1.) > 1e-6:
            grouped.setdefault(step, {}).setdefault(finger, []).append(event['object_local_m'])
    bilateral, valid, centers, spans = 0, 0, [], []
    axis = rule['contact_axis']
    mouth_center = np.asarray(rule.get('mouth_center_xz_m', [0., 0.]), dtype=float)
    if mouth_center.shape != (2,) or not np.isfinite(mouth_center).all():
        raise ValueError('Mouth center must contain finite object-local X/Z coordinates')
    offsets = []
    for per_finger in grouped.values():
        if not all(f in per_finger for f in fingers):
            continue
        bilateral += 1
        points = [np.asarray(per_finger[f], dtype=float) for f in fingers]
        med = np.asarray([np.median(p, axis=0) for p in points])
        center, span = med.mean(axis=0), abs(med[0, axis] - med[1, axis])
        centers.append(center.tolist())
        spans.append(float(span))
        center_offset = float(np.linalg.norm(center[[0, 2]] - mouth_center))
        offsets.append(center_offset)
        relative_med = med.copy()
        relative_med[:, [0, 2]] -= mouth_center
        fractions = []
        for p in points:
            relative = p.copy()
            relative[:, [0, 2]] -= mouth_center
            region = ((np.abs(relative[:, axis]) >= rule['min_abs_contact_axis_m'])
                      & (p[:, 1] >= rule['rim_height_m'] - rule['rim_band_below_m'])
                      & (p[:, 1] <= rule['rim_height_m'] + rule['rim_band_above_m'])
                      & (np.abs(relative[:, 2]) <= rule['maximum_cross_axis_m']))
            fractions.append(float(region.mean()))
        valid += bool(min(fractions) >= rule['minimum_region_fraction']
                      and relative_med[0, axis] * relative_med[1, axis] < 0
                      and span >= rule['minimum_pair_span_m']
                      and center_offset <= rule['maximum_center_offset_m'])
    bilateral_fraction = bilateral / max(1, len(held))
    valid_fraction = valid / max(1, bilateral)
    passed = (bilateral >= rule['minimum_contact_steps']
              and bilateral_fraction >= rule['minimum_simultaneous_bilateral_fraction']
              and valid_fraction >= rule['minimum_region_fraction'])
    return dict(passed=bool(passed), revision=rule['revision'], held_steps=len(held),
                bilateral_steps=bilateral, valid_opposing_steps=valid,
                bilateral_fraction=bilateral_fraction, valid_opposing_fraction=valid_fraction,
                contact_span_median_m=float(np.median(spans)) if spans else None,
                contact_span_min_m=min(spans) if spans else None,
                contact_center_median_m=np.median(centers, axis=0).tolist() if centers else None,
                mouth_center_xz_m=mouth_center.tolist(),
                center_offset_median_m=float(np.median(offsets)) if offsets else None,
                center_offset_max_m=max(offsets) if offsets else None)
