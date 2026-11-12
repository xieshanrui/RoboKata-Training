"""Region events from measured physics; optional requested-only order scoring."""
import numpy as np


CRITERIA = dict(physics_hz=250, entry_samples=5, rearm_samples=5,
    rearm_margin_m=.01, held_bottom_above_m=.79, closed_below=.2,
    release_xy_tolerance_m=.03, release_bottom_max_m=.76, geometry_allowance_m=1e-6)

STOP_SPEED_WINDOW_S = .020  # One recorded RGB interval, five native physics steps.


def route_reference_speed(xyz):
    """Causal displacement speed; do not treat 4 ms contact jitter as travel."""
    window = round(STOP_SPEED_WINDOW_S * CRITERIA['physics_hz'])
    speed = np.full(len(xyz), np.inf)
    speed[window:] = np.linalg.norm(xyz[window:]-xyz[:-window], axis=1) / STOP_SPEED_WINDOW_S
    return speed


def physical_hold_mask(measurements, target, contacts):
    """Measured closed fingers supporting an object clear of the fixed table."""
    if contacts is None:
        raise ValueError('Held-object scoring requires recorded physical contacts')
    n=len(measurements['gripper_command'])
    finger, supported=np.zeros(n,dtype=bool),np.zeros(n,dtype=bool)
    name=f'roboicl_object_{target}'
    for event in contacts:
        step=int(event['physics_step']);bodies=event['bodies']
        if not 0<=step<n or name not in bodies or event['impulse_ns']<=0:
            continue
        finger[step] |= any(b in ('panda_leftfinger','panda_rightfinger') for b in bodies)
        supported[step] |= any(b in ('table','fixed_destination') or b.startswith('roboicl_object_') and b!=name for b in bodies)
    return (measurements['object_bottom_z'][:,target]>.740001) & (measurements['gripper_command']<.2) & finger & ~supported


def score_open_route(measurements, config, contacts=None):
    target = int(config.get('target_index', 0))
    object_xyz = measurements['object_poses'][:, target, :3]
    reference = config['strategy'].get('reference_point', 'object_origin')
    xyz = measurements['hand_pose'][:, :3, 3] if reference == 'panda_hand' else object_xyz
    bottom = measurements['object_bottom_z'][:, target]
    grip = measurements['gripper_command']
    expected = config['strategy']['sequence']
    regions = config['strategy']['regions']
    rule = config['strategy'].get('sequence_rule', 'all_events_exact')
    stops = rule == 'ordered_region_stops'
    stop_criteria = config['strategy'].get('stop_visit',{})
    if stops:
        if set(stop_criteria)!={'maximum_speed_m_s','minimum_samples'}:
            raise ValueError('Explicit stop-visit criteria required')
        if stop_criteria['maximum_speed_m_s']<=0 or stop_criteria['minimum_samples']<CRITERIA['entry_samples']:
            raise ValueError('Invalid stop-visit criteria')
        if config['strategy'].get('motion_profile')=='native_curobo_v2':
            speed=route_reference_speed(xyz)
        else:
            speed=np.linalg.norm(np.gradient(xyz,1/CRITERIA['physics_hz'],axis=0),axis=1) if len(xyz)>1 else np.zeros(len(xyz))
    held = (bottom > CRITERIA['held_bottom_above_m']) & (grip < CRITERIA['closed_below'])
    criteria=dict(CRITERIA)
    if config['strategy'].get('motion_profile')=='native_curobo_v2':
        held=physical_hold_mask(measurements,target,contacts)
        criteria.pop('held_bottom_above_m')
        criteria.update(held_rule='closed, actual finger contact, clear of support',table_z_m=.74)
    runs = np.convolve(held.astype(int), np.ones(5, dtype=int), mode='valid')
    starts = np.flatnonzero(runs == 5)
    begin = int(starts[0]) if len(starts) else None
    events, details, stop_events = [], [], []
    end, release_ok = None, False
    if begin is not None:
        openings = np.flatnonzero(grip[begin:] >= CRITERIA['closed_below'])
        end = int(begin + openings[0]) if len(openings) else len(xyz)-1
        release_ok = bool(len(openings) and bottom[end] <= CRITERIA['release_bottom_max_m']
            and np.linalg.norm(object_xyz[end, :2]-config['destination']) < CRITERIA['release_xy_tolerance_m'])
        for region in regions:
            delta = np.abs(xyz-np.asarray(region['center_m']))
            distance = np.linalg.norm(delta, axis=1)
            if region.get('shape', 'box') == 'sphere':
                inner = distance <= region['radius_m']+CRITERIA['geometry_allowance_m']
                outside = distance > region['radius_m']+CRITERIA['rearm_margin_m']+CRITERIA['geometry_allowance_m']
            else:
                inner = np.all(delta <= region['half_size_m'], axis=1)
                outside = np.any(delta > np.asarray(region['half_size_m'])+.01, axis=1)
            armed, inside_count, outside_count = True, 0, 0
            for step in range(begin, end):
                if armed:
                    inside_count = inside_count+1 if inner[step] and held[step] else 0
                    if inside_count == 5:
                        events.append(dict(region=region['id'], first_step=step-4,
                            confirm_step=step, time_s=(step-4)*.004))
                        armed, outside_count = False, 0
                else:
                    outside_count = outside_count+1 if outside[step] else 0
                    if outside_count == 5:
                        armed, inside_count = True, 0
            details.append(dict(region=region['id'],
                minimum_center_distance_m=float(np.linalg.norm(delta[begin:end], axis=1).min()),
                total_inner_held_s=float(np.sum(inner[begin:end] & held[begin:end])*.004)))
            if stops:
                armed,slow_count,outside_count=True,0,0
                for step in range(begin,end):
                    if armed:
                        slow_count=slow_count+1 if inner[step] and held[step] and speed[step]<=stop_criteria['maximum_speed_m_s'] else 0
                        if slow_count==stop_criteria['minimum_samples']:
                            first=step-slow_count+1
                            stop_events.append(dict(region=region['id'],first_step=first,confirm_step=step,time_s=first*.004))
                            armed,outside_count=False,0
                    else:
                        outside_count=outside_count+1 if outside[step] else 0
                        if outside_count==CRITERIA['rearm_samples']:armed,slow_count=True,0
    events.sort(key=lambda event:(event['first_step'], event['region']))
    geometric_events=events
    if stops:events=sorted(stop_events,key=lambda event:(event['first_step'],event['region']))
    actual = [event['region'] for event in events]
    requested_events = [name for name in actual if name in expected]
    extra_events = [event for event in events if event['region'] not in expected]
    ordered = (requested_events if rule in ('requested_projection_exact','ordered_region_stops') else actual) == expected
    result=dict(expected=expected, actual=actual, events=events, regions=details,
        requested_events=requested_events, extra_events=extra_events,
        sequence_rule=rule, requested_order_success=ordered,
        exact_sequence_success=actual == expected, release_at_destination=release_ok,
        route_success=bool(ordered and release_ok and begin is not None),
        scoring_window_steps=[begin, end], criteria=criteria,
        reference_point=reference, reference_evidence='Measured world position; no expert stage or route progress used')
    if stops:result.update(geometric_events=geometric_events,geometric_sequence=[e['region'] for e in geometric_events],
        stop_visit_criteria=stop_criteria,
        stop_speed_measurement_window_s=STOP_SPEED_WINDOW_S if config['strategy'].get('motion_profile')=='native_curobo_v2' else None,
        event_definition='Measured route-reference speed inside a region for the frozen consecutive sample count; native V2 uses causal20ms displacement, no expert progress/stage labels',
        passing_entries_count=len(geometric_events)-len(events))
    return result
