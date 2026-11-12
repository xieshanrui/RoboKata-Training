"""Apply the visual15 catalog to a RoboICL collector config; physics is preserved."""
import argparse
import copy
import hashlib
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / 'env_cfg/roboicl/visual15_v1.json'


def load_catalog():
    catalog = json.loads(CATALOG.read_text(encoding='utf-8'))
    if len(catalog['views']) != 15 or len(catalog['backgrounds']) != 15:
        raise ValueError('Expected 15 views and 15 backgrounds')
    if not math.isclose(sum(v['probability'] for v in catalog['views']), 1.):
        raise ValueError('View probabilities must sum to one')
    for relative, digest in catalog['texture_sha256'].items():
        path = ROOT / 'assets/background_texture' / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Texture differs from catalog: {relative}')
    return catalog


def apply_profile(config, view_id, background_id, seed, *, center=False):
    catalog = load_catalog()
    view = next(v for v in catalog['views'] if v['id'] == view_id)
    background = next(b for b in catalog['backgrounds'] if b['id'] == background_id)
    rng = random.Random(seed)
    jitter = catalog['jitter']
    def sample(value, key):
        return value if center else value + rng.uniform(-jitter[key], jitter[key])
    az = sample(view['azimuth_deg'], 'azimuth_deg')
    el = sample(view['elevation_deg'], 'elevation_deg')
    distance = sample(view['distance_m'], 'distance_m')
    a, e = math.radians(az), math.radians(el)
    target = catalog['lookat_world_m']
    offset = [distance * math.cos(e) * math.sin(a),
              distance * math.cos(e) * math.cos(a), distance * math.sin(e)]
    camera = {'name': 'head_camera', 'position': [p + d for p, d in zip(target, offset)],
              'forward': [-d for d in offset], 'left': [math.cos(a), -math.sin(a), 0.]}
    result = copy.deepcopy(config)
    randomization = result['domain_randomization']
    if any(randomization.get(k) for k in ('random_background', 'random_light',
            'cluttered_table', 'random_table_height', 'random_head_camera_dis')):
        raise ValueError('Explicit visual profiles require disabled implicit domain randomization')
    for side in ('left', 'right'):
        cameras = result[f'{side}_embodiment_config']['static_camera_list']
        found = [i for i, row in enumerate(cameras) if row['name'] == 'head_camera']
        if len(found) != 1:
            raise ValueError('Expected exactly one head camera in each embodiment config')
        cameras[found[0]] = copy.deepcopy(camera)
    result['roboicl_scene_preset_config'] = {k: background[k] for k in ('wall_texture', 'table_texture')}
    result['camera_profile'] = view_id
    result['appearance_id'] = background_id
    result['gallery_case'].update(view=view_id, scene=background_id)
    result['fixed_camera_definition'] = {'view': view_id, 'azimuth_deg': az,
        'elevation_deg': el, 'distance_m': distance, 'reference_lookat_world_m': target,
        'namespace': catalog['schema'], 'seed': seed, 'center': center}
    result['visual_profile'] = {'catalog': catalog['schema'],
        'catalog_sha256': hashlib.sha256(CATALOG.read_bytes()).hexdigest(),
        'view_id': view_id, 'background_id': background_id, 'seed': seed,
        'source_recording_id': config.get('recording_id'),
        'variant_id': f'{view_id}_{background_id}_{seed}' + ('_center' if center else ''),
        'physical_family_unchanged': True, 'replanning_required_for_visual_change': False}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--view', required=True)
    parser.add_argument('--background', required=True)
    parser.add_argument('--seed', type=int, required=True, help='Reuse one seed for all six tasks in a family')
    parser.add_argument('--center', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = apply_profile(json.loads(args.config.read_text(encoding='utf-8')),
                           args.view, args.background, args.seed, center=args.center)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
