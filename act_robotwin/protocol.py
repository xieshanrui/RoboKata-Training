"""Task identity is used for splitting/pair lookup, never as a model input."""
import hashlib
import json
from pathlib import Path

DATASET = 'xieshanrui/HumanGen-RoboTwin-43Train-7Test'
REVISION = 'e2a347bb78dd48fd869bd7642a51430a60676876'
DINO_REVISION = '47b73eefe95e8d44ec3623f8890bd894b6ea2d6c'
TRAIN_MANIFEST_SHA = '1fa416a57cd93983ceb31393947a0db2f7eaa3f82dfbebb5ae4de983638c0802'
EVAL_TASKS = ('place_object_scale', 'stamp_seal', 'move_stapler_pad',
              'place_bread_basket', 'place_empty_cup', 'stack_blocks_three')
EXCLUDED = 'open_microwave'
CAMERAS = ('observation.images.cam_high', 'observation.images.cam_left_wrist',
           'observation.images.cam_right_wrist')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8*1024**2), b''):
            h.update(part)
    return h.hexdigest()

def training_rows(root):
    root = Path(root)
    if sha(root/'manifests/train.jsonl') != TRAIN_MANIFEST_SHA:
        raise ValueError('Training manifest differs from the pinned HumanGen release')
    split = read(root/'splits/task_splits.json')
    rows = [json.loads(s) for s in (root/'manifests/train.jsonl').read_text(encoding='utf-8').splitlines()]
    tasks = {r['task'] for r in rows}
    assert len(rows) == 2149 and len({r['pair_id'] for r in rows}) == 2149
    assert len(tasks) == 43 and tasks == set(split['train'])
    assert set(split['test']) == set(EVAL_TASKS) | {EXCLUDED}
    assert not tasks & set(split['test']) and all(r['split'] == 'train' for r in rows)
    return rows

def video_paths(row):
    high = Path(row['robot_video'])
    assert high.parent.name == CAMERAS[0]
    return [str(high.parent.parent/c/high.name).replace('\\','/') for c in CAMERAS]

def verify_local_dino(path):
    path=Path(path)
    meta=read(path/'revision.json')
    if meta['revision']!=DINO_REVISION or set(meta['files'])!={'config.json','preprocessor_config.json','model.safetensors'}:
        raise ValueError('Expected the pinned DINOv2-large revision and its three files')
    for name,digest in meta['files'].items():
        if sha(path/name)!=digest:raise ValueError('Changed DINO file: '+name)
