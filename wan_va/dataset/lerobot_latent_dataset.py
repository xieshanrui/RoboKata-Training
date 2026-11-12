# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import get_episode_data_index
import datasets
import hashlib
import json
import numpy as np
from pathlib import Path
from collections.abc import Callable
import os
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import torch
from einops import rearrange
from torch.utils.data import DataLoader
from lerobot.constants import HF_LEROBOT_HOME

from .robotwin_action import preprocess_robotwin_actions
from .lerobot_action import (
    LeRobotActionProcessor,
    action_metadata_paths,
    has_action_transform,
)


DATASET_INDEX_CACHE_VERSION = 6


def dataset_video_keys(dataset_root, fallback):
    if has_action_transform(dataset_root):
        configured = LeRobotActionProcessor(dataset_root).video_keys
        if configured:
            return configured
    return list(fallback)


def _file_fingerprint(path):
    try:
        stat = Path(path).stat()
    except FileNotFoundError:
        return None
    return {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def dataset_index_fingerprint(dataset_root, latent_root, video_keys):
    episodes_path = Path(dataset_root) / 'meta' / 'episodes.jsonl'
    episodes_stat = episodes_path.stat()
    action_transform_path, action_stats_path = action_metadata_paths(dataset_root)
    return {
        'version': DATASET_INDEX_CACHE_VERSION,
        'episodes_size': episodes_stat.st_size,
        'episodes_mtime_ns': episodes_stat.st_mtime_ns,
        'latent_root': str(Path(latent_root).resolve()),
        'video_keys': list(video_keys),
        'action_transform_path': str(action_transform_path.resolve()),
        'action_transform': _file_fingerprint(action_transform_path),
        'action_stats_path': str(action_stats_path.resolve()),
        'action_stats': _file_fingerprint(action_stats_path),
    }


def dataset_index_cache_path(dataset_root, fingerprint):
    cache_key = json.dumps(fingerprint, sort_keys=True).encode('utf-8')
    cache_key = hashlib.sha256(cache_key).hexdigest()[:16]
    return (
        Path(dataset_root)
        / '.cache'
        / 'next_forcing'
        / f'valid_metas_{cache_key}.json'
    )


def dataset_indexes_ready(config):
    if not getattr(config, 'enable_dataset_index_cache', True):
        return False
    if getattr(config, 'rebuild_dataset_index_cache', False):
        return False

    repo_list = recursive_find_file(config.dataset_path, 'info.json')
    repo_list = [Path(path.split('/meta/info.json')[0]) for path in repo_list]
    if not repo_list:
        return False

    for dataset_root in repo_list:
        fingerprint = dataset_index_fingerprint(
            dataset_root,
            dataset_root / 'latents',
            dataset_video_keys(dataset_root, config.obs_cam_keys),
        )
        cache_path = dataset_index_cache_path(dataset_root, fingerprint)
        try:
            with cache_path.open('r', encoding='utf-8') as handle:
                payload = json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return False

        cache_files = payload.get('hf_cache_files')
        if payload.get('fingerprint') != fingerprint:
            return False
        if not isinstance(payload.get('valid_metas'), list):
            return False
        if not cache_files or not all(Path(path).is_file() for path in cache_files):
            return False
    return True

def recursive_find_file(directory, filename='info.json'):
    result = []
    try:
        for root, dirs, files in os.walk(directory):
            if filename in files:
                full_path = os.path.join(root, filename)
                result.append(full_path)
    except PermissionError:
        print(f"Error: can not access {directory}")
    except Exception as e:
        print(f"Error: {e}")
    return result

def construct_lerobot(
    repo_id,
    config,
):
    return LatentLeRobotDataset(
        repo_id=repo_id,
        config=config,
    )

def construct_lerobot_multi_processor(config, 
                                      num_init_worker=8,
                                      ):
    datasets_out_lst = []
    construct_func = partial(
        construct_lerobot,
        config=config,
    )
    repo_list = recursive_find_file(config.dataset_path, 'info.json')
    repo_list = [v.split('/meta/info.json')[0] for v in repo_list]
    if not repo_list:
        raise FileNotFoundError(
            f"No LeRobot datasets found under {config.dataset_path}")
    num_init_worker = min(max(int(num_init_worker), 1), len(repo_list))
    if num_init_worker == 1:
        dataset_iterator = map(construct_func, repo_list)
        if getattr(config, 'rank', 0) == 0:
            dataset_iterator = tqdm(
                dataset_iterator,
                total=len(repo_list),
                desc='Loading dataset indexes',
            )
        datasets_out_lst = list(dataset_iterator)
    else:
        with Pool(num_init_worker) as pool:
            dataset_iterator = pool.imap(construct_func, repo_list)
            if getattr(config, 'rank', 0) == 0:
                dataset_iterator = tqdm(
                    dataset_iterator,
                    total=len(repo_list),
                    desc='Loading dataset indexes',
                )
            datasets_out_lst = list(dataset_iterator)
                
    return datasets_out_lst

class MultiLatentLeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config,
        num_init_worker=None,
    ):
        if num_init_worker is None:
            num_init_worker = getattr(config, 'init_worker', 8)
        self._datasets = construct_lerobot_multi_processor(config, 
                                                           num_init_worker, 
                                                           )
        self.index_cache_hits = sum(
            dataset.index_cache_hit for dataset in self._datasets)
        self.index_cache_misses = len(self._datasets) - self.index_cache_hits
        self.hf_cache_hits = sum(
            dataset.hf_cache_hit for dataset in self._datasets)
        self.item_id_to_dataset_id, self.acc_dset_num = (
            self._get_item_id_to_dataset_id()
        )

    def __len__(
        self,
    ):
        return sum(len(v) for v in self._datasets)

    def _get_item_id_to_dataset_id(self):
        item_id_to_dataset_id = {}
        acc_dset_num = {}
        acc_nums = [0]
        id = 0
        for dset_id, dset in enumerate(self._datasets):
            acc_nums.append(acc_nums[-1] + len(dset))
            for _ in range(len(dset)):
                item_id_to_dataset_id[id] = dset_id
                id += 1
        for did in range(len(self._datasets)):
            acc_dset_num[did] = acc_nums[did]
        return item_id_to_dataset_id, acc_dset_num

    def __getitem__(self, idx) -> dict:
        assert idx < len(self)
        cur_dset = self._datasets[self.item_id_to_dataset_id[idx]]
        local_idx = idx - self.acc_dset_num[self.item_id_to_dataset_id[idx]]
        return cur_dset[local_idx]

class LatentLeRobotDataset(LeRobotDataset):
    def __init__(
        self,
        repo_id,
        config=None,
        latent_root=None,
    ):
        self.repo_id = repo_id
        self.root = HF_LEROBOT_HOME / repo_id
        self.image_transforms = None
        self.delta_timestamps = None
        self.episodes = None
        self.tolerance_s = 1e-4
        self.revision = "v2.1"
        self.video_backend = 'pyav'
        self.delta_indices = None
        self.batch_encoding_size = 1
        self.episodes_since_last_encoding = 0
        self.image_writer = None
        self.episode_buffer = None
        self.root.mkdir(exist_ok=True, parents=True)
        self.meta = LeRobotDatasetMetadata(
            self.repo_id, self.root, self.revision, force_cache_sync=False
        )
        self.episodes = [
            episode_index
            for episode_index, episode in self.meta.episodes.items()
            if episode.get('action_config')
        ]
        if not self.episodes:
            raise ValueError(f'No episodes with action_config found in {self.root}')
        self.episode_position = {
            episode_index: position
            for position, episode_index in enumerate(self.episodes)
        }
        self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)
        
        self.latent_path = (
            Path(latent_root) if latent_root is not None else Path(repo_id) / 'latents'
        )
        self.empty_emb = torch.load(config.empty_emb_path, weights_only=False)
        self.config = config
        self.cfg_prob = config.cfg_prob
        self.action_processor = (
            LeRobotActionProcessor(self.root)
            if has_action_transform(self.root)
            else None
        )
        self.used_video_keys = (
            self.action_processor.video_keys
            if self.action_processor is not None
            and self.action_processor.video_keys
            else list(config.obs_cam_keys)
        )
        if self.action_processor is None:
            self.q01 = np.array(config.norm_stat['q01'], dtype='float')[None]
            self.q99 = np.array(config.norm_stat['q99'], dtype='float')[None]
        fingerprint = self._dataset_index_fingerprint()
        cache_payload = self._read_dataset_index_cache(fingerprint)
        self.hf_dataset = self._load_cached_hf_dataset(cache_payload)
        self.hf_cache_hit = self.hf_dataset is not None
        if self.hf_dataset is None:
            data_paths = [
                self.root / self.meta.get_data_file_path(episode_index)
                for episode_index in self.episodes
            ]
            missing_paths = [path for path in data_paths if not path.is_file()]
            if missing_paths:
                raise FileNotFoundError(
                    f'Missing {len(missing_paths)} selected Robotwin parquet files; '
                    f'first missing file: {missing_paths[0]}'
                )
            self.hf_dataset = self.load_hf_dataset()
        action_columns = (
            self.action_processor.columns
            if self.action_processor is not None
            else ['action', 'observation.state']
        )
        missing_columns = [
            column
            for column in action_columns
            if column not in self.hf_dataset.column_names
        ]
        if missing_columns:
            raise KeyError(
                f'Missing action columns in {self.root}: {missing_columns}'
            )
        self._hf_torch_view = self.hf_dataset.with_format(
            type='torch', columns=action_columns, output_all_columns=False
        )
        self.index_cache_hit = self.parse_meta(
            fingerprint=fingerprint,
            cache_payload=cache_payload,
        )
        if self.index_cache_hit and not self.hf_cache_hit:
            self._save_dataset_index_cache(fingerprint, self.new_metas)

    def _dataset_index_fingerprint(self):
        return dataset_index_fingerprint(
            self.root,
            self.latent_path,
            self.used_video_keys,
        )

    def _dataset_index_cache_path(self, fingerprint):
        return dataset_index_cache_path(self.root, fingerprint)

    def _read_dataset_index_cache(self, fingerprint):
        if not getattr(self.config, 'enable_dataset_index_cache', True):
            return None
        if getattr(self.config, 'rebuild_dataset_index_cache', False):
            return None

        cache_path = self._dataset_index_cache_path(fingerprint)
        try:
            with cache_path.open('r', encoding='utf-8') as handle:
                payload = json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None

        if payload.get('fingerprint') != fingerprint:
            return None
        return payload

    def _load_dataset_index_cache(self, fingerprint, cache_payload=None):
        payload = cache_payload
        if payload is None:
            payload = self._read_dataset_index_cache(fingerprint)
        if payload is None:
            return None
        valid_metas = payload.get('valid_metas')
        if not isinstance(valid_metas, list):
            return None
        return valid_metas

    def _load_cached_hf_dataset(self, cache_payload):
        if cache_payload is None:
            return None
        cache_files = cache_payload.get('hf_cache_files')
        if not cache_files or not all(Path(path).is_file() for path in cache_files):
            return None

        cached_datasets = [
            datasets.Dataset.from_file(path) for path in cache_files
        ]
        if len(cached_datasets) == 1:
            return cached_datasets[0]
        return datasets.concatenate_datasets(cached_datasets)

    def _hf_cache_files(self):
        hf_dataset = getattr(self, 'hf_dataset', None)
        if hf_dataset is None:
            return []
        return [
            cache_file['filename']
            for cache_file in hf_dataset.cache_files
            if Path(cache_file['filename']).is_file()
        ]

    def _save_dataset_index_cache(self, fingerprint, valid_metas):
        if not getattr(self.config, 'enable_dataset_index_cache', True):
            return

        cache_path = self._dataset_index_cache_path(fingerprint)
        temporary_path = cache_path.with_name(
            f'.{cache_path.name}.tmp-{os.getpid()}'
        )
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with temporary_path.open('w', encoding='utf-8') as handle:
                json.dump(
                    {
                        'fingerprint': fingerprint,
                        'valid_metas': valid_metas,
                        'hf_cache_files': self._hf_cache_files(),
                    },
                    handle,
                    separators=(',', ':'),
                )
            os.replace(temporary_path, cache_path)
        except OSError as exc:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
            raise RuntimeError(
                f'Failed to write dataset index cache: {cache_path}'
            ) from exc

    def parse_meta(self, fingerprint=None, cache_payload=None):
        if fingerprint is None:
            fingerprint = self._dataset_index_fingerprint()
        cached_metas = self._load_dataset_index_cache(
            fingerprint,
            cache_payload=cache_payload,
        )
        if cached_metas is not None:
            self.new_metas = cached_metas
            return True

        out = []
        for key, value in self.meta.episodes.items():
            episode_index = value["episode_index"]
            tasks = value["tasks"]
            action_config = value["action_config"]
            for acfg in action_config:
                cur_meta = {
                    "episode_index": episode_index,
                    "tasks": tasks,
                }
                if value.get("icl"):
                    cur_meta["icl"] = value["icl"]
                cur_meta.update(acfg)

                check_statu = self._check_meta(
                    cur_meta["start_frame"],
                    cur_meta["end_frame"],
                    cur_meta["episode_index"],
                    cur_meta,
                )

                if check_statu:
                    out.append(cur_meta)
        self.new_metas = out
        self._save_dataset_index_cache(fingerprint, out)
        return False

    def _latent_file(self, meta, key):
        episode_index = int(meta['episode_index'])
        start_frame = int(meta['start_frame'])
        end_frame = int(meta['end_frame'])
        candidates = [
            Path(self.latent_path)
            / f"chunk-{self.meta.get_episode_chunk(episode_index):03d}"
            / key
            / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
        ]
        source_episode = meta.get('source_episode_index')
        source_start = meta.get('source_frame_start')
        source_end = meta.get('source_frame_end')
        if (
            source_episode is not None
            and source_start is not None
            and source_end is not None
        ):
            source_episode = int(source_episode)
            candidates.append(
                Path(self.latent_path)
                / f"chunk-{source_episode // 1000:03d}"
                / key
                / (
                    f"episode_{source_episode:06d}_"
                    f"{int(source_start)}_{int(source_end)}.pth"
                )
            )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return candidates[-1]

    def _check_meta(self, start_frame, end_frame, episode_index, meta=None):
        meta = meta or {
            'episode_index': episode_index,
            'start_frame': start_frame,
            'end_frame': end_frame,
        }
        for key in self.used_video_keys:
            latent_file = self._latent_file(meta, key)
            if not os.path.exists(latent_file):
                return False
        return True

    def _get_global_idx(self, episode_index: int, local_index: int):
        episode_position = self.episode_position[episode_index]
        ep_start = self.episode_data_index["from"][episode_position]
        return local_index + ep_start

    def _get_range_hf_data(self, start_frame, end_frame):
        batch = self._hf_torch_view[start_frame:end_frame]
        return batch

    def _flatten_latent_dict(self, latent_dict):
        out = {}
        for key, value in latent_dict.items():
            for inner_key, inner_value in value.items():
                new_key = f"{key}.{inner_key}"
                out[new_key] = inner_value
        return out

    def _get_range_latent_data(self, meta):
        out = {}
        for key in self.used_video_keys:
            latent_file = self._latent_file(meta, key)
            assert os.path.exists(latent_file)
            latent_data = torch.load(latent_file, weights_only=False)
            out[key] = latent_data
        
        return self._flatten_latent_dict(out)
    
        
    def _cat_video_latents(self,
                           data_dict
                           ):
        latent_lst = []
        for key in self.used_video_keys:
            latent= data_dict[f"{key}.latent"]
            latent_num_frames = data_dict[f"{key}.latent_num_frames"]
            latent_height = data_dict[f"{key}.latent_height"]
            latent_width = data_dict[f"{key}.latent_width"]
            latent = rearrange(latent, 
                                 '(f h w) c -> f h w c', 
                                 f=latent_num_frames, 
                                 h=latent_height, 
                                 w=latent_width)
            latent_lst.append(latent)
        if self.config.env_type == 'robotwin_tshape':
            wrist_latent = torch.cat(latent_lst[1:], dim=2)
            cat_latent = torch.cat([wrist_latent, latent_lst[0]], dim=1)
        else:
            cat_latent = torch.cat(latent_lst, dim=2)

        first_video_key = self.used_video_keys[0]
        encoder_type = getattr(self.config, 'text_encoder_type', 'umt_dense')
        preferred_key = (
            'local_instruction_emb' if encoder_type == 'umt_dense' else 'text_emb'
        )
        text_emb = None
        for key in (preferred_key, 'task_emb', 'text_emb'):
            text_emb = data_dict.get(f"{first_video_key}.{key}")
            if text_emb is not None:
                break
        if text_emb is None:
            raise KeyError(
                f'No supported text embedding found for {first_video_key}'
            )
        if torch.rand(1).item() < self.cfg_prob:
            text_emb = self.empty_emb

        out_dict = dict(
            latents = cat_latent,
            text_emb = text_emb,
        )
        return out_dict
    
    def _action_post_process(
        self,
        local_start_frame,
        local_end_frame,
        latent_frame_ids,
        action,
        state=None,
    ):
        del local_end_frame
        if self.action_processor is not None:
            action_aligned, action_mask_aligned, _ = self.action_processor.process(
                batch=action,
                source_start_frame=local_start_frame,
                latent_frame_ids=latent_frame_ids,
                temporal_down_rate=4,
            )
            latent_frame_num = (len(latent_frame_ids) - 1) // 4 + 1
            action_aligned = rearrange(
                action_aligned, '(f n) c -> c f n 1', f=latent_frame_num
            )
            action_mask_aligned = rearrange(
                action_mask_aligned, '(f n) c -> c f n 1', f=latent_frame_num
            )
            return (
                torch.from_numpy(action_aligned).float(),
                torch.from_numpy(action_mask_aligned).bool(),
            )
        act_shift = max(int(latent_frame_ids[0] - local_start_frame), 0)
        frame_stride = int(latent_frame_ids[1] - latent_frame_ids[0])
        action = action[act_shift:]
        state = state[act_shift:]

        latent_frame_num = (len(latent_frame_ids) - 1) // 4 + 1
        required_action_num = latent_frame_num * frame_stride * 4
        action_aligned, action_mask_aligned = preprocess_robotwin_actions(
            action=action,
            state=state,
            q01=self.q01,
            q99=self.q99,
            inverse_used_action_channel_ids=(
                self.config.inverse_used_action_channel_ids
            ),
            history_size=frame_stride * 4,
            required_size=required_action_num,
        )
        action_aligned = rearrange(action_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
        action_mask_aligned = rearrange(action_mask_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
        return torch.from_numpy(action_aligned).float(), torch.from_numpy(action_mask_aligned).bool()

    def __getitem__(self, idx) -> dict:
        idx = idx % len(self.new_metas)
        cur_meta = self.new_metas[idx]
        episode_index = cur_meta["episode_index"]
        start_frame = cur_meta["start_frame"]
        end_frame = cur_meta["end_frame"]
        local_start_frame = start_frame
        local_end_frame = end_frame

        ori_data_dict = self._get_range_latent_data(cur_meta)

        latent_frame_ids = ori_data_dict[f"{self.used_video_keys[0]}.frame_ids"]
        start_frame = self._get_global_idx(episode_index, start_frame)
        end_frame = self._get_global_idx(episode_index, end_frame)

        hf_data_frames = self._get_range_hf_data(start_frame, end_frame)
        ori_data_dict.update(hf_data_frames)
        out_dict = self._cat_video_latents(ori_data_dict)

        source_start_frame = int(
            cur_meta.get('source_frame_start', local_start_frame)
        )
        action_input = (
            hf_data_frames
            if self.action_processor is not None
            else ori_data_dict['action']
        )
        state_input = (
            None
            if self.action_processor is not None
            else ori_data_dict['observation.state']
        )
        out_dict['actions'], out_dict['actions_mask'] = self._action_post_process(
            source_start_frame,
            local_end_frame,
            latent_frame_ids,
            action_input,
            state_input,
        )

        out_dict['latents'] = out_dict['latents'].permute(3, 0, 1, 2)
        return out_dict

    def __len__(self):
        return len(self.new_metas)

if __name__ == '__main__':
    from wan_va.configs import VA_CONFIGS
    from tqdm import tqdm
    dset = MultiLatentLeRobotDataset(
        VA_CONFIGS['demo_train']
    )
    for key, value in dset[0].items():
        if isinstance(value, torch.Tensor):
            print(f'{key}: {value.shape} tensor')
        elif isinstance(value, np.ndarray):
            print(f'{key}: {value.shape} np')
        else:
            print(f'{key}: {value}')
    print(len(dset))
    dloader = DataLoader(
            dset,
            batch_size=1,
            shuffle=True,
            num_workers=32,
        )
    max_l = 0
    action_list = []
    for data in tqdm(dloader):
        _, _, F, H, W = data['latents'].shape
        max_l = max(max_l, F*H*W)
        action_list.append(data['actions'].flatten(2).permute(0, 2, 1).flatten(0, 1))
    action_all = torch.cat(action_list, dim=0)
    print(max_l)
    print(action_all.shape, action_all.mean(dim=0), action_all.min(dim=0)[0], action_all.max(dim=0)[0])
    
