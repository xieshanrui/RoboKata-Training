"""Uniform Query-frame indexing with RGB-only Demo selection from frozen pairs."""
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import BatchSampler, DataLoader, DistributedSampler

from data import Dataset, PROMPT, collate, read, write, sha
from openpi.models.tokenizer import PaligemmaTokenizer


class QueryFrames(torch.utils.data.Dataset):
    def __init__(self, cfg):
        record_type = Dataset
        if cfg.get('dataset', 'roboicl') != 'roboicl':
            from transfer_data import TransferRecords
            record_type = TransferRecords
        self.records = record_type(cfg['data'], frames=cfg['demo_frames'], sample_hz=cfg['demo_sample_hz'],
                               seed=cfg['seed'], cache=cfg['demo_cache'])
        self.ends = np.cumsum([r['frames'] for r in self.records.rows])
        self.demos = {r['id']: [] for r in self.records.rows}
        for demo, query in self.records.pairs:
            self.demos[query['id']].append(demo)
        assert all(self.demos.values())
        self.seed, self.epoch = cfg['seed'], 0

    def __len__(self):
        return int(self.ends[-1])

    def resolve(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        ri = int(np.searchsorted(self.ends, index, side='right'))
        row = self.records.rows[ri]
        frame = int(index - (self.ends[ri - 1] if ri else 0))
        candidates = self.demos[row['id']]
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        return row, frame, candidates[int(rng.integers(len(candidates)))]

    def __getitem__(self, index):
        query, frame, demo = self.resolve(index)
        return self.records.window(query, frame, demo)


class RemainingBatches(BatchSampler):
    start = 0

    def __iter__(self):
        return itertools.islice(super().__iter__(), self.start, None)


class FrameLoader:
    """The standard torch distributed sampler/DataLoader, with a saved epoch cursor."""
    dynamic_graph = True  # Demo masks and last-window valid lengths vary by batch.

    def __init__(self, cfg, rank, world):
        self.dataset = QueryFrames(cfg)
        self.rank, self.world = rank, world
        self.global_batch = cfg['global_batch']
        self.image_keys = tuple(cfg.get('query_image_keys', ('base_0_rgb', 'left_wrist_0_rgb')))
        self.contract = None  # Filled by the entry after model configuration is fixed.
        self.gradient_receipt = None
        assert self.global_batch == world * cfg['per_gpu_batch']
        self.sampler = DistributedSampler(self.dataset, num_replicas=world, rank=rank,
                                           shuffle=True, seed=cfg['seed'], drop_last=True)
        self.batches = RemainingBatches(self.sampler, cfg['per_gpu_batch'], drop_last=True)
        self.epoch, self.cursor = 0, 0
        self.tokens, self.mask = PaligemmaTokenizer(48, tokenizer_path=Path(cfg['tokenizer'])).tokenize(PROMPT)
        self.tokens, self.mask = torch.from_numpy(self.tokens), torch.from_numpy(self.mask)
        self.loader = DataLoader(self.dataset, batch_sampler=self.batches, num_workers=0,
                                 collate_fn=self.collate, generator=torch.Generator().manual_seed(cfg['seed'] + rank))

    def collate(self, samples):
        return collate(samples, torch.device('cpu'), self.tokens, self.mask, self.image_keys)

    def __iter__(self):
        self.dataset.epoch = self.epoch
        self.sampler.set_epoch(self.epoch)
        self.batches.start = self.cursor
        for batch in self.loader:
            self.cursor += 1
            yield batch
        self.epoch += 1
        self.cursor = 0

    def state_dict(self):
        return dict(epoch=self.epoch, batches=self.cursor, frames=len(self.dataset),
                    global_batch=self.global_batch, world=self.world, sampler='uniform_query_frames_v1')

    def load_state_dict(self, state):
        expected = self.state_dict()
        assert all(state[k] == expected[k] for k in ('frames', 'global_batch', 'world', 'sampler'))
        assert 0 <= state['batches'] <= len(self.batches)
        self.epoch, self.cursor = state['epoch'], state['batches']

    def record_update(self, model, step, loss, grad_norm, lr, seconds, output):
        values = torch.tensor([loss.detach().item(), float(grad_norm), seconds,
                               torch.cuda.max_memory_allocated()], device=loss.device, dtype=torch.float64)
        if dist.is_initialized():
            avg = values[:1].clone(); dist.all_reduce(avg); values[0] = avg[0] / self.world
            dist.all_reduce(values[1:], op=dist.ReduceOp.MAX)
        if self.rank == 0:
            row = dict(step=step, loss=float(values[0]), gradient_norm=float(values[1]), lr=lr,
                       seconds=float(values[2]), peak_allocated_bytes=int(values[3]),
                       global_batch=self.global_batch, accumulation=1, sampler=self.state_dict())
            row['branch_gradients'] = self.gradient_receipt
            with (output/'metrics.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row)+'\n')

    def verify_gradients(self, model):
        raw = model.module if hasattr(model, 'module') else model
        vision = raw.paligemma_with_expert.paligemma.model.vision_tower
        assert not any(p.requires_grad or p.grad is not None for p in vision.parameters())
        if self.gradient_receipt is None:
            names = ('demo_pool.queries', 'demo_pool.key.weight', 'action_out_proj.weight')
            params = dict(raw.named_parameters())
            values = {n:float(params[n].grad.float().norm()) if params[n].grad is not None else None for n in names}
            assert all(v is not None and np.isfinite(v) and v > 0 for v in values.values()), values
            self.gradient_receipt = values

    def write_checkpoint_receipt(self, folder, step):
        write(folder/'complete.json', dict(step=step, contract=self.contract,
              model_sha256=sha(folder/'model.safetensors'), optimizer_sha256=sha(folder/'optimizer.pt'),
              metadata_sha256=sha(folder/'metadata.pt'), checkpoint_format='openpi_pytorch'))

    def verify_checkpoint(self, folder):
        if (folder/'migration.json').exists():
            migration = read(folder/'migration.json')
            assert migration['passed'] and sha(folder/'model.safetensors') == migration['source_model_sha256']
            assert sha(folder/'optimizer.pt') == migration['optimizer_sha256']
            assert sha(folder/'metadata.pt') == migration['metadata_sha256']
        else:
            marker = read(folder/'complete.json')
            for name, key in [('model.safetensors','model_sha256'),('optimizer.pt','optimizer_sha256'),('metadata.pt','metadata_sha256')]:
                assert sha(folder/name) == marker[key], name
            # Step budget may grow; data, model, sampler and source identities stay fixed.
            assert marker['contract'] == self.contract
