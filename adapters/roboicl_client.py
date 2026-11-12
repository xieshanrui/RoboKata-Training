"""Local binary client for the project-owned SSH Zero-WAM worker."""
import io
import json
from pathlib import Path
import shlex
import struct
import subprocess
import threading

import numpy as np


class Worker:
    def __init__(self, host, python, source, model, remote_output, log, gpu=0, denoise_steps=50):
        log = Path(log)
        log.parent.mkdir(parents=True, exist_ok=True)
        self.stderr = log.open('w', encoding='utf-8')
        command = ['env', f'CUDA_VISIBLE_DEVICES={gpu}', 'OMP_NUM_THREADS=4',
                   f'PYTHONPATH={source}', f'TORCHINDUCTOR_CACHE_DIR={remote_output}/inductor',
                   f'TRITON_CACHE_DIR={remote_output}/triton', f'HF_HOME={remote_output}/hf_cache',
                   'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1',
                   python, '-u', f'{source}/adapters/roboicl_worker.py', '--model', model,
                   '--output', remote_output, '--denoise-steps', str(denoise_steps)]
        self.process = subprocess.Popen(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                                         host, shlex.join(command)], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self.thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self.thread.start()
        self.ready, _ = self.receive()
        if not self.ready.get('ready'):
            raise RuntimeError(self.ready)

    def _drain_stderr(self):
        for line in iter(self.process.stderr.readline, b''):
            self.stderr.write(line.decode('utf-8', errors='replace'))
            self.stderr.flush()

    def _read(self, count):
        parts = []
        while count:
            block = self.process.stdout.read(count)
            if not block:
                raise RuntimeError(f'Zero-WAM worker exited ({self.process.poll()}); see worker log')
            parts.append(block)
            count -= len(block)
        return b''.join(parts)

    def receive(self):
        length = struct.unpack('!Q', self._read(8))[0]
        if length > 256 * 1024 * 1024:
            raise ValueError(f'Invalid worker packet length: {length}')
        with np.load(io.BytesIO(self._read(length)), allow_pickle=False) as packet:
            arrays = {key: packet[key] for key in packet.files}
        meta = json.loads(arrays.pop('meta').tobytes().decode('utf-8'))
        if 'error' in meta:
            raise RuntimeError(meta['error'] + '\n' + meta.get('traceback', ''))
        return meta, arrays

    def request(self, kind, **values):
        arrays = {key: values.pop(key) for key in list(values) if isinstance(values[key], np.ndarray)}
        packet = io.BytesIO()
        meta = dict(kind=kind, **values)
        np.savez_compressed(packet, meta=np.frombuffer(json.dumps(meta).encode('utf-8'), dtype=np.uint8), **arrays)
        payload = packet.getvalue()
        self.process.stdin.write(struct.pack('!Q', len(payload)) + payload)
        self.process.stdin.flush()
        return self.receive()

    def close(self):
        if self.process.poll() is None:
            try:
                self.request('close')
            finally:
                self.process.stdin.close()
                try:
                    self.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
        self.thread.join(timeout=5)
        self.stderr.close()
