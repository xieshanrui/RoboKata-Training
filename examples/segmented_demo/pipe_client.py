"""Simulator side of the finite parent-owned segmented model pipe."""
import numpy as np
from protocol import receive, send


class PolicyOutputError(RuntimeError):
    pass


def checked_receive(stream):
    meta, arrays = receive(stream)
    if "error" in meta:
        error = PolicyOutputError if meta.get("error_kind") == "policy_output" else RuntimeError
        raise error(meta["error"])
    return meta, arrays


class ParentWorker:
    def __init__(self, input_stream, output_stream):
        self.input, self.output = input_stream, output_stream
        self.ready, _ = self.request("identity")

    def request(self, kind, **values):
        arrays = {k: values.pop(k) for k in list(values) if isinstance(values[k], np.ndarray)}
        send(self.output, dict(kind=kind, **values), **arrays)
        return checked_receive(self.input)

    def close(self):
        pass  # Parent owns the worker; each scene closes only its simulator.
