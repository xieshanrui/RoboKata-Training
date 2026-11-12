"""Evaluate Version A with the shared IG rollout and unchanged action protocol."""

import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ig-source", type=Path, required=True)
    parser.add_argument("--demo-features", type=Path, required=True)
    options, remaining = parser.parse_known_args()
    root = Path(__file__).resolve().parents[1]
    ig = options.ig_source.resolve()
    sys.path.insert(0, str(ig))
    sys.path.insert(0, str(root / "backend"))
    # Use the trained RoboKata model/backend, and IG's shared rollout modules.
    from examples.baselines.zero_wam import zero_wam
    zero_wam.__path__.append(str(ig / "examples/baselines/zero_wam/zero_wam"))

    import torch
    import tyro
    from demo_attention import SPEC, add_to_adapter_report, attach_demo_attention
    from ig10k_features import load_features
    from examples.baselines.zero_wam.zero_wam.evaluation import EvalArgs, run
    from examples.baselines.zero_wam.zero_wam.policy import ZeroWAMAgent
    from examples.baselines.zero_wam.zero_wam.runtime import atomic_json_save

    class VersionAPolicy(ZeroWAMAgent):
        def load_adapter(self, state):
            super().load_adapter(state)
            for index in SPEC["layers"]:
                name = f"blocks.{index}.demo_attention.gate"
                actual = self.transformer.blocks[index].demo_attention.gate
                if not torch.equal(actual.detach().cpu(), state[name].to(actual.dtype)):
                    raise ValueError(f"Version A gate failed to load: {name}")

        def select_human(self, record):
            key = record["record_sha256"]
            self.selected_demo = self.feature_rows[key][None].to(device=self.device, dtype=self.dtype)
            self.selected_record = key

        def _stream(self, data, timestep, action, *args, **kwargs):
            inputs = super()._stream(data, timestep, action, *args, **kwargs)
            if action:
                if self.selected_demo is None:
                    raise RuntimeError("Version A requires the selected human RGB features")
                inputs["demo_features"] = self.selected_demo
            return inputs

        def prepare_for_eval(self, *args, **kwargs):
            if self.selected_demo is None:
                raise RuntimeError("No human episode selected")
            super().prepare_for_eval(*args, **kwargs)

        def get_action(self, obs):
            action = super().get_action(obs)
            if self.audit_hooks:
                if not all(self.attention_calls.values()) or not torch.isfinite(action).all():
                    raise RuntimeError("Version A inference did not execute all three attention layers")
                atomic_json_save({"human_record": self.selected_record,
                    "attention_calls": self.attention_calls, "action_shape": list(action.shape),
                    "finite": True, "demo_shape": list(self.selected_demo.shape),
                    "gates": {str(i): self.transformer.blocks[i].demo_attention.gate.item() for i in SPEC["layers"]},
                }, self.audit_output / "first_action_verified.json")
                for hook in self.audit_hooks:
                    hook.remove()
                self.audit_hooks = []
            return action

    def factory(config, checkpoint, manifest_path, args):
        if checkpoint["native_config"].get("ig_demo_attention") != SPEC:
            raise ValueError("Expected a Version A checkpoint")
        if "ig_roboicl_channels" in checkpoint["native_config"]:
            raise ValueError("RoboICL world-hand checkpoints cannot be evaluated as IG joints")
        features = load_features(options.demo_features, manifest_path)
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        actor = VersionAPolicy(config, icl_guidance_scale=args.icl_guidance_scale)
        attach_demo_attention(actor.transformer, SPEC)
        for index in SPEC["layers"]:
            actor.transformer.blocks[index].demo_attention.to(device=actor.device, dtype=actor.dtype)
        add_to_adapter_report(actor.transformer, actor.adaptation_report)
        actor.feature_rows = {r["record_sha256"]: features[i] for i, r in enumerate(manifest["human_episodes"])}
        actor.selected_demo = None
        actor.audit_output = Path(args.output_dir)
        actor.attention_calls = {index: 0 for index in SPEC["layers"]}
        actor.audit_hooks = []
        for index in SPEC["layers"]:
            def count_call(module, inputs, output, index=index):
                actor.attention_calls[index] += 1
            actor.audit_hooks.append(actor.transformer.blocks[index].demo_attention.register_forward_hook(count_call))
        atomic_json_save({
            "global_step": checkpoint["global_step"], "progress": checkpoint.get("progress"),
            "architecture": SPEC, "human_records": list(actor.feature_rows),
            "feature_cache": str(options.demo_features.resolve()),
            "backend_policy": str(Path(sys.modules[ZeroWAMAgent.__module__].__file__).resolve()),
            "conditioning": "same selected full human RGB episode for VAE and DINO",
        }, Path(args.output_dir) / "version_a.json")
        return actor

    args = tyro.cli(EvalArgs, args=remaining)
    run(args, agent_factory=factory)


if __name__ == "__main__":
    main()
