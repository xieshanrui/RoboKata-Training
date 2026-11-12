"""Run dense180 closed-loop policy trials through a local model-owned pipe worker."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
from PIL import Image, ImageOps


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


class Worker:
    def __init__(self, args, output):
        self.configuration = {name: str(getattr(args, name).resolve()) for name in
                              ("worker_python", "worker_script", "worker_source", "checkpoint", "normalization", "tokenizer")}
        self.denoise_steps = args.denoise_steps
        self.output = output.resolve()
        self.artifact_digests = {}
        env = os.environ.copy()
        env["PYTHONPATH"] = str(args.worker_source.resolve()) + os.pathsep + env.get("PYTHONPATH", "")
        self.stderr = (output / "worker.stderr.log").open("w", encoding="utf-8")
        self.messages = queue.Queue()
        command = [str(args.worker_python), "-u", str(args.worker_script.resolve()),
                   "--checkpoint", str(args.checkpoint.resolve()),
                   "--normalization", str(args.normalization.resolve()),
                   "--tokenizer", str(args.tokenizer.resolve()), "--demo", str(args.demo.resolve()),
                   "--seed", str(args.seed), "--denoise-steps", str(args.denoise_steps),
                   "--demo-control", args.demo_control, "--shuffle-seed", str(args.shuffle_seed)]
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self.stderr, text=True, encoding="utf-8", env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        except Exception:
            self.stderr.close()
            raise
        self.timeout = args.worker_timeout
        self.reader = threading.Thread(target=self._read, args=(output / "worker.stdout.log",), daemon=True)
        self.reader.start()
        self.demo_request_id = 0
        self.last_demo_metadata = None
        try:
            if self.receive() != {"ready": True}:
                raise ValueError("Worker did not report ready")
        except Exception:
            self.close()
            raise

    def _read(self, logfile):
        try:
            with logfile.open("w", encoding="utf-8") as log:
                for line in self.process.stdout:
                    log.write(line)
                    log.flush()
                    if line.startswith("ROBOICL:"):
                        self.messages.put(json.loads(line[len("ROBOICL:"):]))
        except Exception as error:
            self.messages.put(error)
        finally:
            self.messages.put(EOFError("Inference worker exited; inspect worker logs"))

    def receive(self):
        try:
            message = self.messages.get(timeout=self.timeout)
        except queue.Empty as error:
            raise TimeoutError("Inference worker timed out") from error
        if isinstance(message, Exception):
            raise message
        if not isinstance(message, dict):
            raise ValueError("Worker protocol requires JSON objects")
        return message

    def predict(self, query, output):
        self.process.stdin.write(json.dumps(dict(input=str(query), output=str(output))) + "\n")
        self.process.stdin.flush()
        response = self.receive()
        if Path(response.get("output", "")).resolve() != output.resolve():
            raise ValueError(f"Unexpected worker response: {response}")
        actions = np.load(output, allow_pickle=False)
        if actions.shape != (50, 180) or not np.isfinite(actions).all():
            raise ValueError("Worker actions must be finite [50,180] dense controls")
        return actions

    def set_demo(self, demo, control, seed, shuffle_seed):
        """Reset one trial's support and inference RNG, with an explicit acknowledgement."""
        self.demo_request_id += 1
        request = dict(set_demo=str(demo.resolve()), demo_control=control, seed=seed,
                       shuffle_seed=shuffle_seed, request_id=self.demo_request_id)
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        response = self.receive()
        if (response.get("demo_ready") != self.demo_request_id
                or Path(response.get("demo", "")).resolve() != demo.resolve()
                or response.get("demo_control") != control or response.get("seed") != seed
                or response.get("shuffle_seed") != shuffle_seed):
            raise ValueError(f"Worker did not acknowledge the requested support/control: {response}")
        self.last_demo_metadata = response
        return response

    def validate_configuration(self, args):
        current = {name: str(getattr(args, name).resolve()) for name in self.configuration}
        if current != self.configuration or args.denoise_steps != self.denoise_steps:
            raise ValueError("A reused worker must keep its checkpoint, normalization, tokenizer, code and denoising configuration")

    def file_digest(self, path):
        """Hash immutable evidence once, checking size/mtime on subsequent trials."""
        key = str(path.resolve())
        stat = path.stat()
        identity = (stat.st_size, stat.st_mtime_ns)
        if key in self.artifact_digests:
            previous, value = self.artifact_digests[key]
            if previous != identity:
                raise ValueError(f"An artifact changed while its inference worker was live: {path}")
            return value
        value = digest(path)
        self.artifact_digests[key] = (identity, value)
        return value

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write('{"stop":true}\n')
                self.process.stdin.flush()
                self.process.wait(timeout=5)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        self.reader.join(timeout=5)
        self.process.stdin.close()
        self.process.stdout.close()
        self.stderr.close()


def observe(task):
    task._update_render()
    task.cameras.update_picture()
    native = task.cameras.get_rgb()
    rgb = np.stack([np.asarray(ImageOps.pad(Image.fromarray(native[name]["rgb"]),
        (224, 224), method=Image.Resampling.BILINEAR)) for name in ("head_camera", "left_camera")])
    state = np.asarray(task.robot.left_entity.get_qpos(), dtype=np.float32)
    if rgb.shape != (2, 224, 224, 3) or rgb.dtype != np.uint8 or state.shape != (9,) or not np.isfinite(state).all():
        raise ValueError("Unexpected native two-view RGB or nine-joint observation")
    return rgb, state


def validate_controls(actions, limits, max_velocity):
    controls = np.asarray(actions).reshape(-1, 18)
    position, velocity = controls[:, :9], controls[:, 9:]
    if not np.isfinite(controls).all():
        raise ValueError("Non-finite predicted controls")
    invalid = np.argwhere((position < limits[:, 0] - 1e-5) | (position > limits[:, 1] + 1e-5))
    if len(invalid):
        step, joint = invalid[0]
        raise ValueError(f"Rejected drive target: prefix step {step}, joint {joint}, "
                         f"value {position[step, joint]}, limits {limits[joint].tolist()}")
    if max_velocity is not None and np.abs(velocity).max() > max_velocity:
        raise ValueError(f"Rejected drive velocity exceeding {max_velocity}")
    return position, velocity


def apply(task, joints, position, velocity):
    for joint, q, dq in zip(joints, position, velocity, strict=True):
        joint.set_drive_target(float(q))
        joint.set_drive_velocity_target(float(dq))
    for entity in (task.robot.left_entity, task.robot.right_entity):
        entity.set_qf(entity.compute_passive_force(gravity=True, coriolis_and_centrifugal=True))
    # This metadata matches native gripper commands; drive targets are never clipped.
    task.robot.left_gripper_val = float(np.clip((position[7:9].mean() + .01) / .06, 0, 1))
    task._step_scene()


def run(args, worker=None):
    root = Path(__file__).resolve().parents[1]
    if worker is not None:
        worker.validate_configuration(args)
    evidence_digest = worker.file_digest if worker is not None else digest
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "queries").mkdir()
    (output / "actions").mkdir()
    (output / "frames").mkdir()
    (output / "source").mkdir()
    config = read(args.episode / "config.json")
    source = read(args.episode / "result.json")
    if not source.get("replay_success"):
        raise ValueError("The episode must be an accepted native replay attempt")
    config.update(save_data=False, record_control=False, save_path=str(output), need_plan=False)
    if args.target_index is not None:
        # Used only by the simulator's diagnostic and the benchmark scorer.
        # No target field is sent to the model worker.
        config["target_index"] = args.target_index
    scorer = contract = None
    if args.scorer.is_file() and args.acceptance_contract.is_file():
        spec = importlib.util.spec_from_file_location("roboicl_rollout_scorer", args.scorer)
        scorer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(scorer)
        contract = read(args.acceptance_contract)
        scorer.validate_contract(contract)
    elif args.formal_evaluation:
        raise ValueError("Formal evaluation requires the root scorer and frozen acceptance contract")
    write(output / "config.json", config)
    provenance = {}
    for path in (Path(__file__), root / "envs/roboicl_pick_place.py", root / "envs/_base_task.py", args.worker_script):
        shutil.copyfile(path, output / "source" / path.name)
        provenance[str(path.resolve())] = evidence_digest(path)
    for path in (args.demo, args.demo.with_name('observation_step.npy'), args.normalization, args.tokenizer, args.checkpoint / "complete.json",
                 args.checkpoint / "model.safetensors", args.episode / "config.json", args.episode / "result.json"):
        provenance[str(path.resolve())] = evidence_digest(path)
    if scorer is not None:
        for path in (args.scorer, args.acceptance_contract):
            shutil.copyfile(path, output / "source" / path.name)
            provenance[str(path.resolve())] = evidence_digest(path)
    receipt = dict(kind="formal_closed_loop_trial" if args.formal_evaluation else "engineering_closed_loop_check",
        receipt_label=args.receipt_label, formal_evaluation_requested=args.formal_evaluation,
        performance_evaluation=args.formal_evaluation, formal_success=False if args.formal_evaluation else None,
        checkpoint=str(args.checkpoint.resolve()),
        episode=str(args.episode.resolve()), demo=str(args.demo.resolve()),
        demo_control=args.demo_control, shuffle_seed=args.shuffle_seed,
        worker_reused=worker is not None,
        worker_log_directory=str(worker.output if worker is not None else output),
        policy_inputs=["Demo RGB", "query RGB", "measured query qpos"],
        target_index_evaluator_only=config["target_index"],
        destination_xy_evaluator_only=config.get("destination", [-.20, .08]),
        initial_state_verified=False,
        expert_clearance_executed=False, execute_tokens=args.execute_tokens,
        physics_steps_per_token=10, max_policy_steps=args.max_steps,
        requested_stability_steps=args.stability_steps, seed=args.seed, denoise_steps=args.denoise_steps,
        bound_controls=args.bound_controls, bounded_position_values=0, predicted_position_values=0,
        source_sha256=provenance, python=sys.version, status="starting", completed_policy_steps=0,
        completed_stability_steps=0, inference_calls=0, error=None)
    write(output / "receipt.json", receipt)
    task = None
    owns_worker = worker is None
    measurements, applied, durations = [], [], []
    try:
        sys.path.insert(0, str(root))
        from envs.roboicl_pick_place import roboicl_pick_place
        task = roboicl_pick_place()
        task.setup_demo(**config)
        initial = task.object_state()
        measurements.append(initial)
        reset_fields = ("object_poses", "left_qpos", "left_qvel", "right_qpos", "object_linear_velocity",
                        "object_angular_velocity", "gripper_command", "left_drive_position", "left_drive_velocity")
        reset_errors = {}
        for key in reset_fields:
            reference = np.asarray(source["replay_initial"][key], dtype=float)
            value = np.asarray(initial[key], dtype=float)
            tolerance = 1e-5 if key == "object_poses" else 1e-6
            if value.shape != reference.shape or not np.allclose(value, reference, atol=tolerance, rtol=0):
                raise ValueError(f"Initial {key} differs from the accepted native replay")
            reset_errors[key] = float(np.max(np.abs(value - reference)))
        initial_rgb, _ = observe(task)
        receipt.update(initial_state_verified=True, initial_state_max_abs_errors=reset_errors,
                       initial_query_rgb_sha256=hashlib.sha256(initial_rgb.tobytes()).hexdigest(),
                       destination_xy_evaluator_only=task.destination.tolist())
        if not np.isclose(task.scene.get_timestep(), .004, atol=1e-9, rtol=0):
            raise ValueError("dense180 requires the qualified 0.004 second physics timestep")
        joints = task.robot.left_entity.get_active_joints()
        expected = [f"panda_joint{i}" for i in range(1, 8)] + ["panda_finger_joint1", "panda_finger_joint2"]
        if [joint.name for joint in joints] != expected:
            raise ValueError("Unexpected active joint order")
        limits = np.asarray(task.robot.left_entity.get_qlimits(), dtype=float).copy()
        if limits.shape != (9, 2) or not np.isfinite(limits).all():
            raise ValueError("Expected nine finite joint limits")
        limits[7:9] = [-.01, .05]  # Native gripper drive range intentionally exceeds URDF physical limits.
        receipt.update(joint_names=expected, drive_position_limits=limits.tolist(),
            max_abs_drive_velocity=args.max_velocity, physics_timestep_s=float(task.scene.get_timestep()))
        if worker is None:
            worker = Worker(args, output)
        receipt["worker_demo_acknowledgement"] = worker.set_demo(args.demo, args.demo_control, args.seed, args.shuffle_seed)
        receipt["status"] = "running"
        while receipt["completed_policy_steps"] < args.max_steps:
            index = receipt["inference_calls"]
            rgb, state = observe(task)
            query = output / "queries" / f"{index:05d}.npz"
            np.savez_compressed(query, rgb=rgb, state=state)
            Image.fromarray(np.concatenate(rgb, axis=1)).save(output / "frames" / f"{task.physics_step:06d}.png")
            started = time.monotonic()
            actions = worker.predict(query, output / "actions" / f"{index:05d}.npy")
            durations.append(time.monotonic() - started)
            receipt["inference_calls"] += 1
            count = min(args.execute_tokens * 10, args.max_steps - receipt["completed_policy_steps"])
            prefix = actions[:args.execute_tokens].reshape(-1, 18)[:count].copy()
            if args.bound_controls:
                bounded = np.clip(prefix[:, :9], limits[:, 0], limits[:, 1])
                receipt['bounded_position_values'] += int(np.count_nonzero(bounded != prefix[:, :9]))
                prefix[:, :9] = bounded
            receipt['predicted_position_values'] += int(prefix[:, :9].size)
            position, velocity = validate_controls(prefix, limits, args.max_velocity)
            for q, dq in zip(position, velocity, strict=True):
                apply(task, joints, q, dq)
                applied.append(np.r_[q, dq])
                measurements.append(task.object_state())
                receipt["completed_policy_steps"] += 1
            write(output / "receipt.json", receipt)
        # Hold the last position with zero drive velocity; no new policy/expert action during settling.
        if args.stability_steps:
            hold = np.asarray([joint.get_drive_target()[0] for joint in joints])
            for _ in range(args.stability_steps):
                apply(task, joints, hold, np.zeros(9))
                applied.append(np.r_[hold, np.zeros(9)])
                measurements.append(task.object_state())
                receipt["completed_stability_steps"] += 1
        receipt["status"] = "completed"
    except Exception as error:
        receipt["status"] = "failed"
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            if measurements:
                xyz = np.asarray([row["positions"] for row in measurements])
                lift = xyz[:, :, 2] - xyz[0, :, 2]
                displacement = np.linalg.norm(xyz - xyz[0], axis=-1)
                lifted = np.argwhere(lift > .06)
                first_step = int(lifted[0, 0]) if len(lifted) else None
                choices = np.flatnonzero(lift[first_step] > .06).tolist() if first_step is not None else []
                target = config["target_index"]
                task.target_lifted = bool(lift[:, target].max() > .06)
                try:
                    native_success = bool(task.check_success())
                except Exception as error:
                    native_success = None
                    receipt["native_diagnostic_error"] = f"{type(error).__name__}: {error}"
                receipt.update(initial=measurements[0], final=measurements[-1],
                    first_lifted_object_indices=choices, first_lift_physics_step=first_step,
                    object_choice_correct=choices == [target], target_index_evaluator_only=target,
                    max_object_lift_m=lift.max(axis=0).tolist(),
                    max_object_displacement_m=displacement.max(axis=0).tolist(),
                    native_task_success=native_success,
                    native_success_lift_semantics="target exceeded initial height by 0.06 m at any observed physics step",
                    task_success=False,
                    inference_latency_s=durations)
                telemetry = dict(object_poses=np.asarray([r["object_poses"] for r in measurements]),
                    physics_step=np.asarray([r["physics_step"] for r in measurements]),
                    gripper_command=np.asarray([r["gripper_command"] for r in measurements]),
                    qpos=np.asarray([r["left_qpos"] for r in measurements]), qvel=np.asarray([r["left_qvel"] for r in measurements]),
                    object_linear_velocity=np.asarray([r["object_linear_velocity"] for r in measurements]),
                    object_angular_velocity=np.asarray([r["object_angular_velocity"] for r in measurements]),
                    object_height_m=xyz[:, :, 2], object_lift_m=lift, object_displacement_m=displacement,
                    applied_controls=np.asarray(applied, dtype=float).reshape(-1, 18))
                np.savez_compressed(output / "measurements.npz", **telemetry)
                if scorer is not None:
                    try:
                        score = scorer.score_rollout(receipt, telemetry, contract)
                        write(output / "score.json", score)
                        receipt.update(score)
                        receipt["task_success"] = bool(score["formal_success"])
                    except Exception as error:
                        receipt.update(status="failed", task_success=False,
                                       formal_success=False if args.formal_evaluation else None,
                                       scoring_error=f"{type(error).__name__}: {error}")
                else:
                    # Retain conservative object-choice diagnostics without a root scorer.
                    receipt["object_choice_correct"] = choices == [target] and np.flatnonzero((lift > .06).any(axis=0)).tolist() == [target]
                    receipt["scoring_unavailable"] = True
                try:
                    rgb, _ = observe(task)
                    Image.fromarray(np.concatenate(rgb, axis=1)).save(output / "frames" / f"{task.physics_step:06d}.png")
                except Exception as error:
                    receipt["final_frame_error"] = f"{type(error).__name__}: {error}"
            write(output / "receipt.json", receipt)
        finally:
            try:
                if worker is not None and owns_worker:
                    worker.close()
            finally:
                if task is not None and hasattr(task, "scene"):
                    task.close_env()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("episode", "output", "worker-python", "worker-script", "worker-source", "checkpoint", "normalization", "tokenizer", "demo"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--execute-tokens", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1000, help="Physics-step cap; formal trials require 5000")
    parser.add_argument("--stability-steps", type=int, default=0)
    parser.add_argument("--max-velocity", type=float, default=None, help="Optional rejection bound; never clips controls")
    parser.add_argument("--bound-controls", action="store_true", help="Project predicted positions to native drive limits and report every changed value; retain raw predictions")
    parser.add_argument("--worker-timeout", type=float, default=600)
    parser.add_argument("--receipt-label", default="engineering_only")
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument("--demo-control", choices=("correct", "masked", "last_frame", "shuffled"), default="correct")
    parser.add_argument("--shuffle-seed", type=int, default=20261003)
    parser.add_argument("--target-index", type=int, choices=(0, 1, 2), help="Evaluator-only target for a counterfactual Demo; never sent to the worker")
    parser.add_argument("--formal-evaluation", action="store_true", help="Require the complete v2 trial protocol and root scoring")
    repository = Path(__file__).resolve().parents[2]
    parser.add_argument("--scorer", type=Path, default=repository / "scripts/score_roboicl_rollout.py")
    parser.add_argument("--acceptance-contract", type=Path, default=repository / "experiments/policy_acceptance_v2_20261002.json")
    args = parser.parse_args()
    if not 1 <= args.execute_tokens <= 50 or args.max_steps <= 0 or args.stability_steps < 0:
        parser.error("Require 1..50 execute tokens, positive max steps, and nonnegative stability steps")
    if args.worker_timeout <= 0 or args.denoise_steps <= 0 or (args.max_velocity is not None and args.max_velocity <= 0):
        parser.error("Timeout, denoise steps, and any velocity bound must be positive")
    if args.formal_evaluation and (args.max_steps != 5000 or args.stability_steps != 250 or not args.bound_controls):
        parser.error("Formal evaluation requires --max-steps 5000 --stability-steps 250 --bound-controls")
    receipt = run(args)
    print(json.dumps({key: receipt[key] for key in ("status", "task_success", "object_choice_correct", "completed_policy_steps")}), flush=True)
    if receipt["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
