"""Render saved policy controls at 25 FPS and verify the original physics trace."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageOps

from rollout_roboicl_policy import apply, digest, observe, read, write


def run(args):
    source = args.trial.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    original = read(source / "receipt.json")
    if original["status"] != "completed":
        raise ValueError("Only completed policy trials can be replayed")
    config = read(source / "config.json")
    config.update(save_data=False, record_control=False, save_path=str(output), need_plan=False)
    trace = np.load(source / "measurements.npz", allow_pickle=False)
    controls = trace["applied_controls"]
    if controls.shape != (5250, 18) or not np.isfinite(controls).all():
        raise ValueError("Expected the complete 5000-step policy plus 250-step hold")
    fields = dict(object_poses="object_poses", qpos="left_qpos", qvel="left_qvel",
                  object_linear_velocity="object_linear_velocity",
                  object_angular_velocity="object_angular_velocity", gripper_command="gripper_command",
                  physics_step="physics_step")
    receipt = dict(status="running", source_trial=str(source), fps=25, physics_dt=.004,
                   frames=0, width=1280, height=480, source_steps=[], policy_rerun=False,
                   interpolation=False, max_abs_error={k: 0.0 for k in fields},
                   original_query_frame_checks=0, started_unix=time.time(),
                   source_sha256={str(p): digest(p) for p in
                       (source / "measurements.npz", source / "receipt.json", source / "config.json",
                        Path(__file__), Path(__file__).with_name("rollout_roboicl_policy.py"))})
    write(output / "receipt.json", receipt)
    task = encoder = None
    try:
        root = Path(__file__).resolve().parents[1]
        os.chdir(root)
        sys.path.insert(0, str(root))
        from envs.roboicl_pick_place import roboicl_pick_place

        task = roboicl_pick_place()
        task.setup_demo(**config)
        if not np.isclose(task.scene.get_timestep(), .004, rtol=0, atol=1e-9):
            raise ValueError("Physics timestep changed")
        joints = task.robot.left_entity.get_active_joints()
        if [j.name for j in joints] != original["joint_names"]:
            raise ValueError("Joint order changed")
        command = [str(args.ffmpeg), "-hide_banner", "-loglevel", "error", "-n",
                   "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "1280x480",
                   "-framerate", "25", "-i", "pipe:0", "-an", "-c:v", "libx264",
                   "-threads", "2", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags",
                   "+faststart", str(output / "video.mp4")]
        receipt["encoder_command"] = command
        with (output / "ffmpeg.log").open("w", encoding="utf-8") as log:
            encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=log, stderr=log,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            for step in range(len(controls) + 1):
                current = task.object_state()
                for saved, live in fields.items():
                    expected = trace[saved][step]
                    value = np.asarray(current[live])
                    error = float(np.max(np.abs(value - expected)))
                    receipt["max_abs_error"][saved] = max(receipt["max_abs_error"][saved], error)
                    if not np.array_equal(value, expected):
                        raise ValueError(f"Replay differs at step {step}, field {saved}, error {error}")
                if step % 10 == 0 and step < len(controls):
                    rgb, _ = observe(task)
                    if step % 100 == 0 and step < 5000:
                        old = np.asarray(Image.open(source / "frames" / f"{step:06d}.png"))
                        if not np.array_equal(np.concatenate(rgb, axis=1), old):
                            raise ValueError(f"Original query image differs at step {step}")
                        receipt["original_query_frame_checks"] += 1
                    native = task.cameras.get_rgb()
                    frames = [np.asarray(ImageOps.pad(Image.fromarray(native[name]["rgb"]),
                              (640, 480), method=Image.Resampling.BILINEAR))
                              for name in ("head_camera", "left_camera")]
                    combined = np.concatenate(frames, axis=1)
                    if step == 0:
                        Image.fromarray(combined).save(output / "poster.png")
                    encoder.stdin.write(combined.tobytes())
                    receipt["frames"] += 1
                    receipt["source_steps"].append(step)
                if step < len(controls):
                    # Policy positions and native get_drive_target() hold positions are float32.
                    # The hold's zero velocities are float64. Preserve the original arithmetic.
                    position = controls[step, :9].astype(np.float32)
                    velocity_dtype = np.float32 if step < original["completed_policy_steps"] else np.float64
                    velocity = controls[step, 9:].astype(velocity_dtype)
                    apply(task, joints, position, velocity)
            encoder.stdin.close()
            if encoder.wait(timeout=60) != 0:
                raise RuntimeError("Video encoding failed; see ffmpeg.log")
        receipt.update(status="completed", trajectory_bitwise_identical=True,
                       verified_states=len(controls) + 1, video_seconds=receipt["frames"] / 25,
                       video_sha256=digest(output / "video.mp4"))
    except BaseException as error:
        receipt.update(status="failed", error=repr(error))
        raise
    finally:
        if encoder is not None and encoder.poll() is None:
            encoder.kill()
            encoder.wait()
        if task is not None and hasattr(task, "scene"):
            task.close_env()
        trace.close()
        receipt["finished_unix"] = time.time()
        write(output / "receipt.json", receipt)
    print(json.dumps({k: receipt[k] for k in ("status", "frames", "video_seconds", "max_abs_error")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("trial", "output", "ffmpeg"):
        parser.add_argument("--" + key, type=Path, required=True)
    run(parser.parse_args())
