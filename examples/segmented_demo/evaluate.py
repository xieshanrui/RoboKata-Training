"""Finite, single-GPU segmented evaluation with a separate simulator interpreter.

One persistent model serves fresh simulator processes. All inputs and attempts
are recorded; policy failures are not retried. No SSH or local service is used.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
from protocol import receive, send
from pipe_client import checked_receive


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def stop(process):
    if process is not None and process.poll() is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        process.wait(timeout=30)


def check_execution(folder, receipt):
    predictions = [np.load(p, allow_pickle=False) for p in sorted(folder.glob("prediction_*.npy"))]
    if not predictions or any(p.shape != (50, 10) for p in predictions):
        raise ValueError("Invalid prediction chunks")
    expected = np.concatenate(predictions)
    with np.load(folder / "policy_recording.npz", allow_pickle=False) as z:
        if len(z["action"]) != receipt["actions_executed"] or not np.array_equal(z["observation_step"], np.arange(len(z["action"])) * 5):
            raise ValueError("Recorded50Hz action clock differs")
        if len(z["action"]) < len(expected) or not np.array_equal(z["action"][:len(expected)], expected):
            raise ValueError("Execute50 did not consume every predicted token")
    with np.load(folder / "queries.npz", allow_pickle=False) as z:
        if not np.array_equal(z["physics_step"], np.arange(len(predictions)) * 250):
            raise ValueError("Feedback cadence differs")
    write(folder / "execution50_verified.json", dict(passed=True, prediction_calls=len(predictions),
          executed_prediction_actions=len(expected), physics_steps_per_call=250, execute_tokens=50, action_hz=50))


def initial_state(folder):
    with np.load(folder / "initial_rgb.npz", allow_pickle=False) as z:
        state = {k: z[k].copy() for k in z.files}
    initial = read(folder / "initial_state.json")
    for key in ("object_poses", "left_qpos", "left_qvel", "left_drive_position", "left_drive_velocity", "object_linear_velocity", "object_angular_velocity"):
        state[key] = np.asarray(initial[key])
    return state


def main(a):
    package = Path(__file__).resolve().parent
    openpi = package.parents[1]
    bundle, output = a.bundle.resolve(), a.output.resolve()
    source = (a.source or openpi).resolve()
    robotwin = a.robotwin.resolve()
    # A venv's Python is usually a symlink. Resolving it selects the base
    # interpreter and loses the dedicated environment's installed packages.
    a.sim_python = a.sim_python.absolute()
    manifest = read(bundle / "bundle.json")
    if manifest["kind"] != "segmented_eval":
        raise ValueError("Expected a sealed segmented evaluation bundle")
    for name, entry in manifest["files"].items():
        p = (bundle / name).resolve()
        if not p.is_relative_to(bundle) or p.stat().st_size != entry["bytes"] or sha(p) != entry["sha256"]:
            raise ValueError("Bundle identity differs: " + name)
    protocol = read(bundle / "protocol.json")
    if protocol.get("execute_tokens") != 50 or protocol.get("action_hz") != 50:
        raise ValueError("This entry requires the frozen execute50 / 50Hz protocol")
    panel = read(bundle / "panel.json")["trials"]
    if a.ids:
        selected = set(a.ids)
        panel = [t for t in panel if t["id"] in selected]
        if {t["id"] for t in panel} != selected:
            raise ValueError("Diagnostic ID not found")
    if not panel or a.wall_seconds <= 0 or a.episode_seconds <= 0:
        raise ValueError("Empty panel or unbounded runtime")
    if any(t.get("condition") != "correct" for t in panel):
        raise ValueError("This entry evaluates the correct-Demo panel only")
    marker = read(a.checkpoint / "complete.json")
    identity = dict(bundle_sha256=sha(bundle / "bundle.json"), checkpoint_sha256=marker["model_sha256"],
                    checkpoint_step=marker["step"], ids=[t["id"] for t in panel], execute_tokens=50,
                    worker_sha256=sha(package / "worker.py"), rollout_sha256=sha(robotwin / "scripts/eval_segmented.py"),
                    model_source=str(source), protocol_sha256=sha(package / "protocol.py"))
    if a.source_amendment:
        identity['source_amendment_sha256'] = sha(a.source_amendment)
    if a.resume:
        if read(output / "identity.json") != identity:
            raise ValueError("Resume identity differs")
    else:
        output.mkdir(parents=True, exist_ok=False)
        write(output / "identity.json", identity)
    # Inspect the explicitly selected GPU; never choose or preempt another job.
    raw = subprocess.check_output(["nvidia-smi", "-i", str(a.gpu), "--query-gpu=uuid,pci.bus_id,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], encoding="utf-8").strip()
    uuid, pci, memory, utilization = [x.strip() for x in raw.split(",")]
    # Utilization is sampled over a prior window: just-exited training may show
    # 100% after its context and memory have gone. Recheck once, never preempt.
    if int(memory) < 100 and int(utilization) >= 5:
        time.sleep(3)
        raw = subprocess.check_output(["nvidia-smi", "-i", str(a.gpu), "--query-gpu=uuid,pci.bus_id,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], encoding="utf-8").strip()
        uuid, pci, memory, utilization = [x.strip() for x in raw.split(",")]
    if int(memory) >= 100 or int(utilization) >= 5:
        raise RuntimeError("Requested GPU is occupied: " + raw)
    if shutil.disk_usage(output).free < a.minimum_free_gib * 1024**3:
        raise RuntimeError("Insufficient storage reserve")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    for key, relative in {"TMPDIR":"tmp", "XDG_CACHE_HOME":"cache/xdg", "CUDA_CACHE_PATH":"cache/cuda", "TORCH_EXTENSIONS_DIR":"cache/torch_extensions", "TRITON_CACHE_DIR":"cache/triton", "WARP_CACHE_PATH":"cache/warp", "CUROBO_CACHE_DIR":"cache/curobo"}.items():
        path=output/relative;path.mkdir(parents=True,exist_ok=True);env[key]=str(path)
    # For machines needing a private Vulkan loader, pass its directory explicitly.
    if a.vulkan_lib:
        env["LD_LIBRARY_PATH"] = str(a.vulkan_lib.resolve()) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    sim_env = dict(env)
    if a.sim_dependencies:
        sim_env['PYTHONPATH'] = str(a.sim_dependencies.resolve()) + os.pathsep + env.get('PYTHONPATH', '')
    probe = "import sapien,torch,json; r=sapien.SapienRenderer();e=sapien.Engine();e.set_renderer(r);s=e.create_scene();print(json.dumps(dict(render_pci=s.render_system.device.pci_string,cuda_uuid=str(torch.cuda.get_device_properties(0).uuid))))"
    p = subprocess.run([str(a.sim_python), "-c", probe], env=sim_env, capture_output=True, encoding="utf-8", timeout=90)
    if p.returncode:
        raise RuntimeError(p.stderr)
    device = json.loads(p.stdout.splitlines()[-1])
    if device["render_pci"].lower().replace("00000000:", "0000:") != pci.lower().replace("00000000:", "0000:") or device["cuda_uuid"].lower() != uuid.removeprefix("GPU-").lower():
        raise RuntimeError("CUDA/render GPU mismatch")
    write(output / "device.json", dict(device, gpu=a.gpu))
    model = child = None
    deadline = time.monotonic() + a.wall_seconds
    results, families = [], {}
    def expire():
        stop(child)
        stop(model)
    timer = threading.Timer(a.wall_seconds, expire)
    timer.start()
    try:
        cmd = [sys.executable, "-u", str(package / "worker.py"), "--source", str(source), "--checkpoint", str(a.checkpoint.resolve()),
               "--eval-data", str(bundle / "eval_data"), "--training-data", str(a.training_data.resolve()), "--tokenizer", str(a.tokenizer.resolve())]
        if a.source_amendment:
            cmd += ["--source-amendment", str(a.source_amendment.resolve())]
        with (output / "model.log").open("a", encoding="utf-8") as log:
            model = subprocess.Popen(cmd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log)
            ready, _ = checked_receive(model.stdout)
            if ready["checkpoint_sha256"] != marker["model_sha256"] or ready["execute_tokens"] != 50 or ready["step"] != marker["step"]:
                raise RuntimeError("Wrong worker identity")
            write(output / "worker_identity.json", ready)
            for trial in panel:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Finite run deadline")
                out = output / "runs" / trial["id"]
                if out.exists():
                    if not a.resume or not (out / "execution50_verified.json").exists():
                        raise RuntimeError("Retained partial attempt needs review; no automatic retry: " + trial["id"])
                    receipt = read(out / "receipt.json")
                    check_execution(out, receipt)
                else:
                    if shutil.disk_usage(output).free < a.minimum_free_gib * 1024**3:
                        raise RuntimeError("Storage reserve exhausted")
                    cmd = [str(a.sim_python), "-X", "utf8", "-B", str(robotwin / "scripts/eval_segmented.py"), "--bundle", str(bundle),
                           "--trial", trial["id"], "--output", str(out), "--openpi", str(openpi), "--checkpoint-step", str(marker["step"])]
                    if a.curobo_source:cmd += ["--curobo-source", str(a.curobo_source.resolve())]
                    if a.sim_dependencies:cmd += ["--sim-dependencies", str(a.sim_dependencies.resolve())]
                    with (output / (trial["id"] + ".log")).open("x", encoding="utf-8") as log:
                        child = subprocess.Popen(cmd, cwd=robotwin, env=sim_env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log)
                        timeout = threading.Timer(min(a.episode_seconds, deadline-time.monotonic()), expire)
                        timeout.start()
                        try:
                            while True:
                                try: meta, arrays = receive(child.stdout)
                                except EOFError: break
                                if meta["kind"] == "identity":answer, values = ready, {}
                                else:
                                    send(model.stdin, meta, **arrays)
                                    answer, values = receive(model.stdout)
                                send(child.stdin, answer, **values)
                            rc=child.wait(timeout=30)
                        finally:
                            timeout.cancel();stop(child)
                    if not (out / "receipt.json").exists():
                        raise RuntimeError("Simulator ended without receipt")
                    receipt=read(out / "receipt.json")
                    if rc or receipt.get("infrastructure_failure") or not receipt["finished"]:
                        raise RuntimeError(receipt.get("error", "Simulator infrastructure failure"))
                    check_execution(out,receipt)
                if receipt["panel_sha256"] != sha(bundle / "panel.json") or not receipt["initial_verified"]:
                    raise RuntimeError("Scene identity differs")
                family=trial["query_record"]["initial_family_id"];state=initial_state(out)
                if family in families and not all(np.array_equal(v, families[family][k]) for k,v in state.items()):
                    raise RuntimeError("Same-family initial state or RGB differs")
                families[family]=state
                results.append(dict(id=trial["id"],split=trial["split"],metrics=receipt["metrics"]))
                write(output / "results.json",dict(rows=results,completed=len(results),target=len(panel)))
            write(output / "complete.json",dict(completed=True,ids=identity["ids"],cases=len(results),checkpoint_step=marker["step"],
                  same_family_reset_verified=True,diagnostic_subset=bool(a.ids or manifest.get("diagnostic_subset")),policy_outcome_retries=0))
    except BaseException as e:
        write(output / "failure.json",dict(error=repr(e),completed=len(results)))
        raise
    finally:
        timer.cancel();stop(child);stop(model)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("bundle","checkpoint","training-data","tokenizer","robotwin","sim-python","output"):
        p.add_argument("--"+key,type=Path,required=True)
    for key in ("source","curobo-source","sim-dependencies","vulkan-lib","source-amendment"):p.add_argument("--"+key,type=Path)
    p.add_argument("--gpu",type=int,required=True);p.add_argument("--ids",nargs="+");p.add_argument("--resume",action="store_true")
    p.add_argument("--wall-seconds",type=float,default=21600);p.add_argument("--episode-seconds",type=float,default=900)
    p.add_argument("--minimum-free-gib",type=float,default=20)
    main(p.parse_args())
