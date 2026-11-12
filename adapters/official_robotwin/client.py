"""Released-weight dual-arm benchmark client; no RoboICL single-arm conversion.

Derived from Zero-WAM's evaluation/robotwin/eval_policy_client_openpi.py.
The upstream license applies; RoboICL changes add bounded attempts and evidence.
"""
import argparse
import ast
from datetime import datetime
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import cv2
import imageio
import imageio_ffmpeg
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

# Resolve relative CLI resource paths before entering the native task directory.
ZERO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ZERO_ROOT))
robotwin_root = Path(os.environ.get("ROBOTWIN_ROOT", "/path/to/official/RoboTwin")).expanduser().resolve()
from evaluation.robotwin.geometry import euler2quat
from evaluation.robotwin.websocket_client_policy import WebsocketClientPolicy

def write_json(data: dict, fpath: Path) -> None:
    """Write data to a JSON file.

    Creates parent directories if they don't exist.

    Args:
        data (dict): The dictionary to write.
        fpath (Path): The path to the output JSON file.
    """
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)







def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance



def get_camera_config(camera_type):
    camera_config_path = os.path.join(
        robotwin_root, "task_config/_camera_config.yml"
    )

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def load_icl_human_video_map(map_path):
    map_path = os.path.abspath(map_path)
    spec = importlib.util.spec_from_file_location(
        "robotwin_icl_human_videos", map_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import ICL human video map: {map_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    mapping = getattr(module, "ROBOTWIN_ICL_HUMAN_VIDEOS", None)
    if not isinstance(mapping, dict):
        raise RuntimeError(
            f"{map_path} must define dict ROBOTWIN_ICL_HUMAN_VIDEOS"
        )
    return mapping


def select_icl_video(mapping, task_name, seed, episode_idx):
    videos = mapping.get(task_name, [])
    if not videos:
        raise RuntimeError(f"No ICL human video configured for {task_name!r}")
    digest = hashlib.md5(task_name.encode("utf-8")).hexdigest()
    rng = random.Random(int(seed) + int(digest[:8], 16) + int(episode_idx))
    return rng.choice(videos)


def resolve_icl_latent_path(video_path, latent_root):
    if not video_path or not latent_root:
        return ""
    video = Path(video_path)
    root = Path(latent_root)
    candidates = []

    parts = video.parts
    if "human_data" in parts:
        relative_parts = parts[parts.index("human_data") + 1 :]
        if relative_parts and relative_parts[0] == root.name:
            relative_parts = relative_parts[1:]
        candidates.append(root.joinpath(*relative_parts).with_suffix(".pth"))

        mirrored_parts = list(parts)
        mirrored_parts[mirrored_parts.index("human_data")] = "human_latents"
        candidates.append(Path(*mirrored_parts).with_suffix(".pth"))

    candidates.append(root / f"{video.stem}.pth")
    return next((str(path) for path in candidates if path.exists()), "")


def save_comparison_video(real_obs_list, imagined_video, action_history, save_path, fps=15):
    writer=imageio.get_writer(save_path,fps=fps,codec='libx264',quality=7)
    try:
        for obs in real_obs_list:
            imgs=[obs[k] for k in ['observation.images.cam_high','observation.images.cam_left_wrist','observation.images.cam_right_wrist']]
            height=imgs[0].shape[0]
            imgs=[cv2.resize(im,(round(im.shape[1]*height/im.shape[0]),height)) if im.shape[0]!=height else im for im in imgs]
            writer.append_data(np.concatenate(imgs,axis=1))
    finally: writer.close()

def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    save_root = usr_args["save_root"]
    policy_name = usr_args["policy_name"]
    video_guidance_scale = usr_args["video_guidance_scale"]
    action_guidance_scale = usr_args["action_guidance_scale"]
    icl_guidance_scale = usr_args["icl_guidance_scale"]
    icl_seed_value = usr_args.get("icl_seed")
    icl_seed = int(
        usr_args.get("seed", 0) if icl_seed_value is None else icl_seed_value
    )
    icl_video_mapping = load_icl_human_video_map(
        usr_args["icl_human_video_map"]
    )
    icl_latent_root = usr_args["icl_latent_root"]
    episode_offset = int(usr_args.get("episode_offset") or 0)
    instruction_type = 'seen'
    save_dir = None
    video_save_dir = None
    video_size = None

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting
    args["save_root"] = save_root
    args["icl_seed"] = icl_seed
    args["icl_video_mapping"] = icl_video_mapping
    args["icl_latent_root"] = icl_latent_root
    args["episode_offset"] = episode_offset

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    save_dir = Path(save_root) / "native_eval" / task_name / current_time.replace(":", "-")
    save_dir.mkdir(parents=True, exist_ok=True)

    if args["eval_video_log"]:
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]

    st_seed = usr_args.get("start_seed")
    # Match the Robotwin evaluation protocol used for the released checkpoint.
    st_seed = 100000 * (1 + seed) if st_seed is None else int(st_seed)
    suc_nums = []
    test_num = usr_args["test_num"]


    model = WebsocketClientPolicy(host="127.0.0.1", port=usr_args['port'])

    st_seed, suc_num = eval_policy(task_name,
                                   TASK_ENV,
                                   args,
                                   model,
                                   st_seed,
                                   test_num=test_num,
                                   video_size=video_size,
                                   instruction_type=instruction_type,
                                   save_visualization=True,
                                   video_guidance_scale=video_guidance_scale,
                                   action_guidance_scale=action_guidance_scale,
                                   icl_guidance_scale=icl_guidance_scale)
    suc_nums.append(suc_num)

    file_path = os.path.join(save_dir, f"_result.txt")
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    print(f"Data has been saved to {file_path}")

def format_obs(observation, prompt):
    return {
                "observation.images.cam_high": observation["observation"]["head_camera"]["rgb"], # H,W,3
                "observation.images.cam_left_wrist": observation["observation"]["left_camera"]["rgb"],
                "observation.images.cam_right_wrist": observation["observation"]["right_camera"]["rgb"],
                "observation.state": observation["joint_action"]["vector"],
                "task": prompt,
            }

def add_eef_pose(new_pose, init_pose):
    new_pose_R = R.from_quat(new_pose[3:7][None])
    init_pose_R = R.from_quat(init_pose[3:7][None])
    out_rot = (init_pose_R * new_pose_R).as_quat().reshape(-1)
    out_trans = new_pose[:3] + init_pose[:3]
    return np.concatenate([out_trans, out_rot, new_pose[7:8]])

def add_init_pose(new_pose, init_pose):
    left_pose = add_eef_pose(new_pose[:8], init_pose[:8])
    right_pose = add_eef_pose(new_pose[8:], init_pose[8:])
    return np.concatenate([left_pose, right_pose])

def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                video_size=None,
                instruction_type=None,
                save_visualization=False,
                video_guidance_scale=5.0,
                action_guidance_scale=5.0,
                icl_guidance_scale=5.0):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = int(args.get("episode_offset", 0))
    fixed_path = os.environ.get("FIXED_SMOKE_MANIFEST")
    fixed_rows = json.loads(Path(fixed_path).read_text(encoding="utf-8"))[task_name] if fixed_path else None
    succ_seed = now_id if os.environ.get("RESUME_OFFICIAL_EPISODES") == "1" else 0
    suc_test_seed_list = []


    now_seed = st_seed
    wall_started = time.monotonic()
    journal = Path(args["save_root"])/"attempts.jsonl"
    journal.parent.mkdir(parents=True, exist_ok=True)
    def record(event, **extra):
        with journal.open("a", encoding="utf-8") as f:
            f.write(json.dumps(dict(event=event, task=task_name, seed=now_seed, episode=now_id, time=time.time(), **extra), default=str)+"\n")
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True

    while succ_seed < test_num:
        if now_seed-st_seed >= int(os.environ.get("MAX_EXPERT_CANDIDATES", "1000")):
            raise RuntimeError("Finite expert candidate budget exhausted")
        if time.monotonic()-wall_started > float(os.environ.get("CLIENT_WALL_SECONDS", "21600")):
            raise TimeoutError("Client finite wall budget exhausted")
        if fixed_rows is not None:
            now_seed = fixed_rows[now_id]["seed"]
        record("expert_candidate")
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError as e:
                record("expert_unstable", error=str(e))
                if fixed_rows is not None: raise RuntimeError("Fixed smoke seed unstable; no replacement") from e
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                frames = traceback.extract_tb(e.__traceback__)
                native_empty_grasp = (
                    isinstance(e, AssertionError)
                    and str(e) == "target_pose cannot be None for move action."
                    and any(f.name == "grasp_actor" and f.filename.endswith("_base_task.py") for f in frames)
                    and any(f.name == "__init__" and f.filename.endswith("action.py") for f in frames)
                )
                TASK_ENV.close_env()
                args["render_freq"] = render_freq
                if native_empty_grasp and fixed_rows is None:
                    record("expert_infeasible", reason="native_empty_grasp_pose", error=str(e), traceback=traceback.format_exc())
                    now_seed += 1
                    continue
                record("infrastructure_failure", error=str(e))
                traceback.print_exc()
                raise

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
            record("expert_feasible")
        else:
            record("expert_infeasible")
            if fixed_rows is not None: raise RuntimeError("Fixed smoke seed infeasible; no replacement")
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        from state_snapshot import snapshot
        write_json(snapshot(TASK_ENV), Path(args['save_root'])/'initial_states'/f'{task_name}_{now_id}_{now_seed}.json')
        cameras = TASK_ENV.cameras
        def traced_pictures():
            cams = ([('left', cameras.left_camera), ('right', cameras.right_camera)] if cameras.collect_wrist_camera else []) + list(zip(cameras.static_camera_name, cameras.static_camera_list))
            for name, camera in cams:
                record('camera_begin', camera=name, step=TASK_ENV.take_action_cnt)
                camera.take_picture()
                record('camera_end', camera=name, step=TASK_ENV.take_action_cnt)
        cameras.update_picture = traced_pictures
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
        instruction = np.random.choice(results[0][instruction_type])
        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

        if TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    imageio_ffmpeg.get_ffmpeg_exe(),
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    f"{TASK_ENV.eval_video_path}/episode{now_id}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        succ = False

        prompt = TASK_ENV.get_instruction()
        if fixed_rows is not None and fixed_rows[now_id].get('prompt'):
            prompt = fixed_rows[now_id]['prompt']
            TASK_ENV.set_instruction(instruction=prompt)
        icl_video_path = select_icl_video(
            args["icl_video_mapping"],
            task_name,
            args["icl_seed"],
            now_id,
        )
        icl_latent_path = resolve_icl_latent_path(
            icl_video_path, args["icl_latent_root"]
        )
        if fixed_rows is not None and fixed_rows[now_id].get('demo_sha256'):
            assert hashlib.sha256(Path(icl_latent_path).read_bytes()).hexdigest() == fixed_rows[now_id]['demo_sha256']
        print(
            f"[ICL] task={task_name} episode={now_id} "
            f"video={icl_video_path} latent={icl_latent_path or 'MP4_FALLBACK'}",
            flush=True,
        )
        ret = model.infer(
            dict(
                reset=True,
                prompt=prompt,
                save_visualization=save_visualization,
                use_icl=True,
                icl_video_path=icl_video_path,
                icl_latent_path=icl_latent_path,
                video_guidance_scale=video_guidance_scale,
                icl_guidance_scale=icl_guidance_scale,
            )
        )

        record("policy_started", prompt=prompt, demo_video=icl_video_path, demo_latent=icl_latent_path, demo_sha256=hashlib.sha256(Path(icl_latent_path).read_bytes()).hexdigest(), target_text_cfg=video_guidance_scale, icl_cfg=icl_guidance_scale)
        first = True
        full_obs_list = []
        gen_video_list = []
        full_action_history = []

        initial_obs = TASK_ENV.get_obs()
        episode_dir = Path(args["save_root"])/"episodes"/task_name/f"{now_id:03d}_{now_seed}"
        episode_dir.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(episode_dir/"initial.npz",head=initial_obs["observation"]["head_camera"]["rgb"],left=initial_obs["observation"]["left_camera"]["rgb"],right=initial_obs["observation"]["right_camera"]["rgb"],state=initial_obs["joint_action"]["vector"])
        inint_eef_pose = initial_obs['endpose']['left_endpose'] + \
        [initial_obs['endpose']['left_gripper']] + \
        initial_obs['endpose']['right_endpose'] + \
        [initial_obs['endpose']['right_gripper']]
        inint_eef_pose = np.array(inint_eef_pose, dtype=np.float64)
        initial_formatted_obs = format_obs(initial_obs, prompt)
        full_obs_list.append(initial_formatted_obs)
        first_obs = None
        while TASK_ENV.take_action_cnt<TASK_ENV.step_lim:
            if first:
                observation = TASK_ENV.get_obs()
                first_obs = format_obs(observation, prompt)

            ret = model.infer(dict(obs=first_obs, prompt=prompt, save_visualization=save_visualization, video_guidance_scale=video_guidance_scale, action_guidance_scale=action_guidance_scale)) #(TASK_ENV, model, observation)
            action = ret['action']
            assert np.isfinite(action).all() and action.ndim==3, 'Invalid policy action'
            if 'video' in ret:
                imagined_video = ret['video']
                gen_video_list.append(imagined_video)
            key_frame_list = []

            assert action.shape[2] % 4 == 0
            action_per_frame = action.shape[2] // 4

            start_idx = 1 if first else 0
            for i in range(start_idx, action.shape[1]):
                for j in range(action.shape[2]):
                    raw_action_step = action[:, i, j].flatten()
                    full_action_history.append(raw_action_step)

                    ee_action = action[:, i, j]
                    if action.shape[0] == 14:
                        ee_action = np.concatenate([
                            ee_action[:3],
                            euler2quat(ee_action[3], ee_action[4], ee_action[5]),
                            ee_action[6:10],
                            euler2quat(ee_action[10], ee_action[11], ee_action[12]),
                            ee_action[13:14]
                        ])
                    elif action.shape[0] == 16:
                        ee_action =  add_init_pose(ee_action, inint_eef_pose)
                        ee_action = np.concatenate([
                            ee_action[:3],
                            ee_action[3:7] / np.linalg.norm(ee_action[3:7]),
                            ee_action[7:11],
                            ee_action[11:15] / np.linalg.norm(ee_action[11:15]),
                            ee_action[15:16]
                        ])
                    else:
                        raise NotImplementedError
                    record('action_begin', step=TASK_ENV.take_action_cnt)
                    TASK_ENV.take_action(ee_action, action_type='ee')
                    record('action_end', step=TASK_ENV.take_action_cnt)

                    if (j+1) % action_per_frame == 0:
                        obs = format_obs(TASK_ENV.get_obs(), prompt)
                        full_obs_list.append(obs)
                        key_frame_list.append(obs)

            first = False

            model.infer(dict(obs = key_frame_list, compute_kv_cache=True, imagine=False, save_visualization=save_visualization, state=action))

            if TASK_ENV.eval_success:
                succ = True
                break


        vis_dir = Path(args['save_root']) / f'stseed-{st_seed}' / 'visualization' / task_name
        vis_dir.mkdir(parents=True, exist_ok=True)
        video_name = f"{now_id}_{prompt.replace(' ', '_')}_{succ}.mp4"
        out_img_file = vis_dir / video_name
        save_comparison_video(
            real_obs_list=full_obs_list,
            imagined_video=None, #gen_video_list,
            action_history=full_action_history,
            save_path=str(out_img_file),
            fps=15 # Suggest adjusting fps based on simulation step
        )
        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        np.save(episode_dir/"predicted_actions.npy",np.asarray(full_action_history))
        write_json(dict(task=task_name,seed=now_seed,episode=now_id,success=bool(succ),actions=int(TASK_ENV.take_action_cnt),predicted_steps=len(full_action_history),prompt=prompt,demo_latent=icl_latent_path,video=str(out_img_file)),episode_dir/"result.json")
        record("policy_finished", success=bool(succ), actions=int(TASK_ENV.take_action_cnt), result=str(episode_dir/"result.json"),video=str(out_img_file))
        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        save_dir = Path(args['save_root']) / f'stseed-{st_seed}' / 'metrics' / task_name
        save_dir.mkdir(parents=True, exist_ok=True)
        out_json_file = save_dir / 'res.json'
        write_json({
          "succ_num": float(TASK_ENV.suc),
          "total_num": float(TASK_ENV.test_num),
          "succ_rate": float(TASK_ENV.suc / TASK_ENV.test_num),
        }, out_json_file)

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        now_seed += 1

    return now_seed, TASK_ENV.suc


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    parser.add_argument("--port", type=int, default=8000, help='remote policy socket port.')
    parser.add_argument("--save_root", type=str, default="results/default_vis_path")
    parser.add_argument("--video_guidance_scale", type=float, default=-1.0)
    parser.add_argument("--action_guidance_scale", type=float, default=1.0)
    parser.add_argument("--icl_guidance_scale", type=float, default=5.0)
    parser.add_argument(
        "--icl_human_video_map",
        type=str,
        default=str(ZERO_ROOT / "evaluation/robotwin/robotwin_icl_human_videos.py"),
    )
    parser.add_argument("--icl_latent_root", type=str, required=True)
    parser.add_argument("--icl_seed", type=int, default=None)
    parser.add_argument("--start_seed", type=int, default=None)
    parser.add_argument("--episode_offset", type=int, default=0)
    parser.add_argument("--test_num", type=int, default=100)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = ast.literal_eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    config.update(
        {
            key: value
            for key, value in vars(args).items()
            if key not in {"config", "overrides"}
        }
    )

    return config


if __name__ == "__main__":

    usr_args = parse_args_and_config()
    for key in ("config", "save_root", "icl_human_video_map", "icl_latent_root"):
        usr_args[key] = str(Path(usr_args[key]).expanduser().resolve())
    fixed = os.environ.get("FIXED_SMOKE_MANIFEST")
    if fixed:
        os.environ["FIXED_SMOKE_MANIFEST"] = str(Path(fixed).expanduser().resolve())
    if not robotwin_root.is_dir():
        raise FileNotFoundError("Set ROBOTWIN_ROOT to the prepared official benchmark checkout")
    sys.path.insert(0, str(robotwin_root))
    os.chdir(robotwin_root)
    from envs import CONFIGS_PATH
    from envs.utils.create_actor import UnStableError
    from description.utils.generate_episode_instructions import generate_episode_descriptions
    main(usr_args)
