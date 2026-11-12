# Installation

## Zero-WAM Environment

The tested environment uses Python 3.10, PyTorch 2.9.0, and CUDA 12.6.

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  torch==2.9.0 \
  torchvision==0.24.0 \
  torchaudio==2.9.0 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt --no-build-isolation
```

The post-training dependencies, including LeRobot 0.3.3, are included in
`requirements.txt`.

## Robotwin Environment

Robotwin evaluation requires a separate Robotwin 2.0 environment. The server
and client may use different Python environments, but they must run on the same
machine so the client can connect to the local inference ports.

```bash
git clone https://github.com/RoboTwin-Platform/RoboTwin.git Robotwin
cd Robotwin
git checkout 2eeec322
```

Install the Vulkan packages and Robotwin dependencies, then download the
Robotwin assets by following the official installation guide:

https://robotwin-platform.github.io/doc/usage/robotwin-install.html

Set the repository location before running an evaluation client:

```bash
export ROBOTWIN_ROOT=/path/to/your/Robotwin
```

## Zero-WAM Evaluation Protocol

To more appropriately assess generalization to unseen tasks, we revise the
success criteria for `move_stapler_pad` and `stamp_seal`:

- `move_stapler_pad`: success requires a released stapler to contact and lie within the pad area, regardless of yaw.
- `stamp_seal`: the target-plane position tolerance is increased from `1 cm` to `3 cm`.

After completing the standard Robotwin installation, back up the two upstream
task files and copy the Zero-WAM versions into the Robotwin environment:

```bash
cp -n "${ROBOTWIN_ROOT}/envs/move_stapler_pad.py" "${ROBOTWIN_ROOT}/envs/move_stapler_pad.py.upstream"
cp -n "${ROBOTWIN_ROOT}/envs/stamp_seal.py" "${ROBOTWIN_ROOT}/envs/stamp_seal.py.upstream"
cp /path/to/Zero-WAM/evaluation/robotwin/envs/move_stapler_pad.py "${ROBOTWIN_ROOT}/envs/move_stapler_pad.py"
cp /path/to/Zero-WAM/evaluation/robotwin/envs/stamp_seal.py "${ROBOTWIN_ROOT}/envs/stamp_seal.py"
```

The `.upstream` files preserve the original success criteria used for the
standard in-domain Robotwin evaluation of these two tasks.

See `README.md` for the training and evaluation commands.
