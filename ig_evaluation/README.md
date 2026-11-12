# IG closed-loop baseline evaluation

`python -m ig_evaluation.runner --config CONFIG --indices INDICES --output OUT`
reuses the frozen IG environment factory, state/RGB processor and rollout.
The external run contract fixes the five unseen tasks, L0–L3, ten episodes per
task/level, seeds100000–100009 and500 native `pd_joint_pos` control steps.
Native `success_once` and `success_at_end` remain unchanged. No target-task
parameter update or outcome-based checkpoint/case selection is performed.

LingBot text uses the preselected epoch60 native export, the first official
human task description and the training action statistics. Its native server
retains25 video/50 action denoising steps, guidance5/1, causal VAE/history and
16 initial then32 subsequent actions. Actual Query RGB is sampled every four
executed control steps. The executed commands populate action history, including
native gripper limits; no future Query image enters the model. Optional upstream
debug tensor dumps are disabled. Initial joint extraction delegates to IG's
processor. A64-step real-simulator startup verification covered three prediction
chunks; the root run records the separate inference and metadata-writing checks.

OSVI uses ten human RGB frames from each selected recording, preserving video
offsets. Its native-joint adapter predicts five waypoints and executes12 linear
commands per waypoint, holding the gripper target over each segment. This is a
model-specific controller adaptation, not an official published OSVI IG result.
Its final-checkpoint simulation verification is required before formal scoring.

Generated configs, source snapshots, input hashes, finite controllers, videos,
actions and run-specific evidence belong in ignored artifacts/outputs. Completed
cases are immutable; a smoke rollout never supplies a benchmark score. A raw
terminal video frame can reflect the official vector environment's auto-reset
when no final RGB is available; success uses native episode metrics.
