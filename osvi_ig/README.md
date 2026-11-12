# OSVI-WM on IG-10K C15: native joint actions

The user selected each dataset's native recorded action representation.
RoboTwin therefore uses its16D EEF action column; IG uses its16D absolute
dual-arm joint target/gripper action column. No FK/IK-generated labels are
used. This IG version supersedes the finite EEF conversion prototype, which
had no formal updates. Keep its diagnostics only as historical evidence.

This is an adaptation of official OSVI-WM, not an official IG benchmark
result. The output still contains1-through-5 waypoint groups, but each
IG waypoint is a joint-action vector. Per-arm native SoftDTW interpolates
the seven joint channels and holds the final gripper channel constant per
segment. The world model and RGB demonstration pathway are retained.

`prepare` reads the frozen C15 training manifest:3000 robot and750 human
episodes from15 tasks. It checks original Parquet actions against the prior
verified cache, preserves30Hz timestamps and concatenated-video offsets,
and samples6 Query/10 human RGB frames and50 action-trajectory targets.
Mean/std (floor0.01) are fitted over all original training action rows,
before sampling. Normalization is invertible and unclipped. Smoke statistics
are separate; formal training always refits all3000 training episodes.

Pairing samples a human RGB clip from the same task using an epoch/query
seed. Task IDs only select pairs; human actions/poses/text never enter the
model. The trainer adds independent clip-consistent ColorJitter, retains
asymmetric mixup and EMA, and omits geometric flips/crops.

Train with `python -m osvi_robotwin.train --dataset ig10k`. The first budget
is72000 updates, actual single-GPU batch64, BF16, LR2e-4 with OneCycle,
no recomputation/accumulation. It is an initial budget, not a convergence or
equal-quality claim. Initial selection is the frozen final update; a
validation-based selection rule must be fixed before scoring candidates.
Keep every5000 updates and the final72000, plus rolling recovery. The previous
10000-update run is retained as a superseded short run; this budget starts from
the original initialization with all schedules set to72000 from the beginning.

`policy.Policy` rejects other action representations and denormalizes to
native16D joint commands directly. The native simulator remains responsible
for joint PD execution. The rollout cadence, limits and validation/test
cases still require a frozen closed-loop evaluation protocol; a passed
training/inference check does not provide task success rates.
