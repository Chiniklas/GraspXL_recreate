# Shadow Fixed Runners

This folder contains two PPO runners for the Shadow Hand fixed-base phase:

1. `runner.py`: the original multi-object trainer. It scans `rsc/<cat_name>` (default `mixed_train`), duplicates each object three times, and launches PPO with hundreds of parallel envs. Use it for large-scale training across all available meshes.

2. `runner_single_object.py`: a specialization for focusing on a single object. It requires `--object <folder_name>` (inside the target `--cat` category) and optional `--repeat N` to set the number of parallel envs for that object. Keep `N` high enough (e.g., 32+) if you want batches comparable to the original runner.

Both runners share the same CLI flags (`-c/--cfg`, `-e/--exp_name`, `-sd/--storedir`, etc.) and rely on the configs in `cfgs/`. After editing these scripts or configs, rerun `python setup.py develop` in the repo root if you modified C++ headers; otherwise Python changes take effect immediately.

## Examples

Large-scale training (original runner):
```bash
python raisimGymTorch/env/envs/shadow_fixed/runner.py \
  -c cfg_reg.yaml -e shadow_fixed_v1 -sd data_all -ln run1
```

Single object (needs existing folder `rsc/mixed_train/Mug_8556_handle`):
```bash
python raisimGymTorch/env/envs/shadow_fixed/runner_single_object.py \
  -c cfg_reg.yaml -e shadow_fixed_mug -sd data_all \
  --cat mixed_train --object Mug_8556_handle --repeat 64
```

## Quick Simulation / Visualization

Start `./raisimUnity/linux/raisimUnity.x86_64` (Auto-connect) before either method:

1. **Demo script (general objects, unseen sets supported):**
   ```bash
   python raisimGymTorch/env/envs/shadow_demo/demo.py \
     -w ../shadow_fixed_v1/2025-11-12-15-26-21/full_0_r.pt
   ```
   `shadow_demo` randomly samples objects from the configured set (e.g., `mixed_train`, ShapeNet test, etc.), so it’s useful for checking how a checkpoint generalizes beyond mugs.

2. **Mug-only viewer (new `viewer.py` in this folder):**
   ```bash
   python raisimGymTorch/env/envs/shadow_fixed/viewer.py \
     -w ../shadow_fixed_mug/2025-11-12-16-04-01/full_0_r.pt \
     --object Mug_8556_handle
   ```
   This script filters to folders whose names start with `Mug` and optionally locks to a specific mug via `--object`. Use it to inspect single-object runs.
