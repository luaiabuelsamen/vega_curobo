"""Ignore link pairs whose fitted spheres already overlap at rest.

The automatic sphere fit is conservative: neighbouring links end up with spheres
that intersect even in the robot's rest pose, and cuRobo then treats the home
configuration as a self-collision and refuses to plan from it.

This samples the rest pose plus small perturbations and ignores only pairs that
overlap there. Collisions that appear during motion are left intact, so this
loosens the model where it is wrong rather than everywhere.

    python scripts/relax_self_collision.py
"""
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg  # noqa: E402
from curobo.types import JointState  # noqa: E402

from vega_curobo.config import ROBOT_CONFIG, load_robot_config  # noqa: E402

SAMPLES = 40
JITTER = 0.1        # radians


def main():
    config = load_robot_config()
    kinematics = config["kinematics"]
    links = kinematics["collision_link_names"]
    sphere_counts = [len(kinematics["collision_spheres"][link]) for link in links]
    sphere_to_link = np.concatenate([[i] * n for i, n in enumerate(sphere_counts)])

    # Self-collision checking off, so we can inspect the spheres ourselves.
    far_away = {"cuboid": {"none": {"dims": [0.01] * 3, "pose": [0, 0, -5, 1, 0, 0, 0]}}}
    planner = MotionPlanner(MotionPlannerCfg.create(
        robot=config, scene_model=far_away,
        collision_cache={"obb": 10, "mesh": 2}, self_collision_check=False))

    rest = planner.default_joint_state.position.clone()
    rng = np.random.default_rng(0)
    overlapping = set()

    for sample in range(SAMPLES):
        q = rest.clone()
        if sample > 0:
            q += torch.tensor(rng.uniform(-JITTER, JITTER, q.shape[-1]),
                              device=q.device, dtype=q.dtype)
        state = JointState.from_position(q.unsqueeze(0), joint_names=planner.joint_names)
        spheres = planner.kinematics.compute_kinematics(state).robot_spheres
        spheres = spheres.reshape(-1, 4).cpu().numpy()

        for i, sphere in enumerate(spheres):
            if sphere[3] <= 0:                                  # disabled slot
                continue
            gaps = (np.linalg.norm(spheres[i + 1:, :3] - sphere[:3], axis=1)
                    - (spheres[i + 1:, 3] + sphere[3]))
            for offset in np.where(gaps < 0)[0]:
                a, b = sphere_to_link[i], sphere_to_link[i + 1 + offset]
                if a != b:
                    overlapping.add((min(a, b), max(a, b)))

    ignore = kinematics.get("self_collision_ignore") or {}
    added = 0
    for a, b in sorted(overlapping):
        first, second = links[a], links[b]
        ignore.setdefault(first, [])
        if second not in ignore[first]:
            ignore[first].append(second)
            added += 1
    kinematics["self_collision_ignore"] = ignore

    # Write back with the placeholders intact, so the config stays relocatable.
    kinematics["asset_root_path"] = "{ROOT}/assets"
    kinematics["urdf_path"] = "{ROOT}/assets/vega.urdf"
    yaml.safe_dump(config, open(ROBOT_CONFIG, "w"), sort_keys=False)
    print(f"added {added} ignore pairs from {len(overlapping)} rest-pose overlaps")


if __name__ == "__main__":
    main()
