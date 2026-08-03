"""Repo paths and the robot config loader."""
import os

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF = os.path.join(ROOT, "assets", "vega.urdf")
ROBOT_CONFIG = os.path.join(ROOT, "configs", "vega_right.yml")

#: cuRobo tool frames, one per hand. Both are massless fixed links at the wrist
#: mounts, so they exist in the URDF but MuJoCo merges them away -- see the
#: README note on frames.
RIGHT_TOOL = "R_ee"
LEFT_TOOL = "L_ee"
TOOL_FRAME = RIGHT_TOOL

#: Right-arm workspace box that is reachable at the home tool orientation
#: (40/40 IK hits sampled uniformly across it). Sampling outside this needs
#: rejection logic: the 7-DoF arm cannot hold an arbitrary orientation anywhere.
WORKSPACE_LO = (0.36, -0.53, 0.85)
WORKSPACE_HI = (0.79, 0.14, 1.45)


def workspace(tool_frame):
    """Reachable box for one hand. The arms are mirror images about y=0, so the
    left box is the right box reflected (home tool poses are y=-0.419 / +0.419)."""
    lo, hi = np.array(WORKSPACE_LO), np.array(WORKSPACE_HI)
    if tool_frame == LEFT_TOOL:
        lo, hi = lo.copy(), hi.copy()
        lo[1], hi[1] = -hi[1], -lo[1]
    return lo, hi

#: A floor at z=0 intersects the base collision spheres, because the Vega stands
#: at the URDF origin. cuRobo then reports every solve infeasible -- with a
#: converged IK and zero pose error, which makes it look like a solver bug.
#: Keep the floor clear of the base.
FLOOR = {"cuboid": {"floor": {"dims": [3.0, 3.0, 0.1],
                              "pose": [0.0, 0.0, -0.5, 1.0, 0.0, 0.0, 0.0]}}}


def load_robot_config(arms="right"):
    """Robot config with its {ROOT} placeholders resolved to this checkout.

    `arms="both"` unlocks the left arm and declares a second tool frame. No
    refit is needed for that: the builder fitted collision spheres to all 28
    links, and locking a joint only removes it from the active chain. Going
    bimanual takes the arm from 10 active joints to 18, and costs roughly 3x per
    MPC solve and 10x per one-shot plan.
    """
    text = open(ROBOT_CONFIG).read().replace("{ROOT}", ROOT)
    config = yaml.safe_load(text)
    if arms == "both":
        kinematics = config["kinematics"]
        locked = kinematics.get("lock_joints") or {}
        for joint in [j for j in locked if j.startswith("L_")]:
            del locked[joint]
        kinematics["tool_frames"] = [RIGHT_TOOL, LEFT_TOOL]
    elif arms != "right":
        raise ValueError(f"arms must be 'right' or 'both', got {arms!r}")
    return config
