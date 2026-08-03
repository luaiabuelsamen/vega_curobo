"""Repo paths and the robot config loader."""
import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF = os.path.join(ROOT, "assets", "vega.urdf")
ROBOT_CONFIG = os.path.join(ROOT, "configs", "vega_right.yml")

#: cuRobo tool frame. A massless fixed link at the wrist mount, so it exists in
#: the URDF but MuJoCo merges it away -- see the README note on frames.
TOOL_FRAME = "R_ee"

#: Right-arm workspace box that is reachable at the home tool orientation
#: (40/40 IK hits sampled uniformly across it). Sampling outside this needs
#: rejection logic: the 7-DoF arm cannot hold an arbitrary orientation anywhere.
WORKSPACE_LO = (0.36, -0.53, 0.85)
WORKSPACE_HI = (0.79, 0.14, 1.45)

#: A floor at z=0 intersects the base collision spheres, because the Vega stands
#: at the URDF origin. cuRobo then reports every solve infeasible -- with a
#: converged IK and zero pose error, which makes it look like a solver bug.
#: Keep the floor clear of the base.
FLOOR = {"cuboid": {"floor": {"dims": [3.0, 3.0, 0.1],
                              "pose": [0.0, 0.0, -0.5, 1.0, 0.0, 0.0, 0.0]}}}


def load_robot_config():
    """Robot config with its {ROOT} placeholders resolved to this checkout."""
    text = open(ROBOT_CONFIG).read().replace("{ROOT}", ROOT)
    return yaml.safe_load(text)
