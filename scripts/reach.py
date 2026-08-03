"""Plan one collision-free reach and play it back.

    python scripts/reach.py --mode frames        # offscreen -> media/reach.mp4
    python scripts/reach.py --mode live          # interactive window
    python scripts/reach.py --target 0.6 -0.3 1.2

Planning takes ~12 s warm (~26 s on the first call, which pays JIT).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if "frames" in sys.argv:
    os.environ["MUJOCO_GL"] = "egl"

import mujoco  # noqa: E402  (must follow MUJOCO_GL)

from vega_curobo.config import TOOL_FRAME, load_robot_config  # noqa: E402
from vega_curobo.scene import Recorder, apply_locked_joints, build_scene, joint_writer  # noqa: E402
from vega_curobo.solvers import (home_state, make_planner, plan_to,  # noqa: E402
                                 tool_pose, tool_pose_at)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["live", "frames"], default="frames")
    parser.add_argument("--target", type=float, nargs=3, default=[0.72, -0.15, 1.05])
    parser.add_argument("--out", default="media/reach.mp4")
    args = parser.parse_args()
    target = np.array(args.target)

    planner = make_planner()
    start = home_state(planner)
    _, quaternion = tool_pose(planner, start)      # keep the home tool orientation

    print(f"planning to {target.tolist()} ...", flush=True)
    plan = plan_to(planner, target, quaternion, start)
    if plan is None:
        sys.exit("planning failed -- target is probably outside the reachable box")
    joint_names, waypoints = plan
    print(f"{len(waypoints)} waypoints over {len(joint_names)} joints", flush=True)

    # Confirm the tool actually lands on the target. A plan can report success
    # while the tip is elsewhere if the goal was fed in the wrong frame, so this
    # check is worth keeping.
    reached, _ = tool_pose_at(planner, joint_names, waypoints[-1])
    print(f"reach error: {1000 * np.linalg.norm(reached - target):.3f} mm", flush=True)

    model = build_scene(marker_pos=target)
    data = mujoco.MjData(model)
    apply_locked_joints(model, data, load_robot_config())
    set_joints = joint_writer(model, data, joint_names)
    set_joints(waypoints[0])

    sequence = list(waypoints) + list(waypoints[::-1])       # out and back
    with Recorder(model, data, mode=args.mode, out=args.out) as recorder:
        while recorder.running:
            for q in sequence:
                if not recorder.running:
                    break
                set_joints(q)
                recorder.capture()
            if args.mode == "frames":
                break


if __name__ == "__main__":
    main()
