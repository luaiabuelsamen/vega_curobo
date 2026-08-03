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

from vega_curobo.config import LEFT_TOOL, RIGHT_TOOL, load_robot_config  # noqa: E402
from vega_curobo.scene import Recorder, apply_locked_joints, build_scene, joint_writer  # noqa: E402
from vega_curobo.solvers import (home_state, make_planner, plan_to,  # noqa: E402
                                 tool_poses, tool_poses_at)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", choices=["right", "both"], default="right")
    parser.add_argument("--mode", choices=["live", "frames"], default="frames")
    parser.add_argument("--target", type=float, nargs=3, default=[0.72, -0.15, 1.05],
                        help="right hand; the left mirrors it in y when --arms both")
    parser.add_argument("--out")
    args = parser.parse_args()
    out = args.out or f"media/reach{'_bimanual' if args.arms == 'both' else ''}.mp4"
    target = np.array(args.target)

    planner = make_planner(args.arms)
    start = home_state(planner)
    # keep each hand's home orientation; the arms cannot reorient freely
    orientations = {f: pose[1] for f, pose in tool_poses(planner, start).items()}

    targets = {RIGHT_TOOL: target}
    if args.arms == "both":
        targets[LEFT_TOOL] = target * [1, -1, 1]

    print("planning to " + ", ".join(f"{f} {np.round(t, 3).tolist()}"
                                     for f, t in targets.items()) + " ...", flush=True)
    plan = plan_to(planner, targets, orientations, start)
    if plan is None:
        sys.exit("planning failed -- target is probably outside the reachable box")
    joint_names, waypoints = plan
    print(f"{len(waypoints)} waypoints over {len(joint_names)} joints", flush=True)

    # Confirm the tool actually lands on the target. A plan can report success
    # while the tip is elsewhere if the goal was fed in the wrong frame, so this
    # check is worth keeping.
    final = tool_poses_at(planner, joint_names, waypoints[-1])
    for frame, target_position in targets.items():
        error = np.linalg.norm(final[frame][0] - target_position)
        print(f"{frame} reach error: {1000 * error:.3f} mm", flush=True)

    model = build_scene(markers=targets)
    data = mujoco.MjData(model)
    apply_locked_joints(model, data, load_robot_config(args.arms))
    set_joints = joint_writer(model, data, joint_names)
    set_joints(waypoints[0])

    sequence = list(waypoints) + list(waypoints[::-1])       # out and back
    with Recorder(model, data, mode=args.mode, out=out) as recorder:
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
