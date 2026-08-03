"""Chase balls that keep moving, under MPC. One arm or both.

Each hand gets a ball at a random reachable point; the controller re-solves every
frame and the arms servo toward them. Within `--tolerance` a ball counts as
caught and that hand gets a new one.

    python scripts/follow_ball.py --mode frames --balls 6
    python scripts/follow_ball.py --arms both --mode frames --balls 4
    python scripts/follow_ball.py --arms both --mode live

This is the case a one-shot planner cannot serve: it needs ~12 s per solve for
one arm and ~125 s for two, by which time the goal has moved. MPC re-solves in
~0.3 s (one arm) or ~0.9 s (two) and each solve returns a short horizon.

Both hands are driven by a single solver over one kinematic chain, so the arms
are coordinated rather than independently controlled: the shared torso and lift
joints are part of the same optimisation.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if "frames" in sys.argv:
    os.environ["MUJOCO_GL"] = "egl"

import mujoco  # noqa: E402  (must follow MUJOCO_GL)

from vega_curobo.config import load_robot_config, workspace  # noqa: E402
from vega_curobo.scene import (Recorder, apply_locked_joints, build_scene,  # noqa: E402
                               joint_writer, set_marker)
from vega_curobo.solvers import (home_state, make_mpc, set_mpc_goal,  # noqa: E402
                                 step_mpc, tool_poses, tool_positions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", choices=["right", "both"], default="right")
    parser.add_argument("--mode", choices=["live", "frames"], default="frames")
    parser.add_argument("--balls", type=int, default=6, help="per hand")
    parser.add_argument("--tolerance", type=float, default=0.05, help="metres, counts as caught")
    parser.add_argument("--patience", type=int, default=120, help="solves before giving up")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out")
    args = parser.parse_args()
    out = args.out or f"media/follow_ball{'_bimanual' if args.arms == 'both' else ''}.mp4"

    rng = np.random.default_rng(args.seed)
    mpc = make_mpc(args.arms)
    state = home_state(mpc)
    mpc.setup(state)
    frames = mpc.tool_frames
    # hold each hand's home orientation; the arms cannot reorient freely
    orientations = {f: pose[1] for f, pose in tool_poses(mpc, state).items()}

    def spawn(frame):
        lo, hi = workspace(frame)
        return rng.uniform(lo, hi)

    balls = {f: spawn(f) for f in frames}
    caught = {f: 0 for f in frames}
    missed = {f: 0 for f in frames}
    since_spawn = {f: 0 for f in frames}

    model = build_scene(markers=balls)
    data = mujoco.MjData(model)
    apply_locked_joints(model, data, load_robot_config(args.arms))
    set_joints = joint_writer(model, data, mpc.joint_names)
    for frame, ball in balls.items():
        set_marker(model, data, frame, ball)
    set_mpc_goal(mpc, balls, orientations)

    with Recorder(model, data, mode=args.mode, out=out) as recorder:
        while recorder.running and any(caught[f] + missed[f] < args.balls for f in frames):
            state, horizon = step_mpc(mpc, state)
            for q in horizon:                      # every horizon step, not just
                set_joints(q)                      # the one we advance to
                recorder.capture()

            positions = tool_positions(mpc, state)
            respawn = False
            for frame in frames:
                if caught[frame] + missed[frame] >= args.balls:
                    continue
                since_spawn[frame] += 1
                distance = np.linalg.norm(positions[frame] - balls[frame])
                if distance < args.tolerance:
                    caught[frame] += 1
                    print(f"{frame} caught {np.round(balls[frame], 3).tolist()} "
                          f"in {since_spawn[frame]} solves ({distance * 100:.1f} cm)", flush=True)
                elif since_spawn[frame] >= args.patience:
                    missed[frame] += 1
                    print(f"{frame} missed {np.round(balls[frame], 3).tolist()}", flush=True)
                else:
                    continue
                balls[frame] = spawn(frame)
                since_spawn[frame] = 0
                set_marker(model, data, frame, balls[frame])
                respawn = True

            if respawn:
                set_mpc_goal(mpc, balls, orientations)

    for frame in frames:
        print(f"{frame}: caught {caught[frame]}, missed {missed[frame]}")


if __name__ == "__main__":
    main()
