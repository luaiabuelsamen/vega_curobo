"""Chase a ball that keeps moving, under MPC.

A ball appears at a random reachable point; the controller re-solves every frame
and the arm servos toward it. Within `--tolerance` it counts as caught and a new
ball spawns.

    python scripts/follow_ball.py --mode frames --balls 6    # -> media/follow_ball.mp4
    python scripts/follow_ball.py --mode live

This is the case a one-shot planner cannot serve: at ~12 s per solve the goal has
long since moved. MPC re-solves in ~0.3 s and each solve returns a short horizon.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if "frames" in sys.argv:
    os.environ["MUJOCO_GL"] = "egl"

import mujoco  # noqa: E402  (must follow MUJOCO_GL)

from vega_curobo.config import WORKSPACE_HI, WORKSPACE_LO, load_robot_config  # noqa: E402
from vega_curobo.scene import Recorder, apply_locked_joints, build_scene, joint_writer  # noqa: E402
from vega_curobo.solvers import home_state, make_mpc, set_mpc_goal, step_mpc, tool_pose  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["live", "frames"], default="frames")
    parser.add_argument("--balls", type=int, default=6, help="how many to chase")
    parser.add_argument("--tolerance", type=float, default=0.05, help="metres, counts as caught")
    parser.add_argument("--patience", type=int, default=120, help="solves before giving up")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="media/follow_ball.mp4")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    mpc = make_mpc()
    state = home_state(mpc)
    mpc.setup(state)
    _, quaternion = tool_pose(mpc, state)          # hold the home tool orientation

    model = build_scene()
    data = mujoco.MjData(model)
    apply_locked_joints(model, data, load_robot_config())
    set_joints = joint_writer(model, data, mpc.joint_names)

    caught = missed = 0
    with Recorder(model, data, mode=args.mode, out=args.out) as recorder:
        for _ in range(args.balls):
            if not recorder.running:
                break
            ball = rng.uniform(WORKSPACE_LO, WORKSPACE_HI)
            data.mocap_pos[0] = ball
            set_mpc_goal(mpc, ball, quaternion)

            for solve in range(args.patience):
                if not recorder.running:
                    break
                state, horizon = step_mpc(mpc, state)
                for q in horizon:                  # every horizon step, not just
                    set_joints(q)                  # the one we advance to
                    recorder.capture()

                position, _ = tool_pose(mpc, state)
                distance = np.linalg.norm(position - ball)
                if distance < args.tolerance:
                    caught += 1
                    print(f"caught at {np.round(ball, 3).tolist()} "
                          f"in {solve + 1} solves ({distance * 100:.1f} cm)", flush=True)
                    break
            else:
                missed += 1
                print(f"missed {np.round(ball, 3).tolist()}", flush=True)

    print(f"caught {caught}, missed {missed}")


if __name__ == "__main__":
    main()
