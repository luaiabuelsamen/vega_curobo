"""Execute a cuRobo plan with geometric impedance control, under gravity.

    python scripts/track_gic.py --mode frames      # -> media/track_gic.mp4
    python scripts/track_gic.py --mode live

Every other demo in this repo writes joint angles into qpos, which is a way of
saying "assume a controller". This one actually runs the controller: cuRobo
plans the trajectory, and torques computed from the SE(3) tracking error drive
the arm along it while gravity pulls on the whole upper body.

The point is compliance, not accuracy. A stiff position servo tracks a plan
until something unexpected touches it and then fights, which is how a light
object gets flung across a table. An impedance controller deflects and pushes
back with a bounded force, so the same collision is survivable -- and it is what
makes force-regulated contact tasks possible at all.

Control law from Seo et al., arXiv:2504.17080; see vega_curobo/control.py.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if "frames" in sys.argv:
    os.environ["MUJOCO_GL"] = "egl"

import mujoco  # noqa: E402  (must follow MUJOCO_GL)

from vega_curobo.config import RIGHT_TOOL, load_robot_config  # noqa: E402
from vega_curobo.control import (GeometricImpedanceController,  # noqa: E402
                                 PoseTrajectory, hold_torque)
from vega_curobo.scene import TOOL_SITE, Recorder, build_scene  # noqa: E402
from vega_curobo.solvers import home_state, make_planner, plan_to, tool_poses  # noqa: E402

# The chain from the base to the tool. The gripper joints move fingers, not the
# tool frame, so they are held rather than driven.
DRIVEN = ["Lift", "torso_flip",
          "R_arm_j1", "R_arm_j2", "R_arm_j3", "R_arm_j4",
          "R_arm_j5", "R_arm_j6", "R_arm_j7"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["live", "frames"], default="frames")
    parser.add_argument("--target", type=float, nargs=3, default=[0.72, -0.15, 1.05])
    parser.add_argument("--duration", type=float, default=6.0, help="seconds to execute")
    parser.add_argument("--settle", type=float, default=1.5, help="seconds holding at the end")
    parser.add_argument("--posture", type=float, default=0.0,
                        help="nullspace posture gain; costs tracking, see README")
    parser.add_argument("--out", default="media/track_gic.mp4")
    args = parser.parse_args()
    target = np.array(args.target)

    planner = make_planner()
    start = home_state(planner)
    orientations = {f: pose[1] for f, pose in tool_poses(planner, start).items()}
    print(f"planning to {target.tolist()} ...", flush=True)
    plan = plan_to(planner, {RIGHT_TOOL: target}, orientations, start)
    if plan is None:
        sys.exit("planning failed")
    joint_names, waypoints = plan
    print(f"{len(waypoints)} waypoints", flush=True)

    model = build_scene(markers={RIGHT_TOOL: target}, physics=True)
    data = mujoco.MjData(model)

    # start the robot at the plan's first waypoint, at rest
    for name, value in zip(joint_names, waypoints[0]):
        data.qpos[model.joint(name).qposadr[0]] = value
    for name, value in (load_robot_config()["kinematics"].get("lock_joints") or {}).items():
        try:
            data.qpos[model.joint(name).qposadr[0]] = value
        except KeyError:
            pass
    mujoco.mj_forward(model, data)

    driven_qpos = [model.joint(n).qposadr[0] for n in DRIVEN]
    driven_dofs = [model.joint(n).dofadr[0] for n in DRIVEN]
    driven_act = [model.actuator(f"{n}_motor").id for n in DRIVEN]
    held = [model.joint(i).name for i in range(model.njnt)
            if model.joint(i).name not in DRIVEN]
    held_dofs = [model.joint(n).dofadr[0] for n in held]
    held_act = [model.actuator(f"{n}_motor").id for n in held]
    held_targets = data.qpos[[model.joint(n).qposadr[0] for n in held]].copy()

    trajectory = PoseTrajectory(model, data, TOOL_SITE, joint_names, waypoints, args.duration)
    controller = GeometricImpedanceController(model, data, TOOL_SITE, driven_dofs)
    if args.posture > 0:
        controller.posture_gain = args.posture
        controller.posture_damping = args.posture / 3
        controller.posture_reference = data.qpos[
            [model.joint(n).qposadr[0] for n in DRIVEN]].copy()

    dt = model.opt.timestep
    steps = int((args.duration + args.settle) / dt)
    render_every = max(int(round(1.0 / 30.0 / dt)), 1)
    errors, final = [], None

    print(f"executing: {steps} steps at {dt * 1000:.0f} ms, gravity on", flush=True)
    with Recorder(model, data, mode=args.mode, out=args.out) as recorder:
        for step in range(steps):
            if not recorder.running:
                break
            pd, Rd, vd, wd, dvd, dwd = trajectory.sample(step * dt)
            data.ctrl[driven_act] = controller.torque(pd, Rd, vd, wd, dvd, dwd)
            data.ctrl[held_act] = hold_torque(model, data, held_dofs, held_targets)
            mujoco.mj_step(model, data)

            position, _ = controller.tool_pose()
            errors.append(np.linalg.norm(position - pd))
            final = (position.copy(), pd.copy())
            if step % render_every == 0:
                recorder.capture()

    errors = np.array(errors)
    tracked = errors[: int(args.duration / dt)]
    print(f"tracking error  mean {1000 * tracked.mean():.1f} mm, "
          f"max {1000 * tracked.max():.1f} mm", flush=True)
    print(f"settled error   {1000 * errors[-1]:.1f} mm", flush=True)
    print(f"tool ended at {np.round(final[0], 4).tolist()}, "
          f"commanded {np.round(final[1], 4).tolist()}", flush=True)
    limits = model.jnt_range[[model.joint(n).id for n in DRIVEN]]
    q = data.qpos[driven_qpos]
    margin = np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q))
    print(f"closest joint-limit margin {margin:+.3f} rad", flush=True)


if __name__ == "__main__":
    main()
