"""Same plan, same obstacle, two controllers. What happens on contact.

    python scripts/compliance.py            # -> media/compliance.mp4, .gif, .csv

A block is placed in the arm's path *after* planning, so neither controller knows
it is there -- the situation every real robot eventually meets. The identical
trajectory is then executed twice:

    stiff       joint-space PD, high gain. What "write the plan into qpos"
                becomes when you give it actuators.
    compliant   geometric impedance control on SE(3).

Both are given gravity compensation and the same trajectory, so the only
difference is what they do when the world disagrees with the plan.

The stiff controller has no notion of contact: its error is in joint space, so a
blocked arm just integrates a larger and larger correction until something moves.
The impedance controller's error is a spatial spring, so contact force is bounded
by how far it has been pushed.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402

from vega_curobo.config import RIGHT_TOOL, load_robot_config  # noqa: E402
from vega_curobo.control import (GeometricImpedanceController,  # noqa: E402
                                 PoseTrajectory, hold_torque)
from vega_curobo.scene import TOOL_SITE, build_scene  # noqa: E402
from vega_curobo.solvers import home_state, make_planner, plan_to, tool_poses  # noqa: E402

DRIVEN = ["Lift", "torso_flip",
          "R_arm_j1", "R_arm_j2", "R_arm_j3", "R_arm_j4",
          "R_arm_j5", "R_arm_j6", "R_arm_j7"]
WALL_HALF = (0.05, 0.06, 0.16)
PANEL = (640, 360)


def contact_force(model, data, geom_id):
    """Total contact force magnitude on a geom this step."""
    total = np.zeros(3)
    wrench = np.zeros(6)
    for i in range(data.ncon):
        contact = data.contact[i]
        if geom_id not in (contact.geom1, contact.geom2):
            continue
        mujoco.mj_contactForce(model, data, i, wrench)
        frame = contact.frame.reshape(3, 3)
        total += frame.T @ wrench[:3]
    return float(np.linalg.norm(total))


def run(model, joint_names, waypoints, trajectory, mode, duration, settle, block_start):
    """Execute the trajectory once. Returns per-step telemetry and frames."""
    data = mujoco.MjData(model)
    for name, value in zip(joint_names, waypoints[0]):
        data.qpos[model.joint(name).qposadr[0]] = value
    for name, value in (load_robot_config()["kinematics"].get("lock_joints") or {}).items():
        try:
            data.qpos[model.joint(name).qposadr[0]] = value
        except KeyError:
            pass
    mujoco.mj_forward(model, data)

    driven_dofs = [model.joint(n).dofadr[0] for n in DRIVEN]
    driven_qpos = [model.joint(n).qposadr[0] for n in DRIVEN]
    driven_act = [model.actuator(f"{n}_motor").id for n in DRIVEN]
    held = [n for n in (model.joint(i).name for i in range(model.njnt))
            if n and n not in DRIVEN]
    held_dofs = [model.joint(n).dofadr[0] for n in held]
    held_act = [model.actuator(f"{n}_motor").id for n in held]
    held_targets = data.qpos[[model.joint(n).qposadr[0] for n in held]].copy()

    controller = GeometricImpedanceController(model, data, TOOL_SITE, driven_dofs)
    plan_qpos = np.array([[w[joint_names.index(n)] for n in DRIVEN] for w in waypoints])

    renderer = mujoco.Renderer(model, height=PANEL[1], width=PANEL[0])
    camera = model.camera("demo").id
    dt = model.opt.timestep
    steps = int((duration + settle) / dt)
    render_every = max(int(round(1.0 / 30.0 / dt)), 1)
    wall = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "support_wall")

    frames, forces, torques, deflection = [], [], [], []
    for step in range(steps):
        t = step * dt
        if mode == "compliant":
            tau = controller.torque(*trajectory.sample(t))
        else:
            # joint-space PD on the planned waypoints, plus gravity compensation
            index = int(np.clip(round(t / duration * (len(plan_qpos) - 1)),
                                0, len(plan_qpos) - 1))
            error = data.qpos[driven_qpos] - plan_qpos[index]
            tau = (-2000.0 * error - 80.0 * data.qvel[driven_dofs]
                   + data.qfrc_bias[driven_dofs])
        data.ctrl[driven_act] = tau
        data.ctrl[held_act] = hold_torque(model, data, held_dofs, held_targets)
        mujoco.mj_step(model, data)

        forces.append(contact_force(model, data, wall))
        torques.append(float(np.abs(tau).max()))
        pd = trajectory.sample(t)[0]
        deflection.append(float(np.linalg.norm(controller.tool_pose()[0] - pd)))
        if step % render_every == 0:
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render())

    renderer.close()
    return dict(frames=frames, force=np.array(forces), torque=np.array(torques),
                deflection=np.array(deflection), dt=dt)


def label(frame, text, colour):
    from PIL import Image, ImageDraw
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, PANEL[0], 26], fill=(15, 16, 18))
    draw.text((10, 7), text, fill=colour)
    return np.asarray(image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--settle", type=float, default=1.5)
    parser.add_argument("--target", type=float, nargs=3, default=[0.72, -0.15, 1.05])
    parser.add_argument("--out", default="media/compliance")
    args = parser.parse_args()

    planner = make_planner()
    start = home_state(planner)
    orientations = {f: pose[1] for f, pose in tool_poses(planner, start).items()}
    print("planning (the block does not exist yet) ...", flush=True)
    plan = plan_to(planner, {RIGHT_TOOL: np.array(args.target)}, orientations, start)
    if plan is None:
        sys.exit("planning failed")
    joint_names, waypoints = plan

    # Put the block on the planned path, two thirds of the way along, nudged so
    # the arm strikes it rather than passing through its centre.
    probe = build_scene(markers={}, physics=True)
    path = PoseTrajectory(probe, mujoco.MjData(probe), TOOL_SITE,
                          joint_names, waypoints, args.duration)
    hit = path.positions[int(0.66 * (len(path.positions) - 1))]
    block_start = np.array([hit[0], hit[1] - 0.02, hit[2] - 0.02])
    print(f"block at {np.round(block_start, 3).tolist()}", flush=True)

    # A fixed wall, not a loose block. A light object just gets launched by both
    # controllers, which measures the impact, not the compliance. Against
    # something immovable the difference is the whole point: one keeps pushing,
    # the other stops at a force set by how far it has been deflected.
    model = build_scene(markers={RIGHT_TOOL: np.array(args.target)}, physics=True,
                        supports={"wall": (block_start, WALL_HALF)})
    trajectory = PoseTrajectory(model, mujoco.MjData(model), TOOL_SITE,
                                joint_names, waypoints, args.duration)

    results = {}
    for mode in ("stiff", "compliant"):
        print(f"running {mode} ...", flush=True)
        results[mode] = run(model, joint_names, waypoints, trajectory, mode,
                            args.duration, args.settle, block_start)
        r = results[mode]
        print(f"  peak contact force {r['force'].max():8.1f} N   "
              f"peak joint torque {r['torque'].max():8.1f}   "
              f"max deflection {100 * r['deflection'].max():5.1f} cm", flush=True)

    import imageio.v2 as imageio
    n = min(len(results["stiff"]["frames"]), len(results["compliant"]["frames"]))
    composed = []
    for i in range(n):
        left = label(results["stiff"]["frames"][i], "STIFF  position control", (235, 120, 100))
        right = label(results["compliant"]["frames"][i], "COMPLIANT  impedance control",
                      (130, 210, 150))
        composed.append(np.hstack([left, right]))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    imageio.mimsave(f"{args.out}.mp4", composed, fps=30, quality=8)
    imageio.mimsave(f"{args.out}.gif", composed[::2], fps=15, palettesize=64, loop=0)
    print(f"wrote {args.out}.mp4 and .gif ({len(composed)} frames)", flush=True)

    dt = results["stiff"]["dt"]
    rows = ["t,stiff_force,compliant_force,stiff_torque,compliant_torque"]
    for i in range(len(results["stiff"]["force"])):
        rows.append(f"{i * dt:.4f},{results['stiff']['force'][i]:.4f},"
                    f"{results['compliant']['force'][i]:.4f},"
                    f"{results['stiff']['torque'][i]:.4f},"
                    f"{results['compliant']['torque'][i]:.4f}")
    open(f"{args.out}.csv", "w").write("\n".join(rows))
    print(f"wrote {args.out}.csv", flush=True)


if __name__ == "__main__":
    main()
