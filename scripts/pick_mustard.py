"""Pick a mustard bottle off a table and set it down somewhere else.

    python scripts/pick_mustard.py --mode frames     # -> media/pick_mustard.mp4
    python scripts/pick_mustard.py --mode live

Every pose here is derived, not typed in. The approach direction comes from a
feasibility search over candidate grasps, the tool goal comes from the measured
jaw offset, and the carried bottle is attached to the robot so the transport is
planned with it in hand rather than around a robot that is pretending to be
empty.

Six segments, and the obstacle bookkeeping is most of the work:

  1. pre-grasp   bottle is a world obstacle, keep clear of it
  2. grasp       bottle disabled, or the jaws closing around it read as collision
  3. lift        bottle attached to the robot, table disabled: the carried
                 object's spheres rest against the surface it just left, which
                 otherwise reports the start state in collision
  4. transport   above the place site
  5. place       lower onto the table
  6. retract     bottle detached and re-enabled at its new pose

Motion is kinematic: joint angles are written straight to qpos and the bottle
follows the tool frame once grasped. This shows what the planner produced. It is
not a physics grasp, and nothing here would hold a real bottle up.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if "frames" in sys.argv:
    os.environ["MUJOCO_GL"] = "egl"

import mujoco  # noqa: E402  (must follow MUJOCO_GL)

from vega_curobo.config import RIGHT_TOOL, ROOT, load_robot_config  # noqa: E402
from vega_curobo.grasp import (find_grasp, matrix_to_quaternion,  # noqa: E402
                               quaternion_to_matrix, tool_pose_for_grasp)
from vega_curobo.scene import (Recorder, apply_locked_joints, build_scene,  # noqa: E402
                               joint_writer, set_object)
from vega_curobo.solvers import (attach_object, detach_object, home_state,  # noqa: E402
                                 make_planner, plan_to, set_obstacle_enabled,
                                 tool_poses, tool_poses_at)

MESH = os.path.join(ROOT, "assets", "objects", "mustard.obj")

TABLE_TOP = 0.85
TABLE_CENTRE = (0.60, -0.20, TABLE_TOP - 0.02)
TABLE_HALF = (0.25, 0.30, 0.02)

# Mesh bounds are 0.0972 x 0.0666 x 0.1914 with its base 0.0839 below the origin,
# so this is where the body sits for the bottle to stand on the table. cuRobo
# plans against a box of these dimensions; the mesh is only for the picture.
BOTTLE_SIZE = np.array([0.0972, 0.0666, 0.1914])
BOTTLE_BASE_OFFSET = 0.0839
PICK_XY = (0.60, -0.28)
PLACE_XY = (0.58, -0.02)
GRASP_HEIGHT = 0.12          # above the table, on the upper body of the bottle
LIFT = 0.16


def bottle_body_position(xy):
    return np.array([xy[0], xy[1], TABLE_TOP + BOTTLE_BASE_OFFSET])


def bottle_obstacle(xy):
    """cuRobo sees a box around the bottle; the mesh is only for the picture."""
    centre = [xy[0], xy[1], TABLE_TOP + BOTTLE_SIZE[2] / 2]
    return {"dims": BOTTLE_SIZE.tolist(), "pose": centre + [1, 0, 0, 0]}


def world_model(pick_xy):
    table_centre = list(TABLE_CENTRE)
    return {"cuboid": {
        "floor": {"dims": [3.0, 3.0, 0.1], "pose": [0, 0, -0.5, 1, 0, 0, 0]},
        "table": {"dims": [2 * h for h in TABLE_HALF], "pose": table_centre + [1, 0, 0, 0]},
        "mustard": bottle_obstacle(pick_xy),
    }}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["live", "frames"], default="frames")
    parser.add_argument("--out", default="media/pick_mustard.mp4")
    args = parser.parse_args()

    # no CUDA graph: this script queries IK for grasp feasibility and then
    # plans, and the solver cannot recapture the graph when the problem shape
    # changes between the two
    planner = make_planner(world=world_model(PICK_XY), attachable=True,
                           use_cuda_graph=False)
    state = home_state(planner)

    grasp_point = np.array([PICK_XY[0], PICK_XY[1], TABLE_TOP + GRASP_HEIGHT])
    place_point = np.array([PLACE_XY[0], PLACE_XY[1], TABLE_TOP + GRASP_HEIGHT])

    # The manoeuvre, defined once: (label, target, standoff along the approach).
    # The feasibility search and the planning loop both read this, so they
    # cannot disagree about what the arm is being asked to do.
    manoeuvre = [
        ("pre-grasp", grasp_point, 0.12),
        ("grasp", grasp_point, 0.0),
        ("lift", grasp_point + [0, 0, LIFT], 0.0),
        ("transport", place_point + [0, 0, LIFT], 0.0),
        ("place", place_point, 0.0),
        ("retract", place_point, 0.12),
    ]

    print("searching for a reachable grasp ...", flush=True)
    # Search with the bottle disabled, for the same reason the grasp segment
    # disables it: at the grasp pose the jaws surround the bottle, so with it
    # still a world obstacle every candidate fails IK on collision. Leaving it
    # enabled here rejects every approach direction while the pre-grasp poses
    # all pass, which looks like an unreachable object rather than a bookkeeping
    # error.
    set_obstacle_enabled(planner, "mustard", False)
    found = find_grasp(planner, [(position, standoff) for _, position, standoff in manoeuvre])
    set_obstacle_enabled(planner, "mustard", True)
    if found is None:
        sys.exit("no reachable grasp: every approach direction failed IK")
    orientation, yaw, tilt = found
    quaternion = matrix_to_quaternion(orientation)
    print(f"grasp approach yaw={yaw}deg tilt={tilt}deg", flush=True)

    def goal(position):
        return {RIGHT_TOOL: position}, {RIGHT_TOOL: quaternion}

    segments = []
    joint_names = None

    def run(label, position, standoff, start):
        nonlocal joint_names
        targets, orientations = goal(tool_pose_for_grasp(position, orientation, standoff))
        plan = plan_to(planner, targets, orientations, start)
        if plan is None:
            sys.exit(f"planning failed on segment '{label}'")
        joint_names, waypoints = plan
        reached, reached_quat = tool_poses_at(planner, joint_names, waypoints[-1])[RIGHT_TOOL]
        error = np.linalg.norm(reached - targets[RIGHT_TOOL])
        # orientation matters as much as position here: the gripper has to stay
        # square to the bottle, and a plan can nail the position while drifting
        # in rotation
        misalign = np.degrees(np.arccos(np.clip(
            2 * float(np.dot(reached_quat, orientations[RIGHT_TOOL])) ** 2 - 1, -1, 1)))
        print(f"  {label:10s} {len(waypoints):4d} waypoints, "
              f"tool error {1000 * error:.2f} mm, {misalign:.2f} deg", flush=True)
        segments.append((label, waypoints))
        return planner.kinematics.get_active_js(_as_state(planner, joint_names, waypoints[-1]))

    def _as_state(planner, names, q):
        import torch
        from curobo.types import JointState
        return JointState.from_position(
            torch.as_tensor(np.asarray([q]), device="cuda:0", dtype=torch.float32),
            joint_names=list(names))

    print("planning ...", flush=True)
    for label, position, standoff in manoeuvre:
        # obstacle bookkeeping, applied before the segment it enables
        if label == "grasp":
            set_obstacle_enabled(planner, "mustard", False)
        elif label == "lift":
            attach_object(planner, state, ["mustard"])
            set_obstacle_enabled(planner, "table", False)
        elif label == "retract":
            detach_object(planner, ["mustard"])
            set_obstacle_enabled(planner, "table", True)
        state = run(label, position, standoff, state)

    # ---- playback -----------------------------------------------------------
    bottle_start = bottle_body_position(PICK_XY)
    model = build_scene(
        markers={},                       # the bottle is the subject here
        table=(TABLE_CENTRE, TABLE_HALF),
        meshes={"mustard": (MESH, bottle_start, (0.85, 0.72, 0.10, 1))})
    data = mujoco.MjData(model)
    apply_locked_joints(model, data, load_robot_config())
    set_joints = joint_writer(model, data, joint_names)
    # place it through set_object so the mesh frame correction is applied
    set_object(model, data, "mustard", bottle_start)

    # The bottle rides the tool between the end of "grasp" and the end of
    # "place". Its pose in the tool frame is captured once, at pickup, and held
    # fixed -- which is what attaching it to the robot means.
    carry = None            # (offset, rotation) of the bottle in the tool frame
    bottle_pose = (bottle_start, np.eye(3))

    with Recorder(model, data, mode=args.mode, out=args.out) as recorder:
        for label, waypoints in segments:
            for q in waypoints:
                if not recorder.running:
                    break
                set_joints(q)
                position, quat = tool_poses_at(planner, joint_names, q)[RIGHT_TOOL]
                rotation = quaternion_to_matrix(quat)
                if carry is not None:
                    bottle_pose = (position + rotation @ carry[0], rotation @ carry[1])
                    set_object(model, data, "mustard", bottle_pose[0],
                               matrix_to_quaternion(bottle_pose[1]))
                    mujoco.mj_forward(model, data)
                recorder.capture()

            if label == "grasp":
                carry = (rotation.T @ (bottle_pose[0] - position), rotation.T @ bottle_pose[1])
            elif label == "place":
                carry = None

    tilt = np.degrees(np.arccos(np.clip(bottle_pose[1][2, 2], -1, 1)))
    print(f"bottle from {np.round(bottle_start, 3).tolist()} "
          f"to {np.round(bottle_pose[0], 3).tolist()}, upright to {tilt:.1f} deg")


if __name__ == "__main__":
    main()
