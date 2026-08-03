"""Turning "grasp this object here" into a tool pose cuRobo can plan to.

Two things stand between a grasp and a goal pose.

The tool frame is not where the fingers are. `R_ee` sits at the wrist mount, and
the jaw midpoint is 15.3 cm further along the tool's local +z, measured from the
gripper geometry rather than guessed. Planning straight to the object's position
therefore drives the wrist into it. `tool_pose_for_grasp` backs the goal off by
that offset.

The arm cannot hold an arbitrary orientation. A grasp direction that is
geometrically sensible is often unreachable, so `find_grasp` searches candidate
approach directions for one whose IK actually converges instead of committing to
a hand-picked pose.
"""
import numpy as np
import torch

from curobo.types import GoalToolPose

#: Jaw midpoint in the tool frame, from the open-gripper geometry. The jaws close
#: along the tool's local x.
GRASP_OFFSET = np.array([0.0, 0.0, 0.153])


def quaternion_to_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def matrix_to_quaternion(R):
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        w = np.sqrt(1.0 + trace) / 2.0
        return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w),
                         (R[0, 2] - R[2, 0]) / (4 * w),
                         (R[1, 0] - R[0, 1]) / (4 * w)])
    i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
    j, k = (i + 1) % 3, (i + 2) % 3
    t = np.sqrt(max(0.0, 1.0 + R[i, i] - R[j, j] - R[k, k]))
    q = np.zeros(4)
    q[i + 1] = t / 2
    q[0] = (R[k, j] - R[j, k]) / (2 * t)
    q[j + 1] = (R[j, i] + R[i, j]) / (2 * t)
    q[k + 1] = (R[k, i] + R[i, k]) / (2 * t)
    return q / np.linalg.norm(q)


def grasp_orientation(yaw, tilt=0.0):
    """A side grasp: the tool's +z points at the object along `yaw` in the
    horizontal plane, tipped down by `tilt`, with the jaws closing horizontally."""
    approach = np.array([np.cos(yaw) * np.cos(tilt),
                         np.sin(yaw) * np.cos(tilt),
                         -np.sin(tilt)])
    jaw_axis = np.cross([0.0, 0.0, 1.0], approach)
    if np.linalg.norm(jaw_axis) < 1e-6:               # straight down, jaws free
        jaw_axis = np.array([1.0, 0.0, 0.0])
    jaw_axis /= np.linalg.norm(jaw_axis)
    return np.column_stack([jaw_axis, np.cross(approach, jaw_axis), approach])


def tool_pose_for_grasp(grasp_position, orientation, standoff=0.0):
    """Tool pose that puts the jaw midpoint at `grasp_position`.

    `standoff` backs the tool off along its approach axis, for a pre-grasp pose.
    """
    offset = GRASP_OFFSET + np.array([0.0, 0.0, standoff])
    return np.asarray(grasp_position) - orientation @ offset


def ik_feasible(planner, position, quaternion):
    """Does IK converge for this tool pose?"""
    goal = GoalToolPose(
        tool_frames=planner.tool_frames,
        position=torch.tensor([[[[list(position)]]]], device="cuda:0", dtype=torch.float32),
        quaternion=torch.tensor([[[[list(quaternion)]]]], device="cuda:0", dtype=torch.float32))
    return bool(planner.ik_solver.solve_pose(goal).success.any())


def find_grasp(planner, poses, yaws=None, tilts=(0.0, 15.0, 30.0)):
    """First approach direction reachable at *every* pose of the manoeuvre.

    `poses` is the whole sequence as (position, standoff) pairs, not just the
    grasp. The gripper holds one orientation from approach to release, so an
    orientation that only works at the grasp is worthless: on this arm, yaw=-45
    passes the grasp and pre-grasp, plans three segments, and then has no
    solution at the transport pose. Checking the sequence up front turns that
    into an immediate answer instead of a failure three plans deep.

    Returns (orientation, yaw, tilt) or None.
    """
    if yaws is None:
        yaws = range(-60, 121, 15)
    for tilt_deg in tilts:
        for yaw_deg in yaws:
            orientation = grasp_orientation(np.radians(yaw_deg), np.radians(tilt_deg))
            quaternion = matrix_to_quaternion(orientation)
            if all(ik_feasible(planner, tool_pose_for_grasp(position, orientation, standoff),
                               quaternion)
                   for position, standoff in poses):
                return orientation, yaw_deg, tilt_deg
    return None
