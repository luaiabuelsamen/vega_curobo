"""The cuRobo side: build a planner or an MPC controller for the right arm.

Both take the same robot config and the same floor obstacle. The split matters:
`MotionPlanner` solves a whole trajectory in one shot (~12 s warm on a Jetson
Orin) and `ModelPredictiveControl` refines a short action sequence every call
(~0.3 s), so one is for planning a motion and the other is for following a goal
that keeps moving.
"""
import numpy as np
import torch

from curobo.model_predictive_control import ModelPredictiveControl, ModelPredictiveControlCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState, Pose

from .config import FLOOR, TOOL_FRAME, load_robot_config

DEVICE = "cuda:0"


def _tensor(values):
    return torch.tensor([list(values)], device=DEVICE, dtype=torch.float32)


def make_planner():
    """One-shot trajectory planner."""
    return MotionPlanner(MotionPlannerCfg.create(
        robot=load_robot_config(), scene_model=FLOOR,
        collision_cache={"obb": 20, "mesh": 2}))


def make_mpc():
    """Reactive controller for a goal that moves."""
    return ModelPredictiveControl(ModelPredictiveControlCfg.create(
        robot=load_robot_config(), scene_model=FLOOR,
        use_cuda_graph=True, optimization_dt=0.1, interpolation_steps=4,
        optimizer_collision_activation_distance=0.05,
        # The default warm-start iteration count costs ~2.4 s per solve on a
        # Jetson Orin, which is slower than the one-shot planner it replaces.
        # 25 is the inner-iteration granularity -- the smallest legal value --
        # and gives ~0.3 s per solve with no visible loss of tracking quality.
        warm_start_optimization_num_iters=25,
        collision_cache={"obb": 20, "mesh": 2}))


def home_state(solver):
    """The solver's default configuration as a JointState with zero derivatives."""
    state = JointState.from_position(
        solver.default_joint_state.position.unsqueeze(0)
        if hasattr(solver, "default_joint_state")
        else solver.default_joint_position.clone().unsqueeze(0),
        joint_names=solver.joint_names)
    state.velocity = torch.zeros_like(state.position)
    state.acceleration = torch.zeros_like(state.position)
    return state


def tool_pose(solver, state):
    """Tool frame position and quaternion (numpy) for a joint state."""
    pose = solver.compute_kinematics(state).tool_poses.to_dict()[TOOL_FRAME]
    return (pose.position.flatten().cpu().numpy(),
            pose.quaternion.flatten().cpu().numpy())


def tool_pose_at(planner, joint_names, q):
    """Tool pose for one raw waypoint, in the plan's joint ordering."""
    state = JointState.from_position(
        torch.as_tensor(np.asarray([q]), device=DEVICE, dtype=torch.float32),
        joint_names=list(joint_names))
    return tool_pose(planner, planner.kinematics.get_active_js(state))


def plan_to(planner, position, quaternion, start, attempts=20):
    """Plan a trajectory to a tool pose. Returns (joint_names, waypoints) or None.

    The interpolated plan carries every joint, including the locked ones, in a
    different order from `planner.joint_names` -- pass it through
    `planner.kinematics.get_active_js` before any forward kinematics.
    """
    goal = GoalToolPose(
        tool_frames=planner.tool_frames,
        # [batch, horizon, link, goalset, dim]
        position=torch.tensor([[[[list(position)]]]], device=DEVICE, dtype=torch.float32),
        quaternion=torch.tensor([[[[list(quaternion)]]]], device=DEVICE, dtype=torch.float32))
    result = planner.plan_pose(goal, start, max_attempts=attempts, enable_graph_attempt=1)
    if result is None or not result.success.any():
        return None
    plan = result.get_interpolated_plan()
    waypoints = plan.position.reshape(-1, plan.position.shape[-1]).cpu().numpy()
    return list(plan.joint_names), waypoints


def set_mpc_goal(mpc, position, quaternion):
    """Point the controller at a new tool pose."""
    pose = Pose(position=_tensor(position), quaternion=_tensor(quaternion))
    mpc.update_goal_tool_poses(
        GoalToolPose.from_poses({TOOL_FRAME: pose},
                                ordered_tool_frames=mpc.tool_frames, num_goalset=1),
        run_ik=False)


def step_mpc(mpc, state):
    """One solve. Returns (next_state, horizon) where horizon is the full action
    sequence -- render every step of it, or the motion moves at the solve rate."""
    result = mpc.optimize_action_sequence(state)
    sequence = result.action_sequence
    if sequence is None or sequence.position.shape[1] == 0:
        return state, np.empty((0, state.position.shape[-1]))
    nxt = JointState.from_position(sequence.position[:, -1, :].clone(),
                                   joint_names=mpc.joint_names)
    nxt.velocity = sequence.velocity[:, -1, :]
    nxt.acceleration = sequence.acceleration[:, -1, :]
    return nxt, sequence.position[0].detach().cpu().numpy()
