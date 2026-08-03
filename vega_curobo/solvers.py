"""The cuRobo side: build a planner or an MPC controller for one or both arms.

Both take the same robot config and the same floor obstacle. The split matters:
`MotionPlanner` solves a whole trajectory in one shot and
`ModelPredictiveControl` refines a short action sequence every call, so one is
for planning a motion and the other is for following a goal that keeps moving.

Goals are dictionaries keyed by tool frame, so single-arm and bimanual use the
same calls; the arm count only changes what you put in the dictionary.
"""
import numpy as np
import torch

from curobo.model_predictive_control import ModelPredictiveControl, ModelPredictiveControlCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState, Pose

from .config import FLOOR, LEFT_TOOL, RIGHT_TOOL, load_robot_config

DEVICE = "cuda:0"


def make_planner(arms="right", world=None, attachable=False, use_cuda_graph=True):
    """One-shot trajectory planner.

    `world` replaces the default floor-only scene. `attachable` adds the
    `attached_object` link the config is generated without, which is what the
    attachment managers hang carried objects off; without it `attach` has
    nowhere to put the object's spheres.

    Set `use_cuda_graph=False` if you mix bare IK queries and full plans on one
    planner. The solver captures a CUDA graph at the first problem shape it
    sees, and a later call with a different shape raises "CUDA graph reset is
    not available" rather than recapturing. Querying IK for grasp feasibility
    and then planning is exactly that pattern.
    """
    config = load_robot_config(arms)
    if attachable:
        add_attachment_link(config)
    return MotionPlanner(MotionPlannerCfg.create(
        robot=config, scene_model=world or FLOOR, use_cuda_graph=use_cuda_graph,
        collision_cache={"obb": 20, "mesh": 2}))


def add_attachment_link(config, tool_frame=RIGHT_TOOL, num_spheres=64):
    """Give the config somewhere to put a carried object.

    `add_object_link: True` alone is not enough -- the attachment managers look
    up a link literally named `attached_object` and raise KeyError without it.
    It has to be declared as an extra link fixed to the tool frame, with a pool
    of empty sphere slots the manager fills when it fits the payload, and it
    must ignore self-collision against the hand that is holding it.
    """
    kinematics = config["kinematics"]
    link = "attached_object"
    if link in kinematics["collision_link_names"]:
        return config
    hand = [name for name in kinematics["collision_link_names"]
            if name.startswith(tool_frame[0] + "_gripper")
            or name in (f"{tool_frame[0]}_arm_l7", f"{tool_frame[0]}_arm_l8")]

    kinematics["add_object_link"] = True
    kinematics["collision_link_names"] = kinematics["collision_link_names"] + [link]
    kinematics["extra_collision_spheres"] = {link: num_spheres}
    kinematics["extra_links"] = {link: {
        "fixed_transform": [0, 0, 0, 1, 0, 0, 0],
        "joint_name": "attach_joint",
        "joint_type": "FIXED",
        "link_name": link,
        "parent_link_name": tool_frame,
    }}
    kinematics.setdefault("self_collision_ignore", {})[link] = hand
    kinematics.setdefault("self_collision_buffer", {})[link] = 0.0
    return config


def make_mpc(arms="right"):
    """Reactive controller for goals that move."""
    return ModelPredictiveControl(ModelPredictiveControlCfg.create(
        robot=load_robot_config(arms), scene_model=FLOOR,
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
    position = (solver.default_joint_state.position.unsqueeze(0)
                if hasattr(solver, "default_joint_state")
                else solver.default_joint_position.clone().unsqueeze(0))
    state = JointState.from_position(position, joint_names=solver.joint_names)
    state.velocity = torch.zeros_like(state.position)
    state.acceleration = torch.zeros_like(state.position)
    return state


def tool_poses(solver, state):
    """{tool frame: (position, quaternion)} as numpy, for a joint state."""
    poses = solver.compute_kinematics(state).tool_poses.to_dict()
    return {frame: (poses[frame].position.flatten().cpu().numpy(),
                    poses[frame].quaternion.flatten().cpu().numpy())
            for frame in solver.tool_frames}


def tool_positions(solver, state):
    """{tool frame: position} as numpy, for a joint state."""
    return {frame: pose[0] for frame, pose in tool_poses(solver, state).items()}


def tool_poses_at(planner, joint_names, q):
    """Tool poses for one raw waypoint, in the plan's joint ordering."""
    state = JointState.from_position(
        torch.as_tensor(np.asarray([q]), device=DEVICE, dtype=torch.float32),
        joint_names=list(joint_names))
    return tool_poses(planner, planner.kinematics.get_active_js(state))


def plan_to(planner, targets, orientations, start, attempts=20):
    """Plan a trajectory to one tool pose per frame.

    `targets` and `orientations` are keyed by tool frame. Returns
    (joint_names, waypoints) or None.

    The interpolated plan carries every joint, including the locked ones, in a
    different order from `planner.joint_names` -- pass it through
    `planner.kinematics.get_active_js` before any forward kinematics.
    """
    frames = planner.tool_frames
    # [batch, horizon, link, goalset, dim]
    position = torch.tensor([[[[list(targets[f])] for f in frames]]],
                            device=DEVICE, dtype=torch.float32)
    quaternion = torch.tensor([[[[list(orientations[f])] for f in frames]]],
                              device=DEVICE, dtype=torch.float32)
    goal = GoalToolPose(tool_frames=frames, position=position, quaternion=quaternion)
    result = planner.plan_pose(goal, start, max_attempts=attempts, enable_graph_attempt=1)
    if result is None or not result.success.any():
        return None
    plan = result.get_interpolated_plan()
    waypoints = plan.position.reshape(-1, plan.position.shape[-1]).cpu().numpy()
    return list(plan.joint_names), waypoints


def set_mpc_goal(mpc, targets, orientations):
    """Point the controller at one tool pose per frame."""
    poses = {
        frame: Pose(
            position=torch.tensor([list(targets[frame])], device=DEVICE, dtype=torch.float32),
            quaternion=torch.tensor([list(orientations[frame])], device=DEVICE,
                                    dtype=torch.float32))
        for frame in mpc.tool_frames}
    mpc.update_goal_tool_poses(
        GoalToolPose.from_poses(poses, ordered_tool_frames=mpc.tool_frames, num_goalset=1),
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


# --------------------------------------------------------------- world editing
def _cores(planner):
    """Both solver cores. They carry independent copies of the collision world,
    so anything that changes it has to be applied to each."""
    cores = []
    for name in ("ik_solver", "trajopt_solver", "graph_planner"):
        core = getattr(getattr(planner, name, None), "core", None)
        if core is not None:
            cores.append(core)
    return cores


def set_obstacle_enabled(planner, name, enabled):
    """Toggle one world obstacle.

    Grasping needs this: with the object still a world obstacle, the fingers
    closing around it read as a collision and no plan exists. Same for the
    surface it rests on once the object is attached and its spheres touch it.
    """
    for core in _cores(planner):
        checker = getattr(core, "scene_collision_checker", None)
        if checker is not None:
            checker.enable_obstacle(name, enable=enabled)


def attach_object(planner, state, names, num_spheres=None):
    """Make world objects part of the robot, so plans account for what it carries."""
    for core in _cores(planner):
        manager = getattr(core, "attachment_manager", None)
        if manager is not None:
            manager.attach_from_scene(state, obstacle_names=list(names),
                                      num_spheres=num_spheres)


def detach_object(planner, names=None):
    """Release carried objects, optionally re-enabling them as world obstacles."""
    for core in _cores(planner):
        manager = getattr(core, "attachment_manager", None)
        if manager is not None:
            manager.detach(enable_obstacle_names=list(names) if names else None)
