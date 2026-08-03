"""Geometric impedance control on SE(3), so a plan can be *executed* rather than
written straight into qpos.

cuRobo stops at a trajectory. Everything else in this repo then teleports the
robot along it, which is fine for showing what the planner produced and useless
the moment anything touches anything. This module closes that gap: it turns a
desired tool pose trajectory into joint torques, with impedance rather than
stiff position tracking, so contact deflects the arm instead of fighting it.

The control law is the geometric impedance controller of

    Seo et al., "Geometric Formulation of Unified Force-Impedance Control on
    SE(3) for Robotic Manipulators", arXiv:2504.17080 (2025)
    reference implementation: github.com/Joohwan-Seo/GUFIC_mujoco

with one addition it does not need and this robot does: a nullspace term. That
paper's arm is a 6-DoF Indy7, where the body Jacobian is square and `J^T` fully
determines the joint torques. The chain here is 9 joints (torso lift, torso
flip, seven arm joints) for a 6-dimensional task, so `J^T` leaves a 3-dimensional
nullspace completely uncommanded and the elbow drifts under gravity while the
tool tracks perfectly. `posture_gain` pulls that nullspace toward a reference
configuration without disturbing the task.
"""
import mujoco
import numpy as np


def hat_map(w):
    """so(3): vector to skew-symmetric matrix."""
    w = np.asarray(w).reshape(-1)
    return np.array([[0.0, -w[2], w[1]],
                     [w[2], 0.0, -w[0]],
                     [-w[1], w[0], 0.0]])


def vee_map(R):
    """so(3): skew-symmetric matrix to vector."""
    return np.array([-R[1, 2], R[0, 2], -R[0, 1]]).reshape(-1, 1)


def adjoint(g):
    """Adjoint of an SE(3) element, translation block first."""
    R, p = g[:3, :3], g[:3, 3]
    adj = np.zeros((6, 6))
    adj[:3, :3] = R
    adj[3:, 3:] = R
    adj[:3, 3:] = hat_map(p) @ R
    return adj


def adjoint_derivative(g, gd, v, w, vd, wd):
    """Time derivative of `adjoint(g^-1 gd)` along the two body twists."""
    v, w = np.asarray(v).reshape(-1, 1), np.asarray(w).reshape(-1, 1)
    vd, wd = np.asarray(vd).reshape(-1, 1), np.asarray(wd).reshape(-1, 1)

    g_ed = np.linalg.inv(g) @ gd
    p_ed, R_ed = g_ed[:3, 3], g_ed[:3, :3]

    dR_ed = hat_map(w) @ R_ed - R_ed @ hat_map(wd)
    dp_ed = -v - hat_map(w) @ p_ed.reshape(-1, 1) + R_ed @ vd

    mat = np.zeros((6, 6))
    mat[:3, :3] = dR_ed
    mat[:3, 3:] = hat_map(p_ed) @ dR_ed + hat_map(dp_ed) @ R_ed
    mat[3:, 3:] = dR_ed
    return mat


def pose_matrix(position, rotation):
    g = np.eye(4)
    g[:3, :3] = rotation
    g[:3, 3] = np.asarray(position).reshape(-1)
    return g


class GeometricImpedanceController:
    """Joint torques that drive a site along a desired SE(3) trajectory.

    `dof_ids` are the velocity-space indices this controller drives. Every other
    joint in the model is somebody else's problem -- see `hold_torque`.
    """

    def __init__(self, model, data, site, dof_ids,
                 translation_gain=(900.0, 900.0, 900.0),
                 rotation_gain=(60.0, 60.0, 60.0),
                 damping_gain=(120.0, 120.0, 120.0, 12.0, 12.0, 12.0),
                 posture_gain=8.0, posture_damping=3.0):
        self.model, self.data = model, data
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        self.dof_ids = np.asarray(dof_ids, dtype=int)
        # qpos and qvel are indexed differently; the posture term reads positions
        self.qpos_ids = np.array([model.jnt_qposadr[model.dof_jntid[d]]
                                  for d in self.dof_ids], dtype=int)
        self.Kp = np.diag(translation_gain)
        self.KR = np.diag(rotation_gain)
        self.Kd = np.diag(damping_gain)
        self.posture_gain = posture_gain
        self.posture_damping = posture_damping
        self.posture_reference = None

    def tool_pose(self):
        """Site pose as (position, rotation)."""
        return (self.data.site_xpos[self.site_id].copy(),
                self.data.site_xmat[self.site_id].copy().reshape(3, 3))

    def body_jacobian(self):
        """6 x n Jacobian in the tool frame, translation rows first."""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
        world = np.vstack([jacp, jacr])[:, self.dof_ids]
        _, R = self.tool_pose()
        return np.block([[R.T, np.zeros((3, 3))], [np.zeros((3, 3)), R.T]]) @ world

    def torque(self, pd, Rd, vd, wd, dvd, dwd):
        """Torques for the driven joints, gravity and Coriolis included."""
        p, R = self.tool_pose()
        Jb = self.body_jacobian()

        g, gd = pose_matrix(p, R), pose_matrix(pd, Rd)
        g_ed = np.linalg.inv(g) @ gd

        Vd = np.hstack([np.asarray(vd).reshape(-1), np.asarray(wd).reshape(-1)]).reshape(-1, 1)
        dVd = np.hstack([np.asarray(dvd).reshape(-1), np.asarray(dwd).reshape(-1)]).reshape(-1, 1)
        Vd_star = adjoint(g_ed) @ Vd
        dVd_star = adjoint_derivative(g, gd, vd, wd, dvd, dwd) @ Vd + adjoint(g_ed) @ dVd

        # elastic wrench: the geometric error, not a naive pose difference
        fp = R.T @ Rd @ self.Kp @ Rd.T @ (p - pd).reshape(-1, 1)
        fR = vee_map(self.KR @ Rd.T @ R - R.T @ Rd @ self.KR)
        fg = np.vstack([fp, fR])

        dq = self.data.qvel[self.dof_ids].reshape(-1, 1)
        Vb = Jb @ dq
        ev = Vb - Vd_star

        full_M = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, full_M, self.data.qM)
        M = full_M[np.ix_(self.dof_ids, self.dof_ids)]
        M_tilde = np.linalg.pinv(Jb @ np.linalg.pinv(M) @ Jb.T)

        wrench = M_tilde @ dVd_star - self.Kd @ ev - fg
        tau = Jb.T @ wrench

        # The task uses six of the nine joint directions. Without this the rest
        # are uncommanded and sag under gravity while the tool still tracks.
        if self.posture_reference is not None:
            error = (self.data.qpos[self.qpos_ids] - self.posture_reference).reshape(-1, 1)
            secondary = -self.posture_gain * error - self.posture_damping * dq
            # Khatib's dynamically consistent nullspace projector. The obvious
            # choice, the orthogonal projector I - J^+ J, is geometrically a
            # nullspace projector but not a dynamically decoupled one: inertia
            # couples the posture torque back into the task, and the tool
            # settles tens of millimetres off its own setpoint. (Note that
            # I - J^T (J^T)^+ is the same orthogonal projector, not a fix.)
            # Weighting by the inertia is what actually decouples them.
            dynamically_consistent = M_tilde @ Jb @ np.linalg.pinv(M)
            projector = np.eye(len(self.dof_ids)) - Jb.T @ dynamically_consistent
            tau += projector @ secondary

        return tau.reshape(-1) + self.data.qfrc_bias[self.dof_ids]


def hold_torque(model, data, dof_ids, targets, stiffness=400.0, damping=40.0):
    """PD *plus gravity compensation* for joints the impedance controller does
    not drive.

    The head, left arm and fingers are locked as far as the planner is concerned,
    but with gravity on something still has to hold them up. The bias term is not
    optional: PD alone lets the left arm sag under its own weight, and since it
    hangs off the same torso lift as the right arm, that sag becomes a large
    disturbance on the joints the impedance controller is trying to regulate.
    """
    error = data.qpos[[model.joint(model.dof_jntid[d]).qposadr[0] for d in dof_ids]] \
        - np.asarray(targets)
    return (-stiffness * error - damping * data.qvel[dof_ids]
            + data.qfrc_bias[dof_ids])


class PoseTrajectory:
    """A joint-space plan resampled as a smooth SE(3) trajectory.

    The controller needs pose, twist and twist rate. Rather than trusting the
    planner's velocity fields to be self-consistent, this evaluates forward
    kinematics on the waypoints and differentiates, which also lets the playback
    be stretched in time independently of the planner's own timing.
    """

    def __init__(self, model, data, site, joint_names, waypoints, duration):
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        addresses = [model.joint(name).qposadr[0] for name in joint_names]

        scratch = mujoco.MjData(model)
        poses = []
        for q in waypoints:
            for i, address in enumerate(addresses):
                scratch.qpos[address] = q[i]
            mujoco.mj_forward(model, scratch)
            poses.append((scratch.site_xpos[site_id].copy(),
                          scratch.site_xmat[site_id].copy().reshape(3, 3)))

        self.positions = np.array([p for p, _ in poses])
        self.rotations = np.array([R for _, R in poses])
        self.duration = float(duration)
        self.dt = self.duration / max(len(poses) - 1, 1)

    def __len__(self):
        return len(self.positions)

    def _index(self, t):
        return int(np.clip(round(t / self.dt), 0, len(self.positions) - 1))

    def sample(self, t):
        """(pd, Rd, vd, wd, dvd, dwd) with the twists in the desired frame."""
        i = self._index(t)
        j0, j1 = max(i - 1, 0), min(i + 1, len(self.positions) - 1)
        span = max((j1 - j0), 1) * self.dt

        pd, Rd = self.positions[i], self.rotations[i]
        dpd = (self.positions[j1] - self.positions[j0]) / span
        dRd = (self.rotations[j1] - self.rotations[j0]) / span

        k0, k1 = max(i - 2, 0), min(i + 2, len(self.positions) - 1)
        wide = max((k1 - k0), 1) * self.dt
        ddpd = (self.positions[k1] - 2 * pd + self.positions[k0]) / (wide / 2) ** 2
        ddRd = (self.rotations[k1] - 2 * Rd + self.rotations[k0]) / (wide / 2) ** 2

        vd = Rd.T @ dpd.reshape(-1, 1)
        wd = vee_map(Rd.T @ dRd)
        dvd = Rd.T @ ddpd.reshape(-1, 1) - hat_map(wd) @ Rd.T @ dpd.reshape(-1, 1)
        dwd = vee_map(Rd.T @ ddRd - hat_map(wd) @ Rd.T @ dRd)
        return pd, Rd, vd.reshape(-1), wd.reshape(-1), dvd.reshape(-1), dwd.reshape(-1)
