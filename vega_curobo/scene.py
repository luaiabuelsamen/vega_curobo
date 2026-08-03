"""The MuJoCo side: one scene, built from the same URDF cuRobo plans against.

Nothing here is specific to a demo. `build_scene` returns a compiled model with
a marker body you can move via `data.mocap_pos[0]`, and `Recorder` handles both
output paths (window or mp4) so the demos stay short.
"""
import os

import mujoco
import numpy as np

from .config import URDF

WIDTH, HEIGHT = 1280, 720

#: Site standing in for the cuRobo tool frame. `R_ee` is a massless fixed link
#: that the compiler merges away, so it cannot be queried in MuJoCo -- but a
#: site placed at the same transform can, and unlike a body it also gives
#: `mj_jacSite` something to hang a Jacobian on.
TOOL_SITE = "tool"
TOOL_PARENT = "R_arm_l7"
TOOL_OFFSET = (0.11597, 0.0, -0.032)      # R_arm_j8, then R_ee_j0's yaw
TOOL_RPY = (0.0, 1.57079, 1.57079)


def look_at(eye, target):
    """xyaxes for a camera at `eye` aimed at `target` (MuJoCo looks down -z)."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    return list(right) + list(np.cross(right, forward))


#: One marker colour per hand, so a bimanual scene reads at a glance.
MARKER_COLOURS = {"R_ee": (0.15, 0.80, 0.30, 0.9), "L_ee": (0.95, 0.55, 0.15, 0.9)}


def _tool_quaternion():
    """Quaternion for Ry then Rz, the two fixed rotations between R_arm_l7 and R_ee."""
    _, pitch, yaw = TOOL_RPY
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    R = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @ \
        np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R.reshape(-1))
    return quat


def build_scene(markers=None, table=None, meshes=None, physics=False,
                obstacles=None, supports=None):
    """Compile the Vega with a ground plane, lights, markers and a framed camera.

    `markers` maps a name (use the tool frame) to a position; each becomes a
    mocap body you can move with `set_marker`. Defaults to a single right-hand
    marker.

    `table` is a (centre, half_extents) pair. `meshes` maps a name to
    (obj_path, position, rgba) and each becomes a mocap body, so a demo can carry
    an object around by writing its pose rather than simulating a grip.

    `supports` maps a name to (position, half_extents) and each becomes a static
    collidable box -- a pedestal for an obstacle to rest on, since a free body
    with nothing under it simply falls out of the scene before the arm arrives.

    `obstacles` maps a name to (position, half_extents, mass) and each becomes a
    *collidable, free-floating* box. Everything else this function adds is
    contype=0 decoration; these are the only props the robot can actually hit,
    which is the point when the demo is about what happens on contact.

    Kinematic by default: gravity is off and the demos write joint angles
    straight into qpos, because what is being shown is the solver's output, not
    contact physics.

    `physics=True` turns gravity on and adds a torque actuator per joint plus the
    tool site, which is what an actual controller needs. A URDF imports with no
    actuators and no sensors at all, so without this there is nothing to command.
    """
    spec = mujoco.MjSpec.from_file(URDF)

    # URDF import defaults to discardvisual=1, which deletes every geom with
    # contype=0/conaffinity=0 at compile time. That silently removes the ground
    # plane and the marker below -- the bodies survive with zero geoms, so the
    # scene looks empty rather than erroring. The robot is unaffected either way.
    spec.compiler.discardvisual = 0
    spec.option.gravity = [0, 0, 0]
    spec.visual.global_.offwidth = WIDTH
    spec.visual.global_.offheight = HEIGHT

    world = spec.worldbody

    # A URDF carries no lights, and the default headlight alone renders the robot
    # almost black.
    directional = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    world.add_light(pos=[1.5, -1.5, 3.0], dir=[-0.4, 0.4, -1.0], type=directional,
                    diffuse=[0.75, 0.75, 0.75], specular=[0.2, 0.2, 0.2])
    world.add_light(pos=[-1.5, 1.5, 2.5], dir=[0.4, -0.4, -1.0], type=directional,
                    diffuse=[0.35, 0.35, 0.38], specular=[0.0, 0.0, 0.0])

    world.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.05],
                   rgba=[0.82, 0.84, 0.87, 1], contype=0, conaffinity=0)

    if table is not None:
        centre, half = table
        world.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=list(centre), size=list(half),
                       rgba=[0.55, 0.42, 0.30, 1], contype=0, conaffinity=0)

    for name, (path, position, rgba) in (meshes or {}).items():
        spec.add_mesh(name=name, file=path)
        body = world.add_body(name=f"object_{name}", pos=list(position), mocap=True)
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, rgba=list(rgba),
                      contype=0, conaffinity=0)

    for name, (position, half) in (supports or {}).items():
        world.add_geom(name=f"support_{name}", type=mujoco.mjtGeom.mjGEOM_BOX,
                       pos=list(position), size=list(half), rgba=[0.45, 0.45, 0.5, 1])

    for name, (position, half, mass) in (obstacles or {}).items():
        body = world.add_body(name=f"obstacle_{name}", pos=list(position))
        body.add_freejoint()
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=list(half),
                      rgba=[0.85, 0.30, 0.25, 1], mass=mass,
                      friction=[1.0, 0.005, 0.0001])

    if markers is None:
        markers = {"R_ee": (0.72, -0.15, 1.05)}
    for name, position in markers.items():
        body = world.add_body(name=f"marker_{name}", pos=list(position), mocap=True)
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.045, 0, 0],
                      rgba=list(MARKER_COLOURS.get(name, (0.15, 0.8, 0.3, 0.9))),
                      contype=0, conaffinity=0)

    if physics:
        spec.option.gravity = [0, 0, -9.81]
        # The URDF specifies no armature and no damping anywhere, and some links
        # have diagonal inertias down to 2e-5. Torque control on that is
        # numerically marginal: even exact gravity compensation, which should
        # leave the arm at rest, diverges. Armature is reflected motor and gear
        # inertia -- physically real, absent from the URDF, and it does not
        # change where the arm settles.
        for joint in spec.joints:
            if joint.type == mujoco.mjtJoint.mjJNT_FREE:
                continue
            finger = "gripper" in joint.name
            joint.armature = 0.01 if finger else 0.1
            joint.damping = [0.5, 0.0, 0.0] if finger else [1.0, 0.0, 0.0]
        parent = spec.body(TOOL_PARENT)
        parent.add_site(name=TOOL_SITE, pos=list(TOOL_OFFSET),
                        quat=list(_tool_quaternion()), size=[0.01, 0, 0],
                        rgba=[0.9, 0.2, 0.2, 0.0])
        for joint in spec.joints:
            if joint.type == mujoco.mjtJoint.mjJNT_FREE:
                continue
            spec.add_actuator(name=f"{joint.name}_motor", target=joint.name,
                              trntype=mujoco.mjtTrn.mjTRN_JOINT,
                              gear=[1, 0, 0, 0, 0, 0],
                              ctrllimited=False, ctrlrange=[-1e6, 1e6],
                              forcelimited=False, forcerange=[-1e6, 1e6])

    eye = [1.9, -1.5, 1.55]
    world.add_camera(name="demo", pos=eye, fovy=50, xyaxes=look_at(eye, [0.35, -0.05, 1.05]))
    return spec.compile()


def set_marker(model, data, name, position):
    """Move one marker. Mocap bodies index into `mocap_pos` by mocapid, which is
    not the body id."""
    data.mocap_pos[model.body(f"marker_{name}").mocapid[0]] = position


def set_object(model, data, name, position, quaternion=None):
    """Move a mesh object placed by `build_scene(meshes=...)`.

    Poses are in the frame the mesh file was authored in. The compiler does
    re-frame mesh assets onto their principal axes of inertia -- `mesh_quat` is
    nowhere near identity for an asymmetric object -- but it compensates on the
    geom, so no correction belongs here. Applying one rotates the object twice.
    """
    index = model.body(f"object_{name}").mocapid[0]
    data.mocap_pos[index] = position
    if quaternion is not None:
        data.mocap_quat[index] = quaternion


def joint_writer(model, data, joint_names):
    """Return `set(q)`, writing a solver joint vector into qpos by joint name."""
    addresses = [model.joint(name).qposadr[0] for name in joint_names]

    def set_joints(q):
        for i, address in enumerate(addresses):
            data.qpos[address] = q[i]
        mujoco.mj_forward(model, data)

    return set_joints


def apply_locked_joints(model, data, robot_config):
    """Pose the joints cuRobo holds fixed.

    They are absent from the solver's output, so without this the tucked left arm
    renders stretched out at qpos 0.
    """
    locked = robot_config.get("kinematics", {}).get("lock_joints") or {}
    for name, value in locked.items():
        try:
            data.qpos[model.joint(name).qposadr[0]] = value
        except KeyError:
            pass                                    # merged away by the compiler
    mujoco.mj_forward(model, data)


class Recorder:
    """Show the scene in a window, or collect frames and write an mp4.

    Offscreen EGL rendering is the reliable path on Jetson, where the interactive
    viewer intermittently segfaults on the board's GL driver.
    """

    def __init__(self, model, data, mode="live", out="media/demo.mp4", fps=30):
        self.model, self.data, self.mode, self.out, self.fps = model, data, mode, out, fps
        self.frames = []
        self.viewer = self.renderer = None
        if mode == "live":
            # bound to its own name: `import mujoco.viewer` here would make
            # `mujoco` a local and shadow the module for the rest of this scope
            import mujoco.viewer as mujoco_viewer
            self.viewer = mujoco_viewer.launch_passive(model, data)
        else:
            self.renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
            self.camera = model.camera("demo").id

    @property
    def running(self):
        return self.viewer is None or self.viewer.is_running()

    def capture(self):
        if self.viewer is not None:
            self.viewer.sync()
        else:
            self.renderer.update_scene(self.data, camera=self.camera)
            self.frames.append(self.renderer.render())

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            return
        self.renderer.close()
        if not self.frames:
            return
        import imageio.v2 as imageio
        os.makedirs(os.path.dirname(self.out) or ".", exist_ok=True)
        imageio.mimsave(self.out, self.frames, fps=self.fps, quality=8)
        print(f"wrote {self.out} ({len(self.frames)} frames)")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
