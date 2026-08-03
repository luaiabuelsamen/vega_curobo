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


def look_at(eye, target):
    """xyaxes for a camera at `eye` aimed at `target` (MuJoCo looks down -z)."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    return list(right) + list(np.cross(right, forward))


def build_scene(marker_pos=(0.72, -0.15, 1.05), marker_rgba=(0.15, 0.8, 0.3, 0.9)):
    """Compile the Vega with a ground plane, lights, a marker and a framed camera.

    Kinematic only: gravity is off and the demos write joint angles straight into
    qpos, because what is being shown is the solver's output, not contact physics.
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

    marker = world.add_body(name="marker", pos=list(marker_pos), mocap=True)
    marker.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.045, 0, 0],
                    rgba=list(marker_rgba), contype=0, conaffinity=0)

    eye = [1.9, -1.5, 1.55]
    world.add_camera(name="demo", pos=eye, fovy=50, xyaxes=look_at(eye, [0.35, -0.05, 1.05]))
    return spec.compile()


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
