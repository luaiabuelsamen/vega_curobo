"""Regenerate configs/vega_right.yml from a Dexmate Vega URDF.

Only needed if you change the robot. The checked-in config was produced by this
script; the sphere fit takes ~8 minutes over 28 links.

    python scripts/build_config.py --urdf /path/to/vega_1u_gripper-obj.urdf

No hand-placed collision spheres: cuRobo's RobotBuilder fits them to each link
mesh and derives the self-collision ignore matrix. The joints we don't drive are
then locked, which leaves the active chain at Lift + torso_flip + R_arm_j1..j7 +
R_gripper_j1.
"""
import argparse
import os
import re
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from curobo.robot_builder import RobotBuilder  # noqa: E402

from vega_curobo.config import ROBOT_CONFIG, ROOT, TOOL_FRAME  # noqa: E402

LOCKED = ["head_j1", "head_j2", "head_j3",
          "L_arm_j1", "L_arm_j2", "L_arm_j3", "L_arm_j4",
          "L_arm_j5", "L_arm_j6", "L_arm_j7", "L_gripper_j1"]


def repair_urdf(source, destination):
    """Resolve mesh paths against the filesystem and clamp degenerate limits.

    The stock URDF carries stale '/obj/' path segments and some zero velocity and
    effort limits, which cuRobo rejects. Everything here is a filesystem check or
    a numeric clamp, so it stays correct if the mesh layout changes.
    """
    base = os.path.dirname(source)
    text = open(source).read()
    unresolved = []

    def resolve(match):
        relative = match.group(1)
        variants = [relative,
                    relative.replace("/obj/", "/"),
                    relative.replace("meshes/obj/", "meshes/"),
                    relative.replace("/meshes/", "/meshes/collision/")]
        for variant in variants:
            candidate = os.path.normpath(os.path.join(base, variant))
            if os.path.isfile(candidate):
                return f'filename="{candidate}"'
        unresolved.append(relative)
        return match.group(0)

    text = re.sub(r'filename="([^"]+)"', resolve, text)
    text = re.sub(r'velocity="0(\.0+)?"', 'velocity="1.0"', text)
    text = re.sub(r'effort="0(\.0+)?"', 'effort="50.0"', text)
    open(destination, "w").write(text)
    print(f"  {len(unresolved)} unresolved mesh paths" if unresolved else "  mesh paths resolved")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", required=True, help="source Vega URDF")
    parser.add_argument("--assets", help="mesh root (defaults to the URDF's directory)")
    parser.add_argument("--out", default=ROBOT_CONFIG)
    args = parser.parse_args()

    repaired = os.path.join(ROOT, "assets", "vega.urdf")
    print("repairing URDF ...")
    repair_urdf(args.urdf, repaired)

    print("fitting collision spheres and self-collision matrix (~8 min) ...")
    builder = RobotBuilder(repaired, asset_path=args.assets or os.path.dirname(args.urdf),
                           tool_frames=[TOOL_FRAME])
    builder.fit_collision_spheres()
    builder.compute_collision_matrix()
    builder.save(builder.build(), args.out)

    config = yaml.safe_load(open(args.out))
    kinematics = config["kinematics"]

    # Lock the unused joints AT THEIR BUILDER DEFAULTS, not at 0.0. Zero is not a
    # neutral pose here: it straightens the tucked left arm out into the right
    # arm's workspace, so every plan starts in self-collision.
    defaults = dict(zip(kinematics["cspace"]["joint_names"],
                        kinematics["cspace"]["retract_config"]))
    locked = kinematics.get("lock_joints") or {}
    for joint in LOCKED:
        if joint in defaults:
            locked[joint] = float(defaults[joint])
    kinematics["lock_joints"] = locked

    # Keep the config relocatable: the loader substitutes {ROOT} at read time.
    kinematics["asset_root_path"] = "{ROOT}/assets"
    kinematics["urdf_path"] = "{ROOT}/assets/vega.urdf"
    yaml.safe_dump(config, open(args.out, "w"), sort_keys=False)

    print("active joints:", kinematics["cspace"]["joint_names"])
    print("locked:", sorted(locked))
    print("wrote", args.out)
    print("\nRest-pose sphere overlaps can still show up as false self-collisions.")
    print("If plans fail at home, run: python scripts/relax_self_collision.py")


if __name__ == "__main__":
    main()
