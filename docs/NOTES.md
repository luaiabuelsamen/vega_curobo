# Notes

Things that cost real time to work out, kept here so they cost you less. Most of
these present as a solver bug and are not one.

## Planning and the scene

**A floor at z=0 makes every solve fail.** The Vega stands at the URDF origin, so
a ground plane at z≈0 intersects the base collision spheres. The start state is
in collision and cuRobo reports every plan infeasible, but IK still converges
with zero position and rotation error, so it reads as a solver bug rather than a
scene bug. Keep the floor clear of the base (`config.FLOOR` sits at z=-0.5). To
diagnose this class of failure, dump the constraints directly:

```python
metrics = planner.ik_solver.metrics_rollout.compute_metrics_from_action(q.view(1, 1, -1))
print(metrics.costs_and_constraints.constraints.names)
print(metrics.costs_and_constraints.constraints.values)
```

**MuJoCo deletes visual-only geoms from a URDF.** `MjSpec.from_file()` on a URDF
sets `compiler.discardvisual = 1`, which drops every geom with
`contype=0/conaffinity=0` at compile time. Any marker or ground plane you add
programmatically disappears: the body survives with `geomnum == 0`, so nothing
errors and the scene just renders empty. `scene.py` sets it back to 0.

**The tool frame does not exist in MuJoCo.** `R_ee` is a massless fixed link, and
the MuJoCo compiler merges those away, so `data.body("R_ee")` raises. To compare
the two engines, reconstruct it from `R_arm_l7` and the fixed transforms. Done
that way, cuRobo and MuJoCo agree on the tool position to 1e-4 mm, which is the
check worth running whenever the URDF changes.

**MPC warm-start iterations dominate the solve.** At the default the controller
took 2422 ms per solve on the Orin, slower than the one-shot planner it was meant
to replace. `warm_start_optimization_num_iters=25` brings that to 312 ms with no
visible loss of tracking. The value must be a multiple of the optimizer's
`inner_iters` (25 here), so 25 is the floor; anything smaller raises.

**One solve returns a horizon, so render all of it.** Advancing to the last state
and drawing one frame animates at the solve rate and looks like a slideshow.
Drawing every step of `action_sequence.position[0]` puts the frame rate at
roughly four times the solve rate for the same motion.

**Do not lock the left arm at 0.0.** Zero is not a neutral pose for this robot:
it straightens the tucked left arm into the right arm's workspace, and plans then
start in self-collision. Lock unused joints at their builder defaults, which is
what `build_config.py` does.

**Not every pose in the reachable box is reachable at any orientation.** The
7-DoF arm cannot hold an arbitrary tool orientation across the workspace. The box
in `config.py` was sampled at the home orientation and hit 40/40; change the
orientation and you need rejection sampling or a fallback.

**Going bimanual needs no refit.** The builder fits collision spheres to all 28
links regardless of which joints you drive, and locking a joint only removes it
from the active chain. So the second arm costs nothing offline: unlock the left
joints, add `L_ee` to `tool_frames`, and the same config drives 18 joints instead
of 10. It is not free at runtime, though, and the two solvers degrade very
differently. MPC goes from 312 ms to ~870 ms per solve, while a one-shot plan
goes from 12 s to 125 s, because a second tool constraint is much harder for the
global search than for a local refinement.

**Cross-midline targets defeat MPC in bimanual mode.** With both arms active, a
goal on the opposite side of the body is where the controller stalls: in a
3-ball-per-hand run the only two misses were a left-hand ball at y=-0.117 and a
right-hand ball at y=+0.048. The per-hand boxes were each validated against the
*planner*, which searches globally; MPC only refines locally, and with the other
arm and the shared torso in the same optimisation it settles into a local
minimum instead of reaching across. Raise `--patience`, or keep each hand on its
own side.

**Interpolated plans use a different joint ordering.** `get_interpolated_plan()`
returns all 21 joints, including locked ones, ordered differently from
`planner.joint_names`. Pass a state through `planner.kinematics.get_active_js`
before any forward kinematics, or it raises on the joint-name mismatch.

## Grasping

**The tool frame is not where the fingers are.** `R_ee` is at the wrist mount and
the jaw midpoint is 15.3 cm further along the tool's local +z, with the jaws
closing along local x. Both numbers came from measuring the gripper's geometry
at its joint limits, not from the URDF or a guess. Planning straight to an
object's position drives the wrist into it.

**Grasp feasibility has to be checked with the object disabled.** At the grasp
pose the jaws surround the bottle, so while it is still a world obstacle every
candidate approach fails IK on collision. Leave it enabled during the search and
you get "no reachable grasp" while every pre-grasp pose passes, which reads as an
unreachable object rather than a bookkeeping mistake.

**Check the whole manoeuvre, not just the grasp.** The gripper holds one
orientation from approach to release, so an orientation that works at the grasp
is worth nothing if it fails at the place. Searching on the grasp alone picked
yaw=-45 here, which planned three segments and then had no solution at the
transport pose. The demo defines the sequence once and the search runs over all
six poses; 21 of the 39 candidate orientations satisfy all of them.

**Attaching a payload needs a link that does not exist yet.** Setting
`add_object_link: True` is not enough: the attachment managers look up a link
literally named `attached_object` and raise `KeyError` without it. It has to be
declared as an extra link fixed to the tool frame, with a pool of empty sphere
slots and self-collision ignored against the hand. The pool must be at least as
large as the fit -- a box-shaped payload wanted 34 spheres against a 32-slot
allocation and the attach raised rather than fitting fewer. Both solver cores
hold their own copy, so attach, detach and obstacle toggles apply to each.

**CUDA graphs and mixed problem shapes do not combine.** The solver captures a
graph at the first problem shape it sees; a later call with a different shape
raises "CUDA graph reset is not available" instead of recapturing. Querying IK
for grasp feasibility and then planning is exactly that pattern, so this demo
builds its planner with `use_cuda_graph=False`.

**One of the YCB collision hulls renders wrong.** `006_mustard_bottle/collision.obj`
reports the right vertex extents, and MuJoCo agrees on those extents, but it
draws about half size: measured against a box of identical stated dimensions at
matched depth, 47 px against 119 px. `textured_simple.obj` renders correctly
(116 px against 119, right for a tapering bottle) and is what this repo ships.
Worth knowing that a mesh's vertex data and what MuJoCo draws can disagree; a
segmentation render against a known-size primitive is the way to settle it.

## Execution and control

**A URDF gives you nothing to command.** It imports with zero actuators and zero
sensors. `build_scene(physics=True)` adds a torque motor per joint, a site at
the tool frame, and gravity.

**The tool frame needs a site, not a body.** `R_ee` is merged away by the
compiler, and `mj_jacSite` needs something that exists to hang a Jacobian on.
The site is placed at the same two fixed transforms and agrees with cuRobo's
tool position exactly.

**Torque control needs armature this URDF does not have.** Armature and damping
are zero throughout and some links have diagonal inertias down to 2e-5. On that
model even *exact gravity compensation* — which should leave the arm at rest —
diverged to |qvel| = 160. Adding armature (0.1 arm, 0.01 fingers) and damping
1.0 brings the same test to |qvel| = 0.14. Armature is reflected motor and gear
inertia: physically real, missing from the URDF, and it does not move where the
arm settles.

**Joints you are not driving still need gravity compensation.** PD alone lets the
left arm sag under its own weight, and because it hangs off the same torso lift
as the right arm, that sag is a large disturbance on exactly the joints the
impedance controller is regulating.

**The nullspace term is off by default, and the experiment says why.** The chain
is 9 joints for a 6-dimensional task, so `J^T` leaves three directions
uncommanded and the elbow wanders. A posture spring projected into the nullspace
should fix that. It does not pay: at gains that meaningfully reduce the drift
(1.85 rad to 1.40 rad) the tool error goes from 0.26 mm to 67 mm. Neither
projector choice helps — note that `I - J^+ J` and `I - J^T (J^T)^+` are the
same orthogonal projector, and Khatib's inertia-weighted one behaves the same
here. Stiffening the joints that are merely held changes nothing either
(400 to 50000 moves the error by 0.5 mm). Left in behind `--posture`, off by
default, as an honest negative result rather than a knob that looks like it works.

**The prismatic `Lift` joint does not respect its limit, and the controller is
not why.** It ends about 9 cm past its 0.4 m upper bound. With no controller, no
actuators, no gravity and no edits to the model, starting it at 0.30 still
leaves it at 0.4900:

```python
m = mujoco.MjModel.from_xml_path("assets/vega.urdf")
m.opt.gravity[:] = [0, 0, 0]
d = mujoco.MjData(m); d.qpos[m.joint("Lift").qposadr[0]] = 0.30
for _ in range(1500): mujoco.mj_step(m, d)
# -> 0.4900, against a declared range of [0, 0.4]
```

cuRobo and MuJoCo agree the range is [0, 0.4] and `jnt_limited` is set, so this
is something about the imported model rather than a disagreement between the two.
Worth knowing before trusting sim contact forces near that joint.
