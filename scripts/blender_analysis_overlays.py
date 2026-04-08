"""
blender_analysis_overlays.py — biomechanics overlays for the SAM 3D Body pitcher.

Run AFTER blender_field_viewer.py has loaded a pitch. Reads the .npz path
from the PitcherMesh object and creates visualization overlays.

  Blender → Text Editor → Open → blender_analysis_overlays.py → Alt+P

EXPORTABLE (survives GLB export to Three.js):
  - Joint marker spheres (shoulder/elbow/wrist/hip/knee/ankle for both sides),
    parented to PitcherMesh, keyframed per-frame from joints_mhr70.
  - Motion path polylines (world-space) for throw_shoulder / throw_elbow /
    throw_wrist, as static mesh line objects.
  - Strike zone wireframe box at home plate.
  - Release point sphere (static) at the detected release frame.
  - Stride ruler on the ground showing pivot-to-stride distance.
  - Timeline markers at detected pitch phase frames.

BLENDER-ONLY (does NOT survive GLB export):
  - Live elbow-angle text with a frame-change handler. The browser version
    should recompute this in JS from the baked joint animation.

The field viewer's per-frame foot-lock handler moves PitcherMesh.location
each frame; world-space overlays read that translation directly rather
than re-deriving it.
"""

import bpy
import math
import os
import sys
import numpy as np
from mathutils import Vector, Matrix

# ─── Make blender_field_viewer importable for constant/helper reuse ──────────
#
# Blender's Text Editor doesn't set __file__ to a real path for scripts opened
# via Text → Open (it ends up as something like '/blender_analysis_overlays.py'
# in 5.1). We walk bpy.data.texts to find the real on-disk path of ANY loaded
# blender_*.py script and use its directory. SAMPLAYS_SCRIPTS_DIR env var is a
# manual override if the search fails.


def _find_script_dir() -> str | None:
    # 1. __file__ IF it points to an existing file
    try:
        f = os.path.abspath(__file__)  # type: ignore[name-defined]
        if os.path.isfile(f):
            return os.path.dirname(f)
    except Exception:
        pass

    # 2. Env var override
    env_dir = os.environ.get("SAMPLAYS_SCRIPTS_DIR", "").strip()
    if env_dir and os.path.isdir(env_dir):
        return env_dir

    # 3. Any Text block in the current .blend whose filepath points at a
    #    blender_*.py on disk
    try:
        for text in bpy.data.texts:
            fp = text.filepath
            if not fp:
                continue
            resolved = bpy.path.abspath(fp)
            if os.path.isfile(resolved) and os.path.basename(resolved).startswith("blender_"):
                candidate_dir = os.path.dirname(resolved)
                if os.path.isfile(os.path.join(candidate_dir, "blender_field_viewer.py")):
                    return candidate_dir
    except Exception:
        pass

    # 4. Common hardcoded fallback for this project
    candidates = [
        os.path.expanduser("~/Source/SamPlaysBaseball/scripts"),
    ]
    for c in candidates:
        if os.path.isfile(os.path.join(c, "blender_field_viewer.py")):
            return c

    return None


_SCRIPT_DIR = _find_script_dir()
if _SCRIPT_DIR is None:
    raise RuntimeError(
        "Cannot locate SamPlaysBaseball scripts directory.\n"
        "Either:\n"
        "  - Run blender_field_viewer.py from the Text Editor first (keeps\n"
        "    its path registered in bpy.data.texts), or\n"
        "  - Set the SAMPLAYS_SCRIPTS_DIR env var before launching Blender."
    )
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import blender_field_viewer as bfv  # noqa: E402

# ─── MHR70 joint indices (mirrors backend/app/pipeline/joint_map.py) ─────────

MHR70 = {
    "left_shoulder": 5,  "right_shoulder": 6,
    "left_elbow":    7,  "right_elbow":    8,
    "left_hip":      9,  "right_hip":      10,
    "left_knee":     11, "right_knee":     12,
    "left_ankle":    13, "right_ankle":    14,
    "right_wrist":   41,
    "left_wrist":    62,
}

# Name → color (RGBA). Joint index is resolved per-handedness at runtime.
MARKER_JOINTS = {
    "throw_shoulder":  (1.00, 0.20, 0.20, 1.0),
    "throw_elbow":     (1.00, 0.55, 0.00, 1.0),
    "throw_wrist":     (1.00, 1.00, 0.00, 1.0),
    "glove_shoulder":  (0.20, 0.50, 1.00, 1.0),
    "throw_hip":       (0.60, 0.20, 0.60, 1.0),
    "glove_hip":       (0.30, 0.30, 0.80, 1.0),
    "stride_knee":     (0.20, 0.80, 0.30, 1.0),
    "stride_ankle":    (0.10, 0.60, 0.20, 1.0),
    "pivot_ankle":     (0.55, 0.55, 0.55, 1.0),
}

MARKER_SPHERE_RADIUS = 0.04  # 4 cm — visible but not obtrusive

MOTION_PATH_JOINTS = ["throw_shoulder", "throw_elbow", "throw_wrist"]

# Strike zone: 17" × 17" at home plate, knee-to-letters height.
# Real strike zone is batter-dependent. These are median values.
SZ_WIDTH_M  = 17 * bfv.IN_TO_M           # 0.432m (plate width)
SZ_DEPTH_M  = 17 * bfv.IN_TO_M
SZ_BOTTOM_M = 0.46                       # ~kneecap height
SZ_TOP_M    = 1.06                       # ~midpoint between hips and shoulders


# ─── HELPERS ─────────────────────────────────────────────────────────────────


def _pipeline_to_blender(v: np.ndarray) -> np.ndarray:
    """Same axis mapping as blender_field_viewer._import_npz: (-x, -z, -y)."""
    out = np.empty_like(v, dtype=np.float32)
    out[..., 0] = -v[..., 0]
    out[..., 1] = -v[..., 2]
    out[..., 2] = -v[..., 1]
    return out


def _resolve_marker_indices(handedness: str) -> dict:
    """Map role names to (mhr70_idx, color) based on pitcher handedness."""
    r = handedness == "right"
    role_to_idx = {
        "throw_shoulder":  MHR70["right_shoulder"] if r else MHR70["left_shoulder"],
        "throw_elbow":     MHR70["right_elbow"]    if r else MHR70["left_elbow"],
        "throw_wrist":     MHR70["right_wrist"]    if r else MHR70["left_wrist"],
        "glove_shoulder":  MHR70["left_shoulder"]  if r else MHR70["right_shoulder"],
        "throw_hip":       MHR70["right_hip"]      if r else MHR70["left_hip"],
        "glove_hip":       MHR70["left_hip"]       if r else MHR70["right_hip"],
        "stride_knee":     MHR70["left_knee"]      if r else MHR70["right_knee"],
        "stride_ankle":    MHR70["left_ankle"]     if r else MHR70["right_ankle"],
        "pivot_ankle":     MHR70["right_ankle"]    if r else MHR70["left_ankle"],
    }
    return {name: (role_to_idx[name], color) for name, color in MARKER_JOINTS.items()}


def _get_or_create_collection(name: str) -> bpy.types.Collection:
    if name in bpy.data.collections:
        return bpy.data.collections[name]
    coll = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(coll)
    return coll


def _move_to_collection(obj: bpy.types.Object, coll: bpy.types.Collection):
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    coll.objects.link(obj)


def _make_colored_material(name: str, color: tuple, emission: float = 0.0) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = 0.3
    # "Emission Color" / "Emission Strength" exist on Blender 4.x+ Principled
    # BSDF. Fall back silently if the user is on an older build.
    if emission > 0 and "Emission Color" in bsdf.inputs:
        bsdf.inputs["Emission Color"].default_value = color
        bsdf.inputs["Emission Strength"].default_value = emission
    return mat


def _make_sphere(name: str, radius: float, color: tuple) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, segments=16, ring_count=8)
    sphere = bpy.context.active_object
    sphere.name = name
    for poly in sphere.data.polygons:
        poly.use_smooth = True
    sphere.data.materials.append(_make_colored_material(f"{name}_Mat", color))
    return sphere


# ─── DATA LOAD ───────────────────────────────────────────────────────────────


def find_pitcher_mesh() -> bpy.types.Object:
    obj = bpy.data.objects.get("PitcherMesh")
    if obj is None:
        print("ERROR: PitcherMesh not found. Run blender_field_viewer.py first.")
        return None
    if "npz_path" not in obj.keys():
        print("ERROR: PitcherMesh has no 'npz_path' custom property.")
        print("       Re-run blender_field_viewer.py with an .npz input.")
        return None
    return obj


def load_joint_data(pitcher_obj):
    """Return (joints_mhr70, handedness) or (None, None).

    Handedness is resolved in the same priority order as the field viewer:
    NPZ 'handedness' field → DEFAULT_HANDEDNESS env/constant → "right".
    Both scripts must agree, otherwise the joint markers stick to the wrong
    arm and stride/release measurements come out tiny.
    """
    npz_path = pitcher_obj["npz_path"]
    if not os.path.exists(npz_path):
        print(f"ERROR: .npz no longer at {npz_path}")
        return None, None
    data = np.load(npz_path)
    if "joints_mhr70" not in data:
        print("ERROR: .npz has no joints_mhr70 array.")
        return None, None
    joints = data["joints_mhr70"]  # (T, 70, 3) pipeline coords

    if "handedness" in data:
        handedness = str(data["handedness"]).lower().strip()
    else:
        handedness = getattr(bfv, "DEFAULT_HANDEDNESS", "right")
    if not handedness.startswith("l"):
        handedness = "right"
    else:
        handedness = "left"
    return joints, handedness


# ─── JOINT MARKERS ───────────────────────────────────────────────────────────


def create_joint_markers(joints: np.ndarray, pitcher_obj, handedness: str) -> dict:
    """One colored sphere per tracked joint, parented to PitcherMesh so it
    inherits the per-frame foot-lock offset. Local position is keyframed
    from joints_mhr70.

    Parent + keyframed local position is the pattern that works cleanly
    through glTF export: Three.js replays the local keyframes and the
    parent's object-level animation on top of them.
    """
    T = int(joints.shape[0])
    marker_specs = _resolve_marker_indices(handedness)
    coll = _get_or_create_collection("JointMarkers")

    created = {}
    for name, (joint_idx, color) in marker_specs.items():
        sphere = _make_sphere(f"Joint_{name}", MARKER_SPHERE_RADIUS, color)
        _move_to_collection(sphere, coll)

        # Parent WITHOUT the default keep-transform inverse so local == world
        # at parent-relative zero, then keyframe local position per frame.
        sphere.parent = pitcher_obj
        sphere.matrix_parent_inverse = Matrix.Identity(4)

        blender_joint_pos = _pipeline_to_blender(joints[:, joint_idx, :])  # (T, 3)
        for t in range(T):
            sphere.location = blender_joint_pos[t].tolist()
            sphere.keyframe_insert(data_path="location", frame=t)

        created[name] = sphere

    print(f"  Created {len(created)} joint markers x {T} keyframes each")
    return created


# ─── MOTION PATHS ────────────────────────────────────────────────────────────


def _compute_world_joint_track(joints: np.ndarray, joint_idx: int, pitcher_obj) -> np.ndarray:
    """Return (T, 3) world-space positions for a joint across the clip.

    Frame-steps through all frames so the foot-lock handler updates
    PitcherMesh.location, then reads parent location directly. Restores
    the original scene frame on exit.
    """
    T = int(joints.shape[0])
    blender_local = _pipeline_to_blender(joints[:, joint_idx, :])  # (T, 3)
    world = np.zeros((T, 3), dtype=np.float32)

    scene = bpy.context.scene
    original_frame = scene.frame_current
    try:
        for t in range(T):
            scene.frame_set(t)
            loc = pitcher_obj.location
            world[t, 0] = blender_local[t, 0] + loc.x
            world[t, 1] = blender_local[t, 1] + loc.y
            world[t, 2] = blender_local[t, 2] + loc.z
    finally:
        scene.frame_set(original_frame)
    return world


def _bake_motion_path(name: str, world_track: np.ndarray, color: tuple):
    """Build a static tube along world_track using a Curve object with
    bevel depth. Curves with bevel are auto-converted to mesh on glTF
    export, so the tube actually shows up in the exported GLB — unlike
    edge-only meshes which the exporter silently drops with a warning.
    """
    T = int(world_track.shape[0])
    if T < 2:
        return None

    curve_data = bpy.data.curves.new(name, type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.bevel_depth = 0.015      # 1.5cm tube radius — thick enough
                                        # to read against grass from any cam
    curve_data.bevel_resolution = 2     # octagonal cross-section (8 sides)
    curve_data.use_fill_caps = True

    spline = curve_data.splines.new(type='POLY')
    # POLY splines start with 1 point; add the rest.
    spline.points.add(T - 1)
    for i in range(T):
        x, y, z = world_track[i].tolist()
        spline.points[i].co = (x, y, z, 1.0)  # (x, y, z, w) for POLY

    obj = bpy.data.objects.new(name, curve_data)
    bpy.context.collection.objects.link(obj)

    # Emission material so the tube glows against the grass.
    obj.data.materials.append(_make_colored_material(f"{name}_Mat", color, emission=3.0))
    return obj


def create_motion_paths(joints: np.ndarray, pitcher_obj, handedness: str):
    marker_specs = _resolve_marker_indices(handedness)
    coll = _get_or_create_collection("MotionPaths")

    for name in MOTION_PATH_JOINTS:
        joint_idx, color = marker_specs[name]
        world_track = _compute_world_joint_track(joints, joint_idx, pitcher_obj)
        obj = _bake_motion_path(f"MotionPath_{name}", world_track, color)
        if obj:
            _move_to_collection(obj, coll)

    print(f"  Baked {len(MOTION_PATH_JOINTS)} world-space motion paths")


# ─── STRIKE ZONE ─────────────────────────────────────────────────────────────


def create_strike_zone():
    """Wireframe box at home plate, knee-to-letters. Static world position.
    Display type = WIRE in Blender; the mesh still exports as geometry for
    Three.js (where it can be rendered as wireframe or transparent)."""
    plate_y = -bfv.MOUND_TO_PLATE_M
    center = Vector((
        0.0,
        plate_y + bfv.PLATE_SIDE_M / 2,
        (SZ_BOTTOM_M + SZ_TOP_M) / 2,
    ))

    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    box = bpy.context.active_object
    box.name = "StrikeZone"
    box.scale = (SZ_WIDTH_M / 2, SZ_DEPTH_M / 2, (SZ_TOP_M - SZ_BOTTOM_M) / 2)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    box.display_type = 'WIRE'
    box.show_in_front = True

    mat = bpy.data.materials.new("StrikeZoneMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    if "Alpha" in bsdf.inputs:
        bsdf.inputs["Alpha"].default_value = 0.25
    mat.blend_method = 'BLEND'
    box.data.materials.append(mat)
    return box


# ─── RELEASE POINT + STRIDE RULER ────────────────────────────────────────────


def _find_phase_frame(joints: np.ndarray, handedness: str, label: str):
    phases = bfv._detect_pitch_phases(joints, handedness=handedness)
    for frame, phase_label in phases.items():
        if phase_label == label:
            return int(frame)
    return None


def _world_pos_at_frame(joints: np.ndarray, frame: int, joint_idx: int, pitcher_obj) -> tuple:
    """Return (x, y, z) world position of a joint at a specific frame,
    including the foot-lock offset. Restores scene frame on exit."""
    scene = bpy.context.scene
    original_frame = scene.frame_current
    try:
        scene.frame_set(frame)
        local = _pipeline_to_blender(joints[frame:frame + 1, joint_idx, :])[0]
        loc = pitcher_obj.location
        return (
            float(local[0] + loc.x),
            float(local[1] + loc.y),
            float(local[2] + loc.z),
        )
    finally:
        scene.frame_set(original_frame)


def create_release_point(joints: np.ndarray, pitcher_obj, handedness: str):
    release_frame = _find_phase_frame(joints, handedness, "release")
    if release_frame is None:
        print("  No release frame detected, skipping release point")
        return None

    marker_specs = _resolve_marker_indices(handedness)
    wrist_idx = marker_specs["throw_wrist"][0]
    world_pos = _world_pos_at_frame(joints, release_frame, wrist_idx, pitcher_obj)

    sphere = _make_sphere("ReleasePoint", 0.08, (1.0, 0.0, 0.0, 1.0))
    sphere.location = world_pos

    # Label hovering above the sphere
    bpy.ops.object.text_add(location=(world_pos[0], world_pos[1], world_pos[2] + 0.15))
    label = bpy.context.active_object
    label.name = "ReleasePoint_Label"
    label.data.body = f"Release @ f{release_frame}"
    label.data.size = 0.10
    label.rotation_euler = (math.radians(90), 0, 0)
    label.data.materials.append(_make_colored_material("ReleasePointLabel_Mat", (1, 1, 1, 1)))

    # Extension (distance from rubber to release point, XY only — the
    # Statcast "extension" metric)
    extension = math.sqrt(world_pos[0] ** 2 + world_pos[1] ** 2)
    print(f"  Release point: frame {release_frame}, world {tuple(round(v, 3) for v in world_pos)}")
    print(f"  Extension (rubber to release, XY): {extension:.2f} m ({extension / bfv.FT_TO_M:.1f} ft)")
    return sphere


def create_stride_ruler(joints: np.ndarray, pitcher_obj, handedness: str):
    """Ground-plane bar from pivot ankle (frame 0) to stride ankle at foot
    strike. Length is the stride length in the same frame of reference."""
    foot_strike_frame = _find_phase_frame(joints, handedness, "acceleration")
    if foot_strike_frame is None:
        print("  No foot-strike frame detected, skipping stride ruler")
        return None

    marker_specs = _resolve_marker_indices(handedness)
    pivot_idx = marker_specs["pivot_ankle"][0]
    stride_idx = marker_specs["stride_ankle"][0]

    pivot_world = _world_pos_at_frame(joints, 0, pivot_idx, pitcher_obj)
    stride_world = _world_pos_at_frame(joints, foot_strike_frame, stride_idx, pitcher_obj)

    dx = stride_world[0] - pivot_world[0]
    dy = stride_world[1] - pivot_world[1]
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1e-6:
        return None

    # Thin ground quad between pivot and stride
    half_w = 0.03
    px = -dy / length * half_w
    py = +dx / length * half_w
    z = 0.012

    verts = [
        (pivot_world[0] - px, pivot_world[1] - py, z),
        (pivot_world[0] + px, pivot_world[1] + py, z),
        (stride_world[0] + px, stride_world[1] + py, z),
        (stride_world[0] - px, stride_world[1] - py, z),
    ]
    mesh = bpy.data.meshes.new("StrideRuler")
    mesh.from_pydata(verts, [], [(0, 1, 2, 3)])
    obj = bpy.data.objects.new("StrideRuler", mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(_make_colored_material("StrideRuler_Mat", (1.0, 0.2, 1.0, 1.0), emission=2.0))

    # Label at the midpoint, hovering above the bar
    midpoint = (
        (pivot_world[0] + stride_world[0]) / 2,
        (pivot_world[1] + stride_world[1]) / 2,
        0.20,
    )
    bpy.ops.object.text_add(location=midpoint)
    label = bpy.context.active_object
    label.name = "StrideRuler_Label"
    label.data.body = f"Stride: {length:.2f}m ({length / bfv.FT_TO_M:.1f}ft)"
    label.data.size = 0.12
    label.rotation_euler = (math.radians(90), 0, 0)
    label.data.materials.append(_make_colored_material("StrideRulerLabel_Mat", (1.0, 0.9, 0.3, 1.0)))

    print(f"  Stride length: {length:.2f} m ({length / bfv.FT_TO_M:.1f} ft) at frame {foot_strike_frame}")
    return obj


# ─── LIVE ELBOW ANGLE TEXT (Blender-only) ────────────────────────────────────


def create_angle_text(joint_markers: dict):
    """Elbow angle readout. Updated per frame by a frame_change_post handler
    — text bodies can't be driven directly in Blender, so a handler reads
    world positions from the joint marker spheres and writes to .data.body.

    Does NOT export to GLB (no text animation in glTF). Browser should
    recompute this in JS from the baked joint animation.
    """
    shoulder = joint_markers.get("throw_shoulder")
    elbow = joint_markers.get("throw_elbow")
    wrist = joint_markers.get("throw_wrist")
    if not (shoulder and elbow and wrist):
        print("  Missing throw-arm markers, skipping angle text")
        return None

    bpy.ops.object.text_add(location=(0.0, 2.5, 3.2))
    text_obj = bpy.context.active_object
    text_obj.name = "ElbowAngleText"
    text_obj.data.size = 0.28
    text_obj.rotation_euler = (math.radians(90), 0, 0)
    text_obj.data.body = "Elbow: -- deg"
    text_obj.data.materials.append(_make_colored_material("ElbowAngleText_Mat", (1, 1, 0.1, 1)))

    # Stash marker names so the handler can find them
    text_obj["shoulder_marker"] = shoulder.name
    text_obj["elbow_marker"] = elbow.name
    text_obj["wrist_marker"] = wrist.name

    return text_obj


def _elbow_angle_handler(scene):
    text_obj = bpy.data.objects.get("ElbowAngleText")
    if text_obj is None or text_obj.type != 'FONT':
        return
    s_name = text_obj.get("shoulder_marker")
    e_name = text_obj.get("elbow_marker")
    w_name = text_obj.get("wrist_marker")
    if not (s_name and e_name and w_name):
        return
    s = bpy.data.objects.get(s_name)
    e = bpy.data.objects.get(e_name)
    w = bpy.data.objects.get(w_name)
    if not (s and e and w):
        return

    s_pos = s.matrix_world.translation
    e_pos = e.matrix_world.translation
    w_pos = w.matrix_world.translation

    v1 = s_pos - e_pos
    v2 = w_pos - e_pos
    if v1.length < 1e-6 or v2.length < 1e-6:
        return
    cos_angle = max(-1.0, min(1.0, v1.normalized().dot(v2.normalized())))
    angle_deg = math.degrees(math.acos(cos_angle))
    text_obj.data.body = f"Elbow: {angle_deg:3.0f} deg"


def _install_elbow_angle_handler():
    handlers = bpy.app.handlers.frame_change_post
    # Idempotent: remove any previous copies by name
    handlers[:] = [h for h in handlers if getattr(h, "__name__", "") != "_elbow_angle_handler"]
    handlers.append(_elbow_angle_handler)


# ─── TIMELINE PHASE MARKERS ──────────────────────────────────────────────────


def add_phase_markers(joints: np.ndarray, handedness: str):
    phases = bfv._detect_pitch_phases(joints, handedness=handedness)
    scene = bpy.context.scene
    # Clear any previous phase markers by known labels
    for name in ("set", "leg lift", "stride", "acceleration", "release", "follow-through"):
        m = scene.timeline_markers.get(name)
        if m is not None:
            scene.timeline_markers.remove(m)
    for frame, label in phases.items():
        scene.timeline_markers.new(name=label, frame=int(frame))
    print(f"  Added {len(phases)} timeline markers")


# ─── MAIN ────────────────────────────────────────────────────────────────────


def main():
    print("\n" + "=" * 60)
    print("  Analysis Overlays")
    print("=" * 60)

    pitcher = find_pitcher_mesh()
    if pitcher is None:
        return

    joints, handedness = load_joint_data(pitcher)
    if joints is None:
        return
    print(f"Loaded joints: {joints.shape} ({handedness}-handed)")

    print("Creating joint markers...")
    markers = create_joint_markers(joints, pitcher, handedness)

    print("Baking motion paths (frame-stepping to capture foot-lock)...")
    create_motion_paths(joints, pitcher, handedness)

    print("Creating strike zone box...")
    create_strike_zone()

    print("Detecting release frame and placing release point...")
    create_release_point(joints, pitcher, handedness)

    print("Measuring stride and placing stride ruler...")
    create_stride_ruler(joints, pitcher, handedness)

    print("Creating live elbow angle text (Blender only)...")
    create_angle_text(markers)
    _install_elbow_angle_handler()

    print("Adding pitch-phase markers to timeline...")
    add_phase_markers(joints, handedness)

    print("-" * 60)
    print("  Overlays created. Scrub the timeline to see live angle text,")
    print("  joint markers tracking the body, and baked world motion paths.")
    print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
