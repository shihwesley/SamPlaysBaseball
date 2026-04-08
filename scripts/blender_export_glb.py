"""
blender_export_glb.py — export the whole scene to a single .glb for Three.js.

Run AFTER blender_field_viewer.py (and optionally blender_analysis_overlays.py).
Writes one GLB containing: field geometry, pitcher mesh with shape-key
animation, foot-lock object-level animation, joint markers, motion paths,
strike zone, release point, stride ruler, all 7 camera presets, lights.

Usage:
  - Open in Blender Text Editor → Alt+P
  - Or headless: blender --background --python scripts/blender_export_glb.py

Output path:
  1. SAMPLAYS_GLB_OUT env var, if set
  2. ../frontend/public/models/pitch.glb (relative to this script)
  3. ./data/export/pitch.glb as a last-resort fallback
"""

import bpy
import os
import sys


def _find_script_dir() -> str | None:
    """Locate the SamPlaysBaseball scripts directory.

    Blender's Text Editor sets __file__ to something like
    '/blender_export_glb.py' in 5.1 (bare filename with a leading slash),
    so os.path.dirname(__file__) returns '/' — which then tries to write
    output to '/data/export/pitch.glb' on the read-only system root. This
    function walks bpy.data.texts and known fallback paths to find the
    real directory."""
    try:
        f = os.path.abspath(__file__)  # type: ignore[name-defined]
        if os.path.isfile(f):
            return os.path.dirname(f)
    except Exception:
        pass

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

    env_dir = os.environ.get("SAMPLAYS_SCRIPTS_DIR", "").strip()
    if env_dir and os.path.isdir(env_dir):
        return env_dir

    candidates = [
        os.path.expanduser("~/Source/SamPlaysBaseball/scripts"),
    ]
    for c in candidates:
        if os.path.isfile(os.path.join(c, "blender_field_viewer.py")):
            return c

    return None


def _resolve_output_path() -> str:
    """Pick an output path that is actually writable.

    Priority:
      1. SAMPLAYS_GLB_OUT env var (explicit override)
      2. <project>/frontend/public/models/pitch.glb if the frontend dir exists
      3. <project>/data/export/pitch.glb if the project dir is writable
      4. ~/Desktop/pitch.glb as a universally writable fallback
    """
    env_path = os.environ.get("SAMPLAYS_GLB_OUT", "").strip()
    if env_path:
        return os.path.abspath(os.path.expanduser(env_path))

    script_dir = _find_script_dir()
    if script_dir:
        project_root = os.path.dirname(script_dir)

        frontend_public = os.path.join(project_root, "frontend", "public")
        if os.path.isdir(frontend_public):
            return os.path.join(frontend_public, "models", "pitch.glb")

        data_dir = os.path.join(project_root, "data")
        if os.path.isdir(data_dir):
            return os.path.join(data_dir, "export", "pitch.glb")

    # Last-resort: user's Desktop. Always writable, always findable.
    desktop = os.path.expanduser("~/Desktop")
    if os.path.isdir(desktop):
        return os.path.join(desktop, "pitch.glb")

    return os.path.expanduser("~/pitch.glb")


def _ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)


def _trim_to_pitch(pad_frames: int = 30):
    """Trim the scene frame range to just the pitch delivery, bounded by
    the timeline phase markers. 370-frame clips contain 4+ seconds of
    follow-through/walk-off that balloons the exported morph target data
    with zero analytical value.

    Uses the 'set' → 'follow-through' phase markers (or 0 → release+pad
    if those markers are missing). Set SAMPLAYS_KEEP_FULL_RANGE=1 to skip
    trimming and export every frame.
    """
    if os.environ.get("SAMPLAYS_KEEP_FULL_RANGE", "").strip() == "1":
        print("  SAMPLAYS_KEEP_FULL_RANGE=1 — keeping full frame range")
        return

    scene = bpy.context.scene
    markers = scene.timeline_markers
    if not markers:
        print("  No timeline markers, keeping full frame range")
        return

    frames_by_label = {m.name: m.frame for m in markers}
    start = frames_by_label.get("set", scene.frame_start)
    end_candidates = [
        frames_by_label.get("follow-through"),
        frames_by_label.get("release"),
        frames_by_label.get("acceleration"),
    ]
    end_candidates = [f for f in end_candidates if f is not None]
    if not end_candidates:
        print("  No end-phase marker found, keeping full frame range")
        return

    end = max(end_candidates) + pad_frames
    end = min(end, scene.frame_end)
    start = max(start, scene.frame_start)

    old_range = (scene.frame_start, scene.frame_end)
    scene.frame_start = int(start)
    scene.frame_end = int(end)
    print(f"  Trimmed frame range: {old_range[0]}..{old_range[1]} → {scene.frame_start}..{scene.frame_end}")


def _scene_summary():
    """Print what's in the scene so you can spot missing pieces before export."""
    scene = bpy.context.scene
    meshes = [o for o in scene.objects if o.type == 'MESH']
    cameras = [o for o in scene.objects if o.type == 'CAMERA']
    lights = [o for o in scene.objects if o.type == 'LIGHT']
    texts = [o for o in scene.objects if o.type == 'FONT']

    print(f"  Meshes:  {len(meshes)}")
    print(f"  Cameras: {len(cameras)}  ({', '.join(c.name for c in cameras)})")
    print(f"  Lights:  {len(lights)}")
    print(f"  Texts:   {len(texts)}  (glTF will NOT export font geometry — convert to mesh first if you want text in the GLB)")

    pitcher = bpy.data.objects.get("PitcherMesh")
    if pitcher is None:
        print("  WARNING: no PitcherMesh — field viewer must run first.")
    else:
        shape_key_count = 0
        if pitcher.type == 'MESH' and pitcher.data.shape_keys:
            shape_key_count = len(pitcher.data.shape_keys.key_blocks) - 1
        print(f"  PitcherMesh: shape keys = {shape_key_count}, npz = {pitcher.get('npz_path', '<unset>')}")


def export_glb(output_path: str):
    """Run the glTF 2.0 exporter with settings tuned for a Three.js frontend.

    Key flags:
      export_format='GLB'        — single binary file, no sidecar textures
      export_animations=True     — shape keys + object-level animation
      export_morph=True          — required for shape-key playback
      export_cameras=True        — all 7 presets come through as
                                    PerspectiveCamera nodes in the scene graph
      export_lights=True         — sun + fill (often overridden in-browser)
      export_apply=True          — apply modifiers before export (safe — the
                                    scripts don't use modifiers, but cheap
                                    insurance against future edits)
      export_draco_mesh_compression_enable=True  — ~5× smaller meshes;
                                                    Three.js needs DRACOLoader
                                                    on the browser side
    """
    print("\n" + "=" * 60)
    print("  GLB Export (Three.js-ready)")
    print("=" * 60)
    _scene_summary()
    _trim_to_pitch()

    _ensure_parent_dir(output_path)

    # Deselect everything — we export the whole scene, not a selection.
    for obj in bpy.context.scene.objects:
        obj.select_set(False)

    print(f"\nExporting to: {output_path}")
    bpy.ops.export_scene.gltf(
        filepath=output_path,
        export_format='GLB',
        use_selection=False,
        export_apply=True,
        export_animations=True,
        export_morph=True,
        # CRITICAL: morph normals double the per-target data. The pitcher
        # mesh has ~110k expanded verts × 369 morph targets, so enabling
        # normals turns a ~450 MB file into a ~900 MB file. Three.js
        # recomputes normals on the fly; the quality hit is negligible
        # on diffuse skin shading.
        export_morph_normal=False,
        export_morph_tangent=False,
        export_frame_range=True,
        export_cameras=True,
        export_lights=True,
        export_yup=True,
        export_draco_mesh_compression_enable=True,
        export_draco_mesh_compression_level=10,   # max compression
        # Aggressive Draco quantization — lower precision means smaller
        # files with imperceptible quality loss for a body mesh animation.
        export_draco_position_quantization=14,    # default 14, range 1-30
        export_draco_normal_quantization=10,      # default 10
        export_draco_texcoord_quantization=12,    # default 12
        export_draco_generic_quantization=12,
    )

    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\n  OK — wrote {size_mb:.2f} MB")
        print(f"  Load in Three.js with GLTFLoader + DRACOLoader.")
    else:
        print("\n  ERROR — export reported success but file missing.")


def main():
    output = _resolve_output_path()
    export_glb(output)


if __name__ == "__main__":
    main()
