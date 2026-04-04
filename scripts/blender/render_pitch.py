"""
Render a single pitch GLB as an animation.

Usage (inside Blender):
    blender --background --python render_pitch.py -- --glb PATH --output DIR [options]
"""
import sys
import os

try:
    import bpy
    HAS_BPY = True
except ImportError:
    HAS_BPY = False


def main():
    import argparse

    argv = sys.argv
    script_args = argv[argv.index("--") + 1:] if "--" in argv else []

    parser = argparse.ArgumentParser(description="Render a single pitch GLB")
    parser.add_argument("--glb", required=True, help="Path to input .glb file")
    parser.add_argument("--output", required=True, help="Output directory for frame images")
    parser.add_argument("--preset", default="catcher", help="Camera preset name")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=-1)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args(script_args)

    if not HAS_BPY:
        print("ERROR: bpy not available. Run inside Blender:")
        print("  blender --background --python render_pitch.py -- --glb PATH --output DIR")
        sys.exit(1)

    # Import scene helpers from same directory
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scene_setup import setup_world, create_mound_mesh, setup_materials, setup_lighting, setup_render_settings
    from camera_presets import set_camera_preset

    os.makedirs(args.output, exist_ok=True)

    # Fresh scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Setup
    setup_world()
    setup_materials()
    create_mound_mesh()
    setup_lighting()
    setup_render_settings(samples=64)

    # Import pitch GLB
    bpy.ops.import_scene.gltf(filepath=args.glb)

    # Camera
    set_camera_preset(args.preset)

    # Frame range
    scene = bpy.context.scene
    scene.render.fps = args.fps
    scene.frame_start = args.start_frame
    if args.end_frame >= 0:
        scene.frame_end = args.end_frame

    # Output path (Blender appends frame number and extension)
    scene.render.filepath = os.path.join(args.output, "frame_")
    scene.render.image_settings.file_format = "PNG"

    print(f"Rendering frames {scene.frame_start}-{scene.frame_end} to {args.output}")
    bpy.ops.render.render(animation=True)
    print("Render complete:", args.output)


if __name__ == "__main__":
    main()
