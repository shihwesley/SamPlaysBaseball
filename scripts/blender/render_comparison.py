"""
Render a pitch comparison in ghost or side-by-side mode.

Usage (inside Blender):
    blender --background --python render_comparison.py -- --glb1 PATH --glb2 PATH --output DIR [options]
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

    parser = argparse.ArgumentParser(description="Render pitch comparison")
    parser.add_argument("--glb1", required=True, help="First pitch GLB path")
    parser.add_argument("--glb2", required=True, help="Second pitch GLB path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--mode", choices=["ghost", "sidebyside"], default="ghost")
    parser.add_argument("--preset", default="catcher", help="Camera preset")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args(script_args)

    if not HAS_BPY:
        print("ERROR: bpy not available. Run inside Blender:")
        print("  blender --background --python render_comparison.py -- --glb1 P1 --glb2 P2 --output DIR")
        sys.exit(1)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scene_setup import setup_world, create_mound_mesh, setup_materials, setup_lighting, setup_render_settings
    from camera_presets import set_camera_preset

    os.makedirs(args.output, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    setup_world()
    setup_materials()
    create_mound_mesh()
    setup_lighting()
    setup_render_settings(samples=64)

    # Import both pitches
    bpy.ops.import_scene.gltf(filepath=args.glb1)
    bpy.ops.import_scene.gltf(filepath=args.glb2)

    meshes = [o for o in bpy.data.objects if o.type == "MESH" and o.name != "Mound"]

    if args.mode == "ghost":
        # Apply ghost material to second mesh
        ghost_mat = bpy.data.materials.get("GhostMat")
        if len(meshes) >= 2 and ghost_mat is not None:
            second = meshes[-1]
            second.data.materials.clear()
            second.data.materials.append(ghost_mat)

        set_camera_preset(args.preset)
        bpy.context.scene.render.filepath = os.path.join(args.output, "ghost_")
        bpy.context.scene.render.image_settings.file_format = "PNG"
        bpy.ops.render.render(animation=True)

    elif args.mode == "sidebyside":
        # Offset second mesh to the right, render together
        if len(meshes) >= 2:
            meshes[-1].location.x += 3.0

        set_camera_preset("first_base")  # side view shows both
        bpy.context.scene.render.filepath = os.path.join(args.output, "sidebyside_")
        bpy.context.scene.render.image_settings.file_format = "PNG"
        bpy.ops.render.render(animation=True)

    print("Comparison render complete:", args.output)


if __name__ == "__main__":
    main()
