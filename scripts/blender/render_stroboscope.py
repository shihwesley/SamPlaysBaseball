"""
Render a stroboscope overlay of multiple pitches at a single phase frame.

Usage (inside Blender):
    blender --background --python render_stroboscope.py -- --glbs P1 P2 P3 --output DIR [options]
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

    parser = argparse.ArgumentParser(description="Render multi-pitch stroboscope overlay")
    parser.add_argument("--glbs", nargs="+", required=True, help="List of .glb file paths")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--phase-frame", type=int, default=0, help="Morph target frame index to display")
    parser.add_argument("--opacity", type=float, default=0.12, help="Per-mesh opacity")
    parser.add_argument("--preset", default="catcher", help="Camera preset")
    args = parser.parse_args(script_args)

    if not HAS_BPY:
        print("ERROR: bpy not available. Run inside Blender:")
        print("  blender --background --python render_stroboscope.py -- --glbs P1 P2 --output DIR")
        sys.exit(1)

    n = len(args.glbs)
    if n == 0:
        print("ERROR: No GLB files provided.")
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
    setup_render_settings(resolution=(1920, 1080), samples=32)

    # Import all GLBs
    for glb_path in args.glbs:
        if not os.path.exists(glb_path):
            print(f"WARNING: GLB not found: {glb_path}")
            continue
        bpy.ops.import_scene.gltf(filepath=glb_path)

    # Apply stroboscope material to all non-mound meshes and set phase frame
    for i, obj in enumerate([o for o in bpy.data.objects if o.type == "MESH" and o.name != "Mound"]):
        # Create unique stroboscope material per mesh
        mat_name = f"StrobeMat_{i}"
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        mat.blend_method = "BLEND"
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            # Slight hue variation across pitches
            hue_shift = (i / max(n, 1)) * 0.3
            bsdf.inputs["Base Color"].default_value = (0.5 + hue_shift, 0.7, 1.0, 1.0)
            bsdf.inputs["Alpha"].default_value = args.opacity
        obj.data.materials.clear()
        obj.data.materials.append(mat)

        # Set morph target to phase_frame
        if obj.data.shape_keys and len(obj.data.shape_keys.key_blocks) > 1:
            key_blocks = obj.data.shape_keys.key_blocks
            for j, kb in enumerate(key_blocks[1:]):  # skip basis (index 0)
                kb.value = 1.0 if j == args.phase_frame else 0.0

    set_camera_preset(args.preset)

    # Single frame render
    bpy.context.scene.frame_set(1)
    out_path = os.path.join(args.output, "stroboscope.png")
    bpy.context.scene.render.filepath = out_path
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(write_still=True)
    print("Stroboscope render complete:", out_path)


if __name__ == "__main__":
    main()
