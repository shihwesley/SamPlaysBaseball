"""Open a benchmark GLB in Blender with proper material and smooth shading.

Usage:
    /Applications/Blender.app/Contents/MacOS/Blender --python scripts/blender_view_benchmark.py -- 60fps
"""

import bpy
import sys
import os

# Parse arg after `--`
strategy = "60fps"
if "--" in sys.argv:
    idx = sys.argv.index("--")
    if idx + 1 < len(sys.argv):
        strategy = sys.argv[idx + 1]

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
glb_path = os.path.join(project_root, "data", "benchmark_fps", f"{strategy}.glb")

if not os.path.exists(glb_path):
    print(f"ERROR: {glb_path} not found")
    sys.exit(1)

print(f"Loading {glb_path}")

# Clear default scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
for block in list(bpy.data.meshes):
    bpy.data.meshes.remove(block)
for block in list(bpy.data.materials):
    bpy.data.materials.remove(block)

# Set scene FPS to 30
bpy.context.scene.render.fps = 30

# Import the GLB
bpy.ops.import_scene.gltf(filepath=glb_path)

# Find all meshes (GLB import may nest them under empties)
meshes = [obj for obj in bpy.data.objects if obj.type == 'MESH']
print(f"Imported {len(meshes)} mesh objects")

for obj in meshes:
    print(f"  {obj.name}: {len(obj.data.vertices)} verts, "
          f"{len(obj.data.materials)} materials")

    # Smooth shading
    for poly in obj.data.polygons:
        poly.use_smooth = True

    # Force a light gray material if none assigned (or override)
    mat = bpy.data.materials.new(name="PitcherMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.65, 0.74, 0.86, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.7
    bsdf.inputs["Metallic"].default_value = 0.0

    # Disable backface culling so we can see through the mesh from any angle
    mat.use_backface_culling = False
    mat.show_transparent_back = True

    obj.data.materials.clear()
    obj.data.materials.append(mat)

    # Recalculate normals to ensure they all point outward
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()

    # Set frame range from shape key animation
    if obj.data.shape_keys and obj.data.shape_keys.animation_data:
        action = obj.data.shape_keys.animation_data.action
        if action:
            start, end = action.frame_range
            bpy.context.scene.frame_start = int(start)
            bpy.context.scene.frame_end = int(end)
            print(f"  Animation range: {int(start)}-{int(end)}")

# Add a sun light
bpy.ops.object.light_add(type='SUN', location=(0, 0, 5))
sun = bpy.context.active_object
sun.data.energy = 3.0
sun.rotation_euler = (0.5, 0.3, 0)

# Set viewport shading and frame view
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                space.shading.type = 'MATERIAL'  # Material Preview
                space.shading.use_scene_lights = True
                space.shading.use_scene_world = True
        for region in area.regions:
            if region.type == 'WINDOW':
                with bpy.context.temp_override(area=area, region=region):
                    bpy.ops.view3d.view_all()
                break

# Reset to first frame
bpy.context.scene.frame_set(bpy.context.scene.frame_start)

# Save as .blend file so we can reopen it
blend_path = os.path.join(project_root, "data", "benchmark_fps", f"{strategy}.blend")
bpy.ops.wm.save_as_mainfile(filepath=blend_path)

print("\n✓ Setup complete. Press Spacebar to play animation.")
print(f"✓ Strategy: {strategy}")
print(f"✓ FPS: {bpy.context.scene.render.fps}")
print(f"✓ Frame range: {bpy.context.scene.frame_start}-{bpy.context.scene.frame_end}")
print(f"✓ Saved: {blend_path}")
