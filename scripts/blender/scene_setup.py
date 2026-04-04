"""Blender scene setup: mound, materials, and lighting for pitch renders."""
import sys
import math

try:
    import bpy
    HAS_BPY = True
except ImportError:
    HAS_BPY = False


def setup_world(bg_color=(0.02, 0.02, 0.02, 1.0)):
    """Set world background color."""
    if not HAS_BPY:
        return
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs["Color"].default_value = bg_color
        bg_node.inputs["Strength"].default_value = 0.5


def create_mound_mesh():
    """Create MLB pitcher's mound geometry (18ft dia, 10.5in rise)."""
    if not HAS_BPY:
        return None
    bpy.ops.mesh.primitive_cylinder_add(
        radius=2.74,   # ~9ft radius (18ft diameter)
        depth=0.267,   # 10.5 inches
        vertices=32,
        location=(0, 0, 0),
    )
    obj = bpy.context.active_object
    obj.name = "Mound"
    mat = bpy.data.materials.get("MoundMat")
    if mat:
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)
    return obj


def setup_materials():
    """Create materials for pitcher, ghost, and mound."""
    if not HAS_BPY:
        return

    # Pitcher skin material
    if "PitcherMat" not in bpy.data.materials:
        mat = bpy.data.materials.new("PitcherMat")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (0.91, 0.72, 0.54, 1.0)  # #e8b88a
            bsdf.inputs["Roughness"].default_value = 0.8

    # Ghost/transparent comparison material
    if "GhostMat" not in bpy.data.materials:
        mat = bpy.data.materials.new("GhostMat")
        mat.use_nodes = True
        mat.blend_method = "BLEND"
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (0.27, 0.53, 1.0, 1.0)  # #4488ff
            bsdf.inputs["Alpha"].default_value = 0.3
            bsdf.inputs["Roughness"].default_value = 0.5

    # Mound dirt material
    if "MoundMat" not in bpy.data.materials:
        mat = bpy.data.materials.new("MoundMat")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (0.55, 0.41, 0.08, 1.0)  # #8B6914
            bsdf.inputs["Roughness"].default_value = 0.95


def setup_lighting():
    """Add key, fill, and rim lights for broadcast-style renders."""
    if not HAS_BPY:
        return

    # Clear existing lights
    for obj in bpy.data.objects:
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)

    # Key light: sun from front-above at 45 degrees
    bpy.ops.object.light_add(type="SUN", location=(5, -5, 8))
    key = bpy.context.active_object
    key.name = "KeyLight"
    key.data.energy = 3.0
    key.rotation_euler = (math.radians(45), 0, math.radians(45))

    # Fill light: area from the left side
    bpy.ops.object.light_add(type="AREA", location=(-8, 0, 4))
    fill = bpy.context.active_object
    fill.name = "FillLight"
    fill.data.energy = 1.0
    fill.data.size = 5.0

    # Rim light: spot from behind
    bpy.ops.object.light_add(type="SPOT", location=(0, 8, 3))
    rim = bpy.context.active_object
    rim.name = "RimLight"
    rim.data.energy = 2.0
    rim.data.spot_size = math.radians(60)
    rim.rotation_euler = (math.radians(-30), 0, math.radians(180))


def setup_render_settings(resolution=(1920, 1080), samples=64, engine="CYCLES"):
    """Configure render engine, resolution, and sample count."""
    if not HAS_BPY:
        return
    scene = bpy.context.scene
    scene.render.engine = engine
    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]
    scene.render.resolution_percentage = 100

    if engine == "CYCLES":
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
    elif engine == "BLENDER_EEVEE_NEXT":
        scene.eevee.taa_render_samples = samples
