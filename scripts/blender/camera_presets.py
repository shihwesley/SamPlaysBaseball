"""Camera preset positions and animation for pitch renders."""
import sys
import math

try:
    import bpy
    import mathutils
    HAS_BPY = True
except ImportError:
    HAS_BPY = False


PRESETS = {
    "catcher":        {"location": (0.0, -8.0, 1.5)},
    "first_base":     {"location": (8.0, -4.0, 1.5)},
    "third_base":     {"location": (-8.0, -4.0, 1.5)},
    "overhead":       {"location": (0.0, 0.0, 10.0)},
    "behind_pitcher": {"location": (0.0, 6.0, 1.5)},
}


def point_camera_at(camera_obj, target=(0.0, 0.0, 0.0)):
    """Add a Track To constraint so camera faces the target point."""
    if not HAS_BPY:
        return
    # Remove existing track constraints
    for c in list(camera_obj.constraints):
        if c.type == "TRACK_TO":
            camera_obj.constraints.remove(c)

    # Create empty at target
    target_name = "CameraTarget"
    if target_name in bpy.data.objects:
        target_obj = bpy.data.objects[target_name]
        target_obj.location = mathutils.Vector(target)
    else:
        bpy.ops.object.empty_add(location=target)
        target_obj = bpy.context.active_object
        target_obj.name = target_name

    constraint = camera_obj.constraints.new(type="TRACK_TO")
    constraint.target = target_obj
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"


def set_camera_preset(preset: str):
    """Position camera for the named preset and point it at origin."""
    if not HAS_BPY:
        return
    preset_key = preset.replace("-", "_")
    if preset_key not in PRESETS:
        preset_key = "catcher"

    # Get or create main camera
    camera = bpy.data.objects.get("Camera")
    if camera is None:
        bpy.ops.object.camera_add()
        camera = bpy.context.active_object
        camera.name = "Camera"
    bpy.context.scene.camera = camera

    loc = PRESETS[preset_key]["location"]
    camera.location = mathutils.Vector(loc)
    point_camera_at(camera, target=(0.0, 0.5, 0.0))


def animate_camera_orbit(start_frame: int, end_frame: int, radius: float = 8.0, height: float = 2.0):
    """Keyframe a circular camera orbit around origin over the given frame range."""
    if not HAS_BPY:
        return
    camera = bpy.data.objects.get("Camera")
    if camera is None:
        bpy.ops.object.camera_add()
        camera = bpy.context.active_object
        camera.name = "Camera"
    bpy.context.scene.camera = camera

    n_frames = max(1, end_frame - start_frame)
    for i, frame in enumerate(range(start_frame, end_frame + 1)):
        angle = (2 * math.pi * i) / n_frames
        x = radius * math.sin(angle)
        y = -radius * math.cos(angle)
        z = height
        camera.location = mathutils.Vector((x, y, z))
        camera.keyframe_insert(data_path="location", frame=frame)
        point_camera_at(camera, target=(0.0, 0.5, 0.0))
