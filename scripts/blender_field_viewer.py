"""
Blender Python script: MLB field scene with SAM 3D Body mesh.

Usage:
  1. Open Blender
  2. Edit > Preferences > Add-ons > enable "Import-Export: glTF 2.0" (usually on by default)
  3. Open this script in Blender's text editor (or Scripting workspace)
  4. Set MESH_PATH below to your .glb or .obj file
  5. Run the script (Alt+P)

Or from command line:
  blender --python scripts/blender_field_viewer.py

The script builds an MLB-regulation mound + home plate area,
imports the pitcher mesh, and sets up camera presets for analysis.

Camera presets are stored as named cameras — switch via View > Cameras
or the camera dropdown in the viewport header.
"""

import bpy
import bmesh
import math
import os
import numpy as np
from mathutils import Vector, Matrix

# ─── CONFIG ──────────────────────────────────────────────────────────────────

# Path to the mesh file (.glb preferred for animation, .obj for single frame)
# Can also be injected at launch time via the SAMPLAYS_MESH_PATH env var
# (used by the backend "Open in Blender" button).
MESH_PATH = os.environ.get("SAMPLAYS_MESH_PATH", "")

# Player height in meters (for scale verification — the pipeline already scales)
PLAYER_HEIGHT_M = 1.88  # default ~6'2", adjust per pitcher

# Default playback FPS, used when the .npz doesn't carry a source_fps array.
# Baseball Savant clips are ~59.94 fps; older .npz files written before the
# fps-capture change land on this default.
SOURCE_FPS = 60

# Pitcher handedness — used by foot-lock and overlay scripts when the .npz
# doesn't carry a 'handedness' field. Override per-clip via the
# SAMPLAYS_HANDEDNESS env var or by editing this constant.
#   "right" → pivot foot = right ankle (mhr70[14]), stride = left
#   "left"  → pivot foot = left ankle  (mhr70[13]), stride = right
DEFAULT_HANDEDNESS = os.environ.get("SAMPLAYS_HANDEDNESS", "right").lower().strip()

# ─── MLB REGULATION DIMENSIONS (meters) ──────────────────────────────────────

FT_TO_M = 0.3048
IN_TO_M = 0.0254

MOUND_DIAMETER_FT = 18.0
MOUND_RADIUS_M = (MOUND_DIAMETER_FT / 2) * FT_TO_M   # 2.7432m
MOUND_RISE_M = 10 * IN_TO_M                             # 0.254m (MLB Rule 2.01: exactly 10")
RUBBER_LENGTH_M = 24 * IN_TO_M                          # 0.6096m
RUBBER_WIDTH_M = 6 * IN_TO_M                            # 0.1524m
MOUND_TO_PLATE_M = 60.5 * FT_TO_M                      # 18.4404m (60'6")

# ── MLB mound profile (Rule 2.01) ──
# Flat top is a 5 ft diameter circle centered on the mound origin.
# Front slope drops 1" per 1 ft for 6 ft, starting 6" in front of the rubber's
# front edge. The strip is ~34" wide (half-width 17").
MOUND_FLAT_RADIUS_M = 2.5 * FT_TO_M                    # 0.762m (5ft dia flat top)
MOUND_SLOPE_DROP_M = 6 * IN_TO_M                        # 0.152m total drop over 6ft
MOUND_SLOPE_LEN_M = 6 * FT_TO_M                         # 1.829m slope length
MOUND_SLOPE_HALF_W_M = 17 * IN_TO_M                     # 0.432m (half of 34" strip)

# Home plate is a pentagon: 17" square with two 12" angled sides
PLATE_SIDE_M = 17 * IN_TO_M       # 0.4318m
PLATE_ANGLE_M = 12 * IN_TO_M      # 0.3048m

# Dirt circle around mound
DIRT_RADIUS_M = 9 * FT_TO_M       # half of 18ft

# Infield dirt (approximate)
BASELINE_M = 90 * FT_TO_M         # 27.432m

# MLB regulation: bases are 18" square (post-2023 rule change, was 15")
BASE_SIZE_M = 18 * IN_TO_M         # 0.4572m
BASE_THICKNESS_M = 0.05            # 5cm — bases are puffy

# Batter's box: 4ft x 6ft, 6" from the side of home plate
BATTERS_BOX_W_M = 4 * FT_TO_M      # 1.219m
BATTERS_BOX_L_M = 6 * FT_TO_M      # 1.829m
BATTERS_BOX_GAP_M = 6 * IN_TO_M    # 0.152m

# Catcher's box: 43" wide, 8 ft long, behind home plate
CATCHERS_BOX_W_M = 43 * IN_TO_M    # 1.092m
CATCHERS_BOX_L_M = 8 * FT_TO_M     # 2.438m

# Infield dirt arc radius from home plate (approximate, varies by stadium)
INFIELD_ARC_RADIUS_M = 95 * FT_TO_M  # 28.96m

# Chalk line strip half-width (real lines are ~3" wide)
CHALK_HALF_W_M = 0.04

# ─── HELPERS ─────────────────────────────────────────────────────────────────


def clear_scene():
    """Remove all objects, meshes, cameras, lights from the default scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Clean orphaned data
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.cameras:
        if block.users == 0:
            bpy.data.cameras.remove(block)
    for block in bpy.data.lights:
        if block.users == 0:
            bpy.data.lights.remove(block)


def _mound_height(x: float, y: float) -> float:
    """MLB regulation mound height at (x, y) in meters.

    Rubber front edge faces home plate (-Y direction). The flat top is a 5ft
    diameter circle at MOUND_RISE_M. Outside the flat top, height tapers
    linearly to the 18ft circle edge. In the front slope zone (a rectangular
    strip in front of the rubber), height drops 1"/1ft for 6ft, overriding
    the radial taper where the slope is lower.

    Outside the 18ft circle: 0.
    """
    r_xy = math.sqrt(x * x + y * y)
    if r_xy >= MOUND_RADIUS_M:
        return 0.0

    # Base height: flat top or linear radial taper
    if r_xy <= MOUND_FLAT_RADIUS_M:
        base_h = MOUND_RISE_M
    else:
        t = (r_xy - MOUND_FLAT_RADIUS_M) / (MOUND_RADIUS_M - MOUND_FLAT_RADIUS_M)
        base_h = MOUND_RISE_M * (1.0 - t)

    # Front slope override (rectangular strip in front of rubber, toward home).
    # Rubber center at origin, rubber front edge at y = -RUBBER_WIDTH_M/2.
    # Slope starts 6" further toward home.
    y_slope_start = -(RUBBER_WIDTH_M / 2 + 6 * IN_TO_M)       # -0.229m
    y_slope_end = y_slope_start - MOUND_SLOPE_LEN_M            # -2.058m

    if y_slope_end < y < y_slope_start and abs(x) < MOUND_SLOPE_HALF_W_M:
        s = (y_slope_start - y) / MOUND_SLOPE_LEN_M            # 0..1
        slope_h = MOUND_RISE_M - s * MOUND_SLOPE_DROP_M
        # Slope dominates where it's lower than the radial taper.
        return min(base_h, slope_h)

    return base_h


def create_mound():
    """
    Build the pitcher's mound as a Cartesian heightfield matching MLB Rule 2.01.

    - 18 ft diameter circle (MOUND_RADIUS_M)
    - 5 ft diameter flat top at MOUND_RISE_M (10")
    - Front slope: 1"/1ft drop for 6ft, starting 6" in front of the rubber,
      in a ~34" wide strip (see _mound_height).
    - Radial taper elsewhere.

    Cartesian grid is used (vs polar) because the front slope is a rectangular
    strip and a polar grid cuts it unevenly. The outer edge is slightly jagged
    but hidden by the dirt circle at Z=0.001.
    """
    mesh = bpy.data.meshes.new("Mound")
    bm = bmesh.new()

    # Resolution: 60x60 grid → ~2800 verts inside the circle (2.4× the old
    # cosine mesh). Enough to resolve the front slope cleanly.
    resolution = 60
    step = (2 * MOUND_RADIUS_M) / resolution

    # Extend slightly past the radius so the boundary clamps cleanly to 0.
    edge_buffer = step * 0.5
    r_cutoff = MOUND_RADIUS_M + edge_buffer

    verts: dict[tuple[int, int], "bmesh.types.BMVert"] = {}
    for i in range(resolution + 1):
        x = -MOUND_RADIUS_M + i * step
        for j in range(resolution + 1):
            y = -MOUND_RADIUS_M + j * step
            r_xy = math.sqrt(x * x + y * y)
            if r_xy > r_cutoff:
                continue
            h = _mound_height(x, y)
            verts[(i, j)] = bm.verts.new((x, y, h))

    # Build quads where all 4 corners exist; skip partial cells (edge).
    for i in range(resolution):
        for j in range(resolution):
            v00 = verts.get((i, j))
            v10 = verts.get((i + 1, j))
            v11 = verts.get((i + 1, j + 1))
            v01 = verts.get((i, j + 1))
            corners = [v for v in (v00, v10, v11, v01) if v is not None]
            if len(corners) == 4:
                bm.faces.new([v00, v10, v11, v01])
            elif len(corners) == 3:
                # Triangle at the circle boundary
                bm.faces.new(corners)

    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new("Mound", mesh)
    bpy.context.collection.objects.link(obj)

    # Smooth shading to hide the grid (still exports fine — shading is
    # a vertex normal hint, not a geometry change).
    for poly in mesh.polygons:
        poly.use_smooth = True

    # Material: brown dirt (solid Principled BSDF — round-trips to Three.js)
    mat = bpy.data.materials.new("MoundDirt")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.45, 0.30, 0.15, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.95
    obj.data.materials.append(mat)

    return obj


def create_rubber():
    """Pitching rubber — centered on mound top, oriented toward home plate (-Y)."""
    bpy.ops.mesh.primitive_cube_add(
        size=1,
        location=(0, 0, MOUND_RISE_M + 0.01),  # slightly above mound surface
        scale=(RUBBER_LENGTH_M / 2, RUBBER_WIDTH_M / 2, 0.02),
    )
    rubber = bpy.context.active_object
    rubber.name = "PitchingRubber"

    mat = bpy.data.materials.new("RubberWhite")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.9, 0.9, 0.9, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.6
    rubber.data.materials.append(mat)

    return rubber


def create_home_plate():
    """
    Home plate pentagon at regulation distance from the rubber.
    Plate is 60'6" from the rubber, measured to the back point.
    """
    # Home plate position: -Y from mound center
    plate_y = -MOUND_TO_PLATE_M

    mesh = bpy.data.meshes.new("HomePlate")
    bm = bmesh.new()

    half = PLATE_SIDE_M / 2
    # Pentagon vertices (MLB spec): square front, pointed back
    # Front edge faces the pitcher (+Y direction from plate)
    pts = [
        (-half, plate_y + PLATE_SIDE_M, 0.005),                     # front-left
        (half, plate_y + PLATE_SIDE_M, 0.005),                      # front-right
        (half, plate_y + PLATE_SIDE_M - PLATE_ANGLE_M, 0.005),      # right side
        (0, plate_y, 0.005),                                         # back point
        (-half, plate_y + PLATE_SIDE_M - PLATE_ANGLE_M, 0.005),     # left side
    ]

    verts = [bm.verts.new(p) for p in pts]
    bm.faces.new(verts)

    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new("HomePlate", mesh)
    bpy.context.collection.objects.link(obj)

    mat = bpy.data.materials.new("PlateWhite")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.95, 0.95, 0.95, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.5
    obj.data.materials.append(mat)

    return obj


def create_ground_plane():
    """Outfield grass plane at Z=0, sized to accommodate the 108m outfield
    wall plus a border. Centered on home plate so the plane extends far
    enough beyond the wall that no camera sees the edge."""
    # 260m square centered at home plate — extends ~130m in every direction.
    size = 260.0
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, -MOUND_TO_PLATE_M + size / 4, 0))
    ground = bpy.context.active_object
    ground.name = "Field"

    mat = _outfield_grass_material("Grass")
    ground.data.materials.append(mat)

    return ground


def create_dirt_circle():
    """Flat dirt circle around the mound base at Z=0.001 (just above grass)."""
    bpy.ops.mesh.primitive_circle_add(
        vertices=64,
        radius=DIRT_RADIUS_M,
        fill_type='NGON',
        location=(0, 0, 0.001),
    )
    dirt = bpy.context.active_object
    dirt.name = "DirtCircle"

    mat = bpy.data.materials.new("InfieldDirt")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.55, 0.38, 0.20, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.95
    dirt.data.materials.append(mat)

    return dirt


def create_foul_lines():
    """Foul lines from home plate to ~100m beyond the infield, drawn as
    chalk strips so they're visible from the catcher cam."""
    line_length = 100
    plate_back_x, plate_back_y = 0, -MOUND_TO_PLATE_M

    for name, angle_deg in [("FoulLine_1B", 45), ("FoulLine_3B", -45)]:
        angle = math.radians(angle_deg)
        end_x = plate_back_x + line_length * math.sin(angle)
        end_y = plate_back_y + line_length * math.cos(angle)
        _add_chalk_strip(name, plate_back_x, plate_back_y, end_x, end_y, z=0.004)


def _white_material(name: str):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.95, 0.95, 0.95, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.6
    return mat


def _dirt_material(name: str):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.55, 0.38, 0.20, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.95
    return mat


def create_bases():
    """Create 1B, 2B, 3B as 18" square cubes at MLB-spec positions.

    Home plate is at (0, -MOUND_TO_PLATE_M, 0). Baselines run at 45° from
    home plate corners. The infield is a square (rotated 45°) with 90 ft
    sides. Bases sit at the corners of that square.
    """
    # Home plate back-point in Blender coords:
    home_back = Vector((0, -MOUND_TO_PLATE_M, 0))

    # Baseline direction vectors (normalized): home → 1B is +X +Y diagonal
    diag = math.sqrt(2) / 2
    to_1b = Vector((diag, diag, 0))
    to_3b = Vector((-diag, diag, 0))
    to_2b_axis = Vector((0, 1, 0))

    # 1B / 3B at 90 ft from home along their respective baselines
    pos_1b = home_back + to_1b * BASELINE_M
    pos_3b = home_back + to_3b * BASELINE_M
    # 2B at 90√2 ft straight forward from home (apex of the diamond)
    pos_2b = home_back + to_2b_axis * (BASELINE_M * math.sqrt(2))

    half = BASE_SIZE_M / 2

    for label, pos in [("Base_1B", pos_1b), ("Base_2B", pos_2b), ("Base_3B", pos_3b)]:
        bpy.ops.mesh.primitive_cube_add(
            size=1,
            location=(pos.x, pos.y, BASE_THICKNESS_M / 2),
            scale=(half, half, BASE_THICKNESS_M / 2),
        )
        base = bpy.context.active_object
        base.name = label
        base.data.materials.append(_white_material(f"{label}_Mat"))


def create_batters_boxes():
    """Two 4ft x 6ft batter's boxes flanking home plate, drawn as
    hollow chalk-line rectangles. Front edge aligns with the front of
    the plate; centered along the plate's depth axis."""
    plate_back_y = -MOUND_TO_PLATE_M
    plate_front_y = plate_back_y + PLATE_SIDE_M

    # Box bounds: width is 4ft, length is 6ft. The MLB spec puts the box
    # so that its front-back midline is offset 6" from the side of the plate
    # and the box is centered front-to-back on the plate's middle (so it
    # extends 3 ft forward and 3 ft back from the plate center).
    plate_center_y = (plate_back_y + plate_front_y) / 2
    box_back_y = plate_center_y - BATTERS_BOX_L_M / 2
    box_front_y = plate_center_y + BATTERS_BOX_L_M / 2
    plate_half_w = PLATE_SIDE_M / 2

    for label, sign in [("BattersBox_R", +1), ("BattersBox_L", -1)]:
        x_inner = sign * (plate_half_w + BATTERS_BOX_GAP_M)
        x_outer = sign * (plate_half_w + BATTERS_BOX_GAP_M + BATTERS_BOX_W_M)
        x_min = min(x_inner, x_outer)
        x_max = max(x_inner, x_outer)

        # Build a hollow rectangle as 4 thin strips at z = 0.005
        for side, (x0, y0, x1, y1) in {
            "front": (x_min, box_front_y, x_max, box_front_y),
            "back":  (x_min, box_back_y,  x_max, box_back_y),
            "inner": (x_inner, box_back_y, x_inner, box_front_y),
            "outer": (x_outer, box_back_y, x_outer, box_front_y),
        }.items():
            _add_chalk_strip(f"{label}_{side}", x0, y0, x1, y1, z=0.006)


def create_catchers_box():
    """Catcher's box: 43" wide, 8 ft long, behind home plate."""
    plate_back_y = -MOUND_TO_PLATE_M
    half_w = CATCHERS_BOX_W_M / 2
    box_far_y = plate_back_y - CATCHERS_BOX_L_M

    edges = {
        "front_l": (-half_w, plate_back_y, -half_w, box_far_y),
        "front_r": (+half_w, plate_back_y, +half_w, box_far_y),
        "back":    (-half_w, box_far_y,    +half_w, box_far_y),
    }
    for side, (x0, y0, x1, y1) in edges.items():
        _add_chalk_strip(f"CatchersBox_{side}", x0, y0, x1, y1, z=0.006)


def create_baselines():
    """Chalk strips along the home→1B and home→3B baselines (foul lines
    are separate; these are the dirt-side baseline chalks)."""
    home_back = Vector((0, -MOUND_TO_PLATE_M, 0))
    diag = math.sqrt(2) / 2
    end_1b = home_back + Vector((diag, diag, 0)) * BASELINE_M
    end_3b = home_back + Vector((-diag, diag, 0)) * BASELINE_M

    _add_chalk_strip("Baseline_1B", home_back.x, home_back.y, end_1b.x, end_1b.y, z=0.005)
    _add_chalk_strip("Baseline_3B", home_back.x, home_back.y, end_3b.x, end_3b.y, z=0.005)


def create_infield_dirt_arc():
    """MLB infield skin — a region bounded by the two foul lines from home
    and a circular arc at 95 ft radius measured from the *mound center*
    (the origin in our Blender coords), not from home plate.

    Previous version measured the 95 ft from home plate, which meant the
    dirt only reached Y≈10m and never covered 2B (at Y≈20m). This version
    extends the dirt all the way around to the back of the infield.

    Shape: apex at home, edges along the 1B and 3B foul lines out to where
    they intersect the arc, arc across the back. Triangulated as a fan
    from home — valid because the region is star-shaped with respect to
    home plate.
    """
    home_back_x, home_back_y = 0.0, -MOUND_TO_PLATE_M
    arc_radius = 95 * FT_TO_M  # 28.96m from mound center at origin

    # Intersect each foul line with the mound-centered arc.
    # 1B foul line: (t * √2/2, home_y + t * √2/2), t ≥ 0
    # On arc:  x² + y² = arc_radius²
    #  =>  t² + 2·home_y·(√2/2)·t + (home_y² - R²) = 0
    diag = math.sqrt(2) / 2
    b = 2 * home_back_y * diag
    c = home_back_y ** 2 - arc_radius ** 2
    disc = b * b - 4 * c
    if disc < 0:
        print("  Infield arc: foul line misses the mound arc, falling back to small fan")
        t = 38.0  # safe fallback length along foul line
    else:
        t = (-b + math.sqrt(disc)) / 2

    foul_1b_end = (t * diag, home_back_y + t * diag)
    foul_3b_end = (-t * diag, home_back_y + t * diag)

    # Angular sweep of the arc between the two foul-line intersections.
    # Measured from the +X axis (atan2 convention). The arc sweeps through
    # the back of the infield (+Y side), so we go counterclockwise.
    start_angle = math.atan2(foul_1b_end[1], foul_1b_end[0])
    end_angle = math.atan2(foul_3b_end[1], foul_3b_end[0])
    if end_angle < start_angle:
        end_angle += 2 * math.pi

    verts = [(home_back_x, home_back_y, 0.0008)]  # fan apex at home

    n_arc = 48
    for i in range(n_arc + 1):
        frac = i / n_arc
        angle = start_angle + frac * (end_angle - start_angle)
        x = arc_radius * math.cos(angle)
        y = arc_radius * math.sin(angle)
        verts.append((x, y, 0.0008))

    # Fan triangulation from home plate to each pair of consecutive arc points.
    # The first triangle runs along the 1B foul line; the last along the 3B
    # foul line; the middle triangles sweep through the back of the infield.
    faces_idx = [(0, i, i + 1) for i in range(1, n_arc + 1)]

    mesh = bpy.data.meshes.new("InfieldDirt")
    mesh.from_pydata(verts, [], faces_idx)
    obj = bpy.data.objects.new("InfieldDirt", mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(_dirt_material("InfieldDirtMat"))


# ─── FIELD DETAIL (A-patch) ──────────────────────────────────────────────────
# All geometry below uses solid Principled BSDF materials so the scene
# round-trips cleanly to glTF/Three.js. No procedural shaders, no
# Geometry Nodes — what you see in Blender is what the browser gets.


def _infield_grass_material(name: str):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    # Slightly darker/richer green than outfield grass so the cutout shows.
    bsdf.inputs["Base Color"].default_value = (0.17, 0.42, 0.13, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.9
    return mat


def _outfield_grass_material(name: str):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.13, 0.35, 0.11, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.9
    return mat


def _wall_material(name: str):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.05, 0.15, 0.06, 1.0)  # dark green
    bsdf.inputs["Roughness"].default_value = 0.85
    return mat


def _foul_pole_material(name: str):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (1.0, 0.85, 0.0, 1.0)  # yellow
    bsdf.inputs["Roughness"].default_value = 0.4
    return mat


def create_infield_grass_diamond():
    """Infield grass cutout — a diamond inside the baselines so a dirt
    strip remains around the bases and along the baseline paths. Matches
    the 'cutout' look of most MLB parks.

    Sits at Z=0.002: above the dirt arc (0.0008) and below chalk strips
    (0.005+), so baselines remain visible on top."""
    home_back = Vector((0, -MOUND_TO_PLATE_M, 0))
    diag = math.sqrt(2) / 2
    to_1b = Vector((diag, diag, 0))
    to_3b = Vector((-diag, diag, 0))
    to_2b_axis = Vector((0, 1, 0))

    pos_1b = home_back + to_1b * BASELINE_M
    pos_3b = home_back + to_3b * BASELINE_M
    pos_2b = home_back + to_2b_axis * (BASELINE_M * math.sqrt(2))

    # Inset each diamond corner ~8 ft toward the infield center so a dirt
    # baseline strip shows along each of the four paths.
    infield_center = home_back + to_2b_axis * (BASELINE_M * math.sqrt(2) / 2)
    inset_m = 8 * FT_TO_M

    def inset_toward_center(pos: Vector) -> Vector:
        direction = (infield_center - pos).normalized()
        return pos + direction * inset_m

    corners = [
        inset_toward_center(home_back),
        inset_toward_center(pos_1b),
        inset_toward_center(pos_2b),
        inset_toward_center(pos_3b),
    ]

    z = 0.002
    verts = [(c.x, c.y, z) for c in corners]
    faces_idx = [(0, 1, 2, 3)]

    mesh = bpy.data.meshes.new("InfieldGrass")
    mesh.from_pydata(verts, [], faces_idx)
    obj = bpy.data.objects.new("InfieldGrass", mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(_infield_grass_material("InfieldGrassMat"))
    return obj


def create_running_lane():
    """3 ft wide chalk running lane in foul territory, from the midpoint
    between home and 1B to 1B. Inner edge is the 1B foul line itself
    (already drawn as FoulLine_1B), so we only draw the outer edge plus
    two cross-ticks at the start and end of the lane."""
    home_back = Vector((0, -MOUND_TO_PLATE_M, 0))
    diag = math.sqrt(2) / 2
    to_1b = Vector((diag, diag, 0))
    # Perpendicular into foul territory (right-hand rotation of to_1b by -90°).
    perp_foul = Vector((diag, -diag, 0))
    lane_w = 3 * FT_TO_M  # 0.914m

    pos_1b = home_back + to_1b * BASELINE_M
    mid = home_back + to_1b * (BASELINE_M / 2)

    # Inner = on the foul line. Outer = 3 ft into foul territory.
    inner_start = mid
    outer_start = mid + perp_foul * lane_w
    inner_end = pos_1b
    outer_end = pos_1b + perp_foul * lane_w

    # Outer parallel line
    _add_chalk_strip(
        "RunningLane_outer",
        outer_start.x, outer_start.y, outer_end.x, outer_end.y,
        z=0.006,
    )
    # Cross ticks at start and end (so the lane reads as a closed rectangle)
    _add_chalk_strip(
        "RunningLane_start",
        inner_start.x, inner_start.y, outer_start.x, outer_start.y,
        z=0.006,
    )
    _add_chalk_strip(
        "RunningLane_end",
        inner_end.x, inner_end.y, outer_end.x, outer_end.y,
        z=0.006,
    )


def create_on_deck_circles():
    """Two 5 ft diameter on-deck circles, one per side, in foul territory
    where the dugouts would be. Rendered as filled tan discs — MLB on-deck
    circles are either dirt patches or logo discs; a flat tan circle reads
    correctly from every camera angle."""
    radius = 2.5 * FT_TO_M  # 5 ft diameter
    z = 0.003

    home_back = Vector((0, -MOUND_TO_PLATE_M, 0))
    diag = math.sqrt(2) / 2
    # Placement: roughly where on-deck circles sit in a typical MLB park —
    # between home plate and the on-deck-side baseline, offset into foul
    # territory. Two symmetric positions.
    offset_along_baseline = 12 * FT_TO_M
    offset_perp_foul = 14 * FT_TO_M

    positions = {
        "OnDeck_1B": home_back
            + Vector((diag, diag, 0)) * offset_along_baseline
            + Vector((diag, -diag, 0)) * offset_perp_foul,
        "OnDeck_3B": home_back
            + Vector((-diag, diag, 0)) * offset_along_baseline
            + Vector((-diag, -diag, 0)) * offset_perp_foul,
    }

    for name, pos in positions.items():
        n_segs = 32
        verts = [(pos.x, pos.y, z)]  # center
        for i in range(n_segs):
            angle = 2 * math.pi * i / n_segs
            verts.append((pos.x + radius * math.cos(angle),
                          pos.y + radius * math.sin(angle),
                          z))
        faces_idx = []
        for i in range(1, n_segs):
            faces_idx.append((0, i, i + 1))
        faces_idx.append((0, n_segs, 1))

        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(verts, [], faces_idx)
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(obj)

        mat = bpy.data.materials.new(f"{name}_Mat")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (0.62, 0.48, 0.28, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.95
        obj.data.materials.append(mat)


def create_coaches_boxes():
    """Two 20 ft × 10 ft chalk coach's boxes in foul territory alongside
    1B and 3B. Drawn as hollow chalk rectangles — 4 strips per box.

    The box runs parallel to the baseline, with its inner long edge set
    back ~6 ft from the foul line (into foul territory). The near end
    starts 10 ft behind the base toward home; the far end is 20 ft
    beyond that."""
    box_len = 20 * FT_TO_M        # 6.10m along baseline
    box_wid = 10 * FT_TO_M        # 3.05m perpendicular
    perp_inset = 6 * FT_TO_M      # inner edge sits this far off the foul line
    behind_base = 10 * FT_TO_M    # how far behind the base the near end starts

    home_back = Vector((0, -MOUND_TO_PLATE_M, 0))
    diag = math.sqrt(2) / 2

    configs = {
        "CoachBox_1B": {
            "along": Vector((diag, diag, 0)),    # home → 1B
            "perp_foul": Vector((diag, -diag, 0)),
            "base_pos": home_back + Vector((diag, diag, 0)) * BASELINE_M,
        },
        "CoachBox_3B": {
            "along": Vector((-diag, diag, 0)),
            "perp_foul": Vector((-diag, -diag, 0)),
            "base_pos": home_back + Vector((-diag, diag, 0)) * BASELINE_M,
        },
    }

    for name, cfg in configs.items():
        along = cfg["along"]
        perp = cfg["perp_foul"]
        near_edge = cfg["base_pos"] - along * behind_base
        far_edge = near_edge + along * box_len
        inner_near = near_edge + perp * perp_inset
        outer_near = inner_near + perp * box_wid
        inner_far = far_edge + perp * perp_inset
        outer_far = inner_far + perp * box_wid

        edges = {
            "inner": (inner_near, inner_far),
            "outer": (outer_near, outer_far),
            "near":  (inner_near, outer_near),
            "far":   (inner_far, outer_far),
        }
        for side, (a, b) in edges.items():
            _add_chalk_strip(f"{name}_{side}", a.x, a.y, b.x, b.y, z=0.006)


# Outfield dimensions for the B-patch scenery.
# MLB parks vary wildly (330–400+ ft). A single circular arc at ~355 ft
# is a symmetric compromise that reads as "a ballpark" from every angle.
OUTFIELD_WALL_RADIUS_M = 108.0            # ~354 ft from home
WARNING_TRACK_WIDTH_M = 15 * FT_TO_M       # 4.57m
WALL_HEIGHT_M = 2.5                        # ~8 ft wall
FOUL_POLE_HEIGHT_M = 15.0                  # tall for visibility
FOUL_POLE_RADIUS_M = 0.1


def _outfield_arc_points(radius: float, n_segments: int = 64):
    """Return (x, y) points along the outfield arc, home plate at arc center.
    Arc spans -45° to +45° (from home's +Y axis) — i.e. fair territory only."""
    home_y = -MOUND_TO_PLATE_M
    pts = []
    for i in range(n_segments + 1):
        angle = math.radians(-45 + 90 * i / n_segments)
        x = radius * math.sin(angle)
        y = home_y + radius * math.cos(angle)
        pts.append((x, y))
    return pts


def create_warning_track():
    """15 ft wide dirt ring along the inside of the outfield wall arc."""
    inner_r = OUTFIELD_WALL_RADIUS_M - WARNING_TRACK_WIDTH_M
    outer_r = OUTFIELD_WALL_RADIUS_M
    n_segments = 64

    inner_pts = _outfield_arc_points(inner_r, n_segments)
    outer_pts = _outfield_arc_points(outer_r, n_segments)

    z = 0.0012  # just above outfield grass, below the wall base
    verts = []
    for (ix, iy), (ox, oy) in zip(inner_pts, outer_pts):
        verts.append((ix, iy, z))
        verts.append((ox, oy, z))

    faces_idx = []
    for i in range(n_segments):
        i0 = 2 * i
        i1 = 2 * i + 1
        i2 = 2 * (i + 1)
        i3 = 2 * (i + 1) + 1
        faces_idx.append((i0, i1, i3, i2))

    mesh = bpy.data.meshes.new("WarningTrack")
    mesh.from_pydata(verts, [], faces_idx)
    obj = bpy.data.objects.new("WarningTrack", mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(_dirt_material("WarningTrackMat"))
    return obj


def create_outfield_wall():
    """Extruded arc wall along the outfield edge. 8 ft tall, dark green.
    Built as a strip of vertical quads along the arc."""
    n_segments = 64
    arc_pts = _outfield_arc_points(OUTFIELD_WALL_RADIUS_M, n_segments)

    verts = []
    for x, y in arc_pts:
        verts.append((x, y, 0.0))
        verts.append((x, y, WALL_HEIGHT_M))

    faces_idx = []
    for i in range(n_segments):
        b0 = 2 * i       # bottom near
        t0 = 2 * i + 1   # top near
        b1 = 2 * (i + 1) # bottom far
        t1 = 2 * (i + 1) + 1
        faces_idx.append((b0, b1, t1, t0))

    mesh = bpy.data.meshes.new("OutfieldWall")
    mesh.from_pydata(verts, [], faces_idx)
    obj = bpy.data.objects.new("OutfieldWall", mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(_wall_material("OutfieldWallMat"))
    return obj


def create_foul_poles():
    """Two tall yellow cylinders at the ends of the outfield wall (where
    the foul lines meet the wall)."""
    home_y = -MOUND_TO_PLATE_M
    angle_1b = math.radians(45)
    angle_3b = math.radians(-45)

    positions = {
        "FoulPole_1B": (
            OUTFIELD_WALL_RADIUS_M * math.sin(angle_1b),
            home_y + OUTFIELD_WALL_RADIUS_M * math.cos(angle_1b),
        ),
        "FoulPole_3B": (
            OUTFIELD_WALL_RADIUS_M * math.sin(angle_3b),
            home_y + OUTFIELD_WALL_RADIUS_M * math.cos(angle_3b),
        ),
    }

    for name, (x, y) in positions.items():
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=16,
            radius=FOUL_POLE_RADIUS_M,
            depth=FOUL_POLE_HEIGHT_M,
            location=(x, y, FOUL_POLE_HEIGHT_M / 2),
        )
        pole = bpy.context.active_object
        pole.name = name
        pole.data.materials.append(_foul_pole_material(f"{name}_Mat"))


def _add_chalk_strip(name: str, x0: float, y0: float, x1: float, y1: float, z: float = 0.005):
    """Add a thin white chalk strip from (x0,y0) to (x1,y1) at height z.
    Builds a quad with CHALK_HALF_W_M perpendicular thickness so it's
    visible from the catcher cam at distance."""
    dx, dy = x1 - x0, y1 - y0
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1e-6:
        return
    # Perpendicular vector (normalized × half width)
    px = -dy / length * CHALK_HALF_W_M
    py = +dx / length * CHALK_HALF_W_M

    verts = [
        (x0 - px, y0 - py, z),
        (x0 + px, y0 + py, z),
        (x1 + px, y1 + py, z),
        (x1 - px, y1 - py, z),
    ]
    faces_idx = [(0, 1, 2, 3)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces_idx)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(_white_material(f"{name}_Mat"))


def create_distance_markers():
    """Add text labels for key distances."""
    markers = [
        ("60'6\" (18.44m)", (2, -MOUND_TO_PLATE_M / 2, 0.01), 0.3),
        ("Mound", (MOUND_RADIUS_M + 0.5, 0, 0.01), 0.25),
        ("Home", (1.0, -MOUND_TO_PLATE_M - 0.5, 0.01), 0.25),
    ]

    for text, location, size in markers:
        bpy.ops.object.text_add(location=location)
        txt = bpy.context.active_object
        txt.data.body = text
        txt.data.size = size
        txt.name = f"Label_{text[:10]}"
        # Rotate to face up (text lies flat on ground)
        txt.rotation_euler = (math.radians(90), 0, 0)

        mat = bpy.data.materials.new(f"Label_{text[:6]}")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        txt.data.materials.append(mat)


# ─── CAMERA PRESETS ──────────────────────────────────────────────────────────

def add_camera(name, location, target=(0, 0, 1.0), lens=50):
    """
    Create a named camera pointing at target.
    target defaults to approximate chest height of pitcher on mound.
    """
    cam_data = bpy.data.cameras.new(name)
    cam_data.lens = lens
    cam_data.clip_end = 200

    cam_obj = bpy.data.objects.new(name, cam_data)
    bpy.context.collection.objects.link(cam_obj)

    cam_obj.location = Vector(location)

    # Point camera at target
    direction = Vector(target) - Vector(location)
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_euler = rot_quat.to_euler()

    return cam_obj


def setup_cameras():
    """Create analysis camera presets."""
    # Target: pitcher's torso, roughly chest height on the mound
    pitcher_chest = (0, 0, MOUND_RISE_M + 1.4)

    cameras = {}

    # 1. Catcher's view — behind home plate looking at pitcher
    cameras["Cam_Catcher"] = add_camera(
        "Cam_Catcher",
        location=(0, -MOUND_TO_PLATE_M - 2, 1.2),
        target=pitcher_chest,
        lens=85,
    )

    # 2. Side view from 1B side — shows arm slot and stride length
    cameras["Cam_Side_1B"] = add_camera(
        "Cam_Side_1B",
        location=(12, -3, 1.5),
        target=pitcher_chest,
        lens=70,
    )

    # 3. Side view from 3B side
    cameras["Cam_Side_3B"] = add_camera(
        "Cam_Side_3B",
        location=(-12, -3, 1.5),
        target=pitcher_chest,
        lens=70,
    )

    # 4. Overhead — top-down for landing alignment
    cameras["Cam_Overhead"] = add_camera(
        "Cam_Overhead",
        location=(0, -3, 20),
        target=(0, -3, 0),
        lens=35,
    )

    # 5. Broadcast angle — center field camera
    cameras["Cam_Broadcast"] = add_camera(
        "Cam_Broadcast",
        location=(0, 25, 5),
        target=(0, -MOUND_TO_PLATE_M / 2, 1.0),
        lens=135,
    )

    # 6. 45-degree — shows both arm path and stride
    cameras["Cam_45deg"] = add_camera(
        "Cam_45deg",
        location=(8, 8, 4),
        target=pitcher_chest,
        lens=60,
    )

    # 7. Close-up arm slot — zoomed on release point area
    cameras["Cam_ArmSlot"] = add_camera(
        "Cam_ArmSlot",
        location=(4, 1, 2.5),
        target=(0, 0, MOUND_RISE_M + 1.8),  # shoulder/arm height
        lens=100,
    )

    # Set catcher view as default
    bpy.context.scene.camera = cameras["Cam_Catcher"]

    return cameras


# ─── LIGHTING ────────────────────────────────────────────────────────────────

def setup_lighting():
    """Stadium-style lighting: overhead sun + fill."""
    # Main sun light (high noon, slightly behind pitcher)
    bpy.ops.object.light_add(type='SUN', location=(0, 5, 20))
    sun = bpy.context.active_object
    sun.name = "Sun_Main"
    sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(30), 0, 0)

    # Fill light from catcher side
    bpy.ops.object.light_add(type='AREA', location=(0, -15, 8))
    fill = bpy.context.active_object
    fill.name = "Fill_Catcher"
    fill.data.energy = 100
    fill.data.size = 5

    # Ambient — use world background
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.5, 0.65, 0.85, 1.0)  # sky blue
    bg.inputs["Strength"].default_value = 0.3


# ─── MESH IMPORT ─────────────────────────────────────────────────────────────

def import_pitcher_mesh(filepath):
    """
    Import the pitcher mesh (.glb, .obj, or .npz) and position on the rubber.

    The SAM 3D Body pipeline outputs in Y-up, meters.
    The ground plane aligner already places the mesh at the correct
    mound height with ankles at ground + MOUND_HEIGHT_M.

    For GLB: Blender's GLTF importer handles Y-up → Z-up conversion.
    For OBJ: Blender's OBJ importer handles axis conversion.
    For NPZ: we build the mesh directly in Blender, converting Y-up → Z-up.
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".npz":
        return _import_npz(filepath)

    if ext == ".glb" or ext == ".gltf":
        bpy.ops.import_scene.gltf(filepath=filepath)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=filepath)
    else:
        print(f"Unsupported format: {ext}. Use .glb, .obj, or .npz")
        return None

    # The imported object(s) should be selected
    imported = bpy.context.selected_objects
    if not imported:
        print("No objects imported!")
        return None

    # If multiple objects imported, parent them to an empty
    if len(imported) > 1:
        bpy.ops.object.empty_add(location=(0, 0, 0))
        parent = bpy.context.active_object
        parent.name = "PitcherMesh"
        for obj in imported:
            obj.parent = parent
        mesh_root = parent
    else:
        mesh_root = imported[0]
        mesh_root.name = "PitcherMesh"

    _apply_pitcher_material(imported)
    return mesh_root


def _import_npz(filepath):
    """
    Build a Blender mesh directly from an .npz file.

    Loads vertices (T, 18439, 3) and faces (36874, 3).
    Creates the base mesh from frame 0, then adds remaining frames
    as shape keys for animation scrubbing.

    Pipeline coordinate frame (camera-space, OpenCV/image convention):
        +X right, +Y down, +Z into screen (away from camera)

    Blender coordinate frame:
        +X right, +Y away from viewer, +Z up

    Mapping (verified empirically against the catcher cam preset and a
    right-handed pitcher's throwing arm):
        blender_x = -pipeline_x   (un-mirrors handedness — see parity note)
        blender_y = -pipeline_z   (source video is shot from CENTER FIELD,
                                    so pitcher's nose is at larger pipeline_z;
                                    negating puts the nose at -Y, closer to
                                    the catcher cam at (0, -20.4, 1.2))
        blender_z = -pipeline_y   (image-down flips to up)

    Parity note: each axis swap or negation flips parity. Even total = pure
    rotation (handedness preserved); odd total = mirror (right-handed
    pitcher appears left-handed). Counting: 1 swap (Y↔Z) + 3 negations =
    4 = even. Negating X is what brings us to even — without it we had
    1 swap + 2 negations = 3 = mirror.

    History of bugs in this function:
      - Original   (x, -z, y):  upside down AND facing wrong way
      - First fix  (x, +z, -y): right-side up, facing wrong way
      - Second fix (x, -z, -y): right-side up, facing catcher, MIRRORED
      - Current    (-x, -z, -y): right-side up, facing catcher, correct chirality
    """
    data = np.load(filepath)
    vertices = data["vertices"]  # (T, V, 3)

    # Faces: check npz first, then fall back to cached faces.npy
    if "faces" in data:
        faces = data["faces"]
    else:
        script_dir = os.path.dirname(os.path.abspath(filepath))
        # Walk up to find data/faces.npy
        search = filepath
        for _ in range(5):
            search = os.path.dirname(search)
            candidate = os.path.join(search, "data", "faces.npy")
            if os.path.exists(candidate):
                faces = np.load(candidate)
                print(f"Loaded faces from cache: {candidate}")
                break
        else:
            print("Error: no faces in .npz and no data/faces.npy cache found.")
            print("Run: python scripts/export_mesh.py --backfill-faces")
            return None

    n_frames = vertices.shape[0]
    n_verts = vertices.shape[1]

    # Source FPS: prefer the value baked into the .npz at inference time,
    # fall back to the SOURCE_FPS module default. Stored on the object so
    # setup_animation can pick it up without a global.
    if "source_fps" in data:
        source_fps = float(data["source_fps"])
    else:
        source_fps = float(SOURCE_FPS)

    print(f"NPZ: {n_frames} frames, {n_verts} vertices, "
          f"{faces.shape[0]} faces @ {source_fps:.2f} fps")

    # Scale audit: body height vs MLB-spec field dimensions.
    # Body height = max vertical span across all frames in pipeline coords.
    # In pipeline +Y is image-down, so vertical span = pipeline_y range.
    body_y_span_per_frame = (vertices[:, :, 1].max(axis=1)
                              - vertices[:, :, 1].min(axis=1))
    body_h_max = float(body_y_span_per_frame.max())
    body_h_med = float(np.median(body_y_span_per_frame))
    print(f"  Scale audit: body {body_h_med:.2f}m median / {body_h_max:.2f}m peak  "
          f"|  mound {MOUND_DIAMETER_FT:.0f}ft dia × {MOUND_RISE_M*100:.0f}cm rise  "
          f"|  rubber→plate {MOUND_TO_PLATE_M:.2f}m")

    # See docstring above for the derivation. Final mapping:
    #   blender = (-pipeline_x, -pipeline_z, -pipeline_y)
    # Vectorized — keeps shape (V, 3) so we can apply ground-plane offset
    # in one shot rather than per-vertex tuples.
    def y_up_to_z_up_arr(v: np.ndarray) -> np.ndarray:
        out = np.empty_like(v, dtype=np.float32)
        out[:, 0] = -v[:, 0]
        out[:, 1] = -v[:, 2]
        out[:, 2] = -v[:, 1]
        return out

    # ------------------------------------------------------------------
    # Foot-lock offsets (per-frame translation applied at object level,
    # NOT baked into shape keys).
    #
    # Two modes are precomputed and stored on the object so the user can
    # toggle between them in the N-panel without re-launching:
    #
    #   "ankle" (Mode A, default — full 3D anchor):
    #       Tracks the right ankle joint (joints_mhr70 index 14, the
    #       PIVOT FOOT for a right-handed pitcher) and translates the
    #       entire body so that joint sits exactly on the rubber across
    #       all three axes (X, Y, Z). The planted foot stays glued to the
    #       rubber from set position through release. The body rotates
    #       and pivots around that anchor.
    #
    #       Caveat: in reality the pivot foot lifts off the rubber late
    #       in the follow-through. We don't have world translation in the
    #       model output so we can't track that step-off — the body will
    #       appear to rotate around the rubber instead of stepping
    #       forward during follow-through. Acceptable for mechanics
    #       analysis through release.
    #
    #   "vertex" (Mode B — Z-only float prevention):
    #       For each frame, use the literal lowest mesh vertex Z and
    #       translate Z only. Doesn't anchor X/Y, so the body slides
    #       around but never floats above the ground. Useful for
    #       comparison; you saw this make the body "float" earlier.
    #
    # Rationale: SAM 3D Body outputs body-canonical poses (no world
    # translation). Per-frame offsets are the only way to make the
    # animation look grounded.
    # ------------------------------------------------------------------
    RUBBER_Z = MOUND_RISE_M + 0.005  # 5mm clearance above mound surface
    RUBBER_POS = np.array([0.0, 0.0, RUBBER_Z], dtype=np.float32)

    # Convert ALL frames once. (T, V, 3) blender coords.
    blender_verts = np.empty_like(vertices, dtype=np.float32)
    blender_verts[:, :, 0] = -vertices[:, :, 0]
    blender_verts[:, :, 1] = -vertices[:, :, 2]
    blender_verts[:, :, 2] = -vertices[:, :, 1]

    # Mode B (vertex Z-only): lowest vertex per frame, Z only.
    vertex_lowest_z = blender_verts[:, :, 2].min(axis=1)  # (T,)
    vertex_lock_offsets_z = (RUBBER_Z - vertex_lowest_z).astype(np.float32)
    vertex_lock_offsets = np.zeros((n_frames, 3), dtype=np.float32)
    vertex_lock_offsets[:, 2] = vertex_lock_offsets_z

    # Mode A (ankle anchor — X/Y per frame, Z static):
    #
    # The hybrid: compute a static Z offset from frame 0 so the body sits
    # on the rubber vertically (no bobbing), then compute per-frame X/Y
    # offsets that compensate for the pivot foot's lateral drift in
    # canonical space. The pivot foot stays glued to the rubber laterally
    # while the body never bobs up/down.
    #
    # Pivot detection: the pivot foot has *less* total motion across the
    # clip than the stride foot. We auto-detect rather than hardcoding,
    # so this works for both right- and left-handed pitchers.
    #
    # We deliberately do NOT compensate Z drift — the model's per-frame
    # canonical Z is too noisy (we measured 1m of drift, which would
    # bob the body by 1m vertically). The static Z from frame 0 keeps
    # the body's vertical position stable.
    if 'joints_mhr70' in data and data['joints_mhr70'].shape[1] > 14:
        ankles_pipeline = data['joints_mhr70'][:, [13, 14], :]  # (T, 2, 3)
        ankles_blender = np.empty_like(ankles_pipeline, dtype=np.float32)
        ankles_blender[:, :, 0] = -ankles_pipeline[:, :, 0]
        ankles_blender[:, :, 1] = -ankles_pipeline[:, :, 2]
        ankles_blender[:, :, 2] = -ankles_pipeline[:, :, 1]

        # Compute motion totals for diagnostic purposes (and as a fallback
        # when handedness is missing).
        motion = []
        for k in range(2):
            d = np.linalg.norm(np.diff(ankles_blender[:, k, :], axis=0), axis=1)
            motion.append(float(d.sum()))

        # Resolve pivot foot in priority order:
        #   1. NPZ 'handedness' field (most authoritative)
        #   2. DEFAULT_HANDEDNESS module constant (env var or top-of-file)
        #   3. Motion-based heuristic (last resort)
        # The motion heuristic is unreliable on long clips with walk-off:
        # both feet end up with similar total motion (e.g. L=14.51m vs
        # R=14.90m) and the choice flips arbitrarily.
        #
        # RHP → pivot = right ankle (mhr70[14], pair index 1)
        # LHP → pivot = left ankle  (mhr70[13], pair index 0)
        def _handedness_to_pair_idx(h: str):
            h = h.lower().strip()
            if h.startswith('r'):
                return 1
            if h.startswith('l'):
                return 0
            return None

        pivot_idx_in_pair = None
        pivot_source = None
        if 'handedness' in data:
            pivot_idx_in_pair = _handedness_to_pair_idx(str(data['handedness']))
            if pivot_idx_in_pair is not None:
                pivot_source = f"NPZ handedness={data['handedness']}"

        if pivot_idx_in_pair is None:
            pivot_idx_in_pair = _handedness_to_pair_idx(DEFAULT_HANDEDNESS)
            if pivot_idx_in_pair is not None:
                pivot_source = f"DEFAULT_HANDEDNESS={DEFAULT_HANDEDNESS}"

        if pivot_idx_in_pair is None:
            pivot_idx_in_pair = int(np.argmin(motion))
            pivot_source = "motion heuristic (fallback)"

        pivot_label = "left (mhr70[13])" if pivot_idx_in_pair == 0 else "right (mhr70[14])"
        print(f"  Pivot foot: {pivot_label} [{pivot_source}]  "
              f"motion: L={motion[0]:.2f}m  R={motion[1]:.2f}m")

        pivot_traj = ankles_blender[:, pivot_idx_in_pair, :]  # (T, 3) blender coords

        # Static Z: snap frame 0's lowest vertex to the rubber surface,
        # apply that same Z offset to every frame.
        static_z = RUBBER_Z - float(blender_verts[0, :, 2].min())

        # Per-frame X/Y: compensate so the pivot foot stays at the
        # rubber's lateral position (0, 0). offset_x = -pivot_x.
        ankle_lock_offsets = np.zeros((n_frames, 3), dtype=np.float32)
        ankle_lock_offsets[:, 0] = -pivot_traj[:, 0]  # X compensation
        ankle_lock_offsets[:, 1] = -pivot_traj[:, 1]  # Y compensation
        ankle_lock_offsets[:, 2] = static_z           # static Z, no bobbing
    else:
        print("  WARN: no joints_mhr70 in npz, ankle lock falls back to vertex Z-only")
        ankle_lock_offsets = vertex_lock_offsets.copy()

    print(f"  Foot-lock offset stats (per-axis range):")
    for label, off in [("ankle (3D)", ankle_lock_offsets), ("vertex (Z-only)", vertex_lock_offsets)]:
        print(f"    {label:18s} "
              f"X[{off[:,0].min():+.2f},{off[:,0].max():+.2f}] "
              f"Y[{off[:,1].min():+.2f},{off[:,1].max():+.2f}] "
              f"Z[{off[:,2].min():+.2f},{off[:,2].max():+.2f}]")

    # Build base mesh from frame 0 — body-canonical, no Z offset baked in.
    base_verts = [tuple(map(float, v)) for v in blender_verts[0]]
    face_list = [tuple(int(i) for i in f) for f in faces]

    mesh = bpy.data.meshes.new("PitcherMesh")
    mesh.from_pydata(base_verts, [], face_list)
    mesh.update()

    obj = bpy.data.objects.new("PitcherMesh", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Phase markers from joint trajectories (used by the HUD).
    phase_markers = _detect_pitch_phases(
        data['joints_mhr70'] if 'joints_mhr70' in data else None,
        handedness="right",
    )
    print(f"  Phase markers: {phase_markers}")

    # Stash everything setup_animation and the frame handler need.
    # Offsets are (T, 3) — full XYZ translation per frame.
    obj["source_fps"] = source_fps
    obj["ankle_lock_offsets"] = ankle_lock_offsets.tolist()
    obj["vertex_lock_offsets"] = vertex_lock_offsets.tolist()
    obj["foot_lock_mode"] = "ankle"  # default; toggle in N-panel
    obj["n_frames"] = int(n_frames)
    # Phase markers as parallel int/str lists (Blender ID props don't
    # support arbitrary dicts well; the HUD reconstructs the mapping).
    sorted_markers = sorted(phase_markers.items())
    obj["phase_frames"] = [int(f) for f, _ in sorted_markers]
    obj["phase_labels"] = [str(lbl) for _, lbl in sorted_markers]

    # Initial position: frame 0 ankle-lock offset (full XYZ)
    obj.location = (
        float(ankle_lock_offsets[0, 0]),
        float(ankle_lock_offsets[0, 1]),
        float(ankle_lock_offsets[0, 2]),
    )

    # Add shape keys for animation (if multiple frames)
    if n_frames > 1:
        obj.shape_key_add(name="Basis", from_mix=False)

        for t in range(1, n_frames):
            sk = obj.shape_key_add(name=f"Frame_{t:03d}", from_mix=False)
            for vi in range(blender_verts.shape[1]):
                sk.data[vi].co = (
                    float(blender_verts[t, vi, 0]),
                    float(blender_verts[t, vi, 1]),
                    float(blender_verts[t, vi, 2]),
                )

            if t % 20 == 0:
                print(f"  Shape keys: {t}/{n_frames - 1}...")

        print(f"  Created {n_frames - 1} shape keys")

    _apply_pitcher_material([obj])

    # Stash .npz path on the object so blender_analysis_overlays.py can
    # re-open the joint data without guessing.
    obj["npz_path"] = filepath
    obj["source_fps"] = float(source_fps)

    return obj


def _apply_pitcher_material(objects):
    """Apply skin-tone material to mesh objects."""
    mat = bpy.data.materials.new("PitcherBody")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.7, 0.55, 0.45, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.7
    bsdf.inputs["Alpha"].default_value = 0.85
    if hasattr(mat, 'blend_method'):
        mat.blend_method = 'BLEND'

    for obj in objects:
        if obj.type == 'MESH':
            obj.data.materials.clear()
            obj.data.materials.append(mat)


def _detect_pitch_phases(joints_mhr70, handedness="right"):
    """Derive coarse pitch phases from joint trajectories.

    joints_mhr70: (T, 70, 3) array in pipeline coords (+Y down).
    Returns dict: {frame_index: phase_label}. Frames between markers
    inherit the most recent label, so the HUD can render any frame.

    Heuristics:
      - Set:           frame 0 → 5% before knee starts rising
      - Leg lift:      → frame of max stride-knee height
      - Stride:        → frame of stride-ankle min Z (foot strike)
      - Acceleration:  → frame of throwing-wrist max horizontal velocity
      - Release:       1 frame at peak wrist velocity
      - Follow-through: → end of clip

    All "Z" reasoning is in pipeline coords where +Y is image-down,
    so "knee height" = -knee_y (more negative = higher off ground).
    """
    if joints_mhr70 is None or joints_mhr70.shape[0] < 10:
        return {0: "set"}

    j = np.asarray(joints_mhr70, dtype=np.float32)  # (T, 70, 3)
    T = j.shape[0]

    # Stride leg = front leg = LEFT for right-handed pitcher
    stride_knee_idx = 11 if handedness == "right" else 12
    stride_ankle_idx = 13 if handedness == "right" else 14
    throw_wrist_idx = 41 if handedness == "right" else 62

    # "Up" in pipeline coords = -y. Knee height = -y[:, knee, 1].
    knee_up = -j[:, stride_knee_idx, 1]    # (T,)
    ankle_up = -j[:, stride_ankle_idx, 1]  # (T,)

    # Smooth a touch — these arrays can be noisy.
    def smooth(a, w=5):
        if len(a) < w:
            return a
        kernel = np.ones(w, dtype=np.float32) / w
        return np.convolve(a, kernel, mode="same")

    knee_up_s = smooth(knee_up)
    ankle_up_s = smooth(ankle_up)

    # Leg lift peak: max knee height
    leg_lift_frame = int(np.argmax(knee_up_s))

    # Foot strike: ankle minimum *after* the leg lift, within a 1.5s window.
    # The window matters — if we search to end-of-clip, the argmin can land
    # on a noisy tail frame (pitcher walking off the mound) and cascade into
    # release_frame == T-1, which then collides with follow-through in the
    # markers dict and wipes out both labels.
    if leg_lift_frame < T - 5:
        window = min(T - leg_lift_frame, 90)  # ~1.5s at 60fps
        post_lift = ankle_up_s[leg_lift_frame:leg_lift_frame + window]
        foot_strike_frame = leg_lift_frame + int(np.argmin(post_lift))
    else:
        foot_strike_frame = T - 1

    # Set→leg-lift transition: where knee starts rising. Find the first
    # frame after frame 5 whose knee height exceeds frame 0 by 10cm.
    baseline_knee = knee_up_s[:5].mean() if T > 5 else knee_up_s[0]
    set_end = leg_lift_frame
    for f in range(5, leg_lift_frame):
        if knee_up_s[f] - baseline_knee > 0.10:
            set_end = f
            break

    # Release: throwing wrist velocity peak (frame-to-frame magnitude)
    wrist_pos = j[:, throw_wrist_idx, :]  # (T, 3)
    wrist_vel = np.linalg.norm(np.diff(wrist_pos, axis=0), axis=1)  # (T-1,)
    wrist_vel_s = smooth(wrist_vel, w=3)
    # Look for peak after foot strike
    if foot_strike_frame < len(wrist_vel_s) - 2:
        post_strike = wrist_vel_s[foot_strike_frame:]
        release_frame = foot_strike_frame + int(np.argmax(post_strike))
    else:
        release_frame = min(foot_strike_frame + 3, T - 1)

    # Build a sparse marker dict; the HUD's lookup fills in gaps.
    markers = {
        0: "set",
        set_end: "leg lift",
        leg_lift_frame: "stride",
        foot_strike_frame: "acceleration",
        release_frame: "release",
        min(release_frame + 2, T - 1): "follow-through",
    }
    return markers


def _foot_lock_handler(scene, depsgraph=None):
    """frame_change_pre handler — animates obj.location per frame based
    on the active foot-lock mode. Each offset entry is a 3-vector
    (XYZ translation) so the body can be anchored in all axes, not just
    Z. Reads the precomputed offset arrays stashed on PitcherMesh."""
    obj = bpy.data.objects.get("PitcherMesh")
    if obj is None:
        return
    n_frames = obj.get("n_frames", 0)
    if n_frames <= 0:
        return
    frame = max(0, min(int(scene.frame_current), n_frames - 1))
    mode = obj.get("foot_lock_mode", "ankle")
    key = "ankle_lock_offsets" if mode == "ankle" else "vertex_lock_offsets"
    offsets = obj.get(key)
    if offsets is None:
        return
    off = offsets[frame]
    obj.location = (float(off[0]), float(off[1]), float(off[2]))

    # Update the HUD text in the same handler so the phase label and
    # mode indicator stay in sync with the playhead. Cheaper than a
    # second handler call.
    hud = bpy.data.objects.get("PhaseHUD")
    if hud is not None and hud.type == 'FONT':
        phase = _lookup_phase(obj, frame)
        hud.data.body = f"{phase.upper()}\nframe {frame}/{n_frames - 1}\nlock: {mode}"


def _lookup_phase(obj, frame):
    """Find the active phase label for a given frame using the sparse
    marker arrays stashed on the PitcherMesh object."""
    frames = obj.get("phase_frames")
    labels = obj.get("phase_labels")
    if not frames or not labels:
        return "—"
    active = labels[0]
    for f, lbl in zip(frames, labels):
        if frame >= int(f):
            active = lbl
        else:
            break
    return str(active)


def create_phase_hud():
    """Create a 3D text object pinned in front of the catcher cam.

    Parented to the camera so it follows the camera and always sits in
    the upper-left corner of the rendered frame. Updated per frame by
    _foot_lock_handler.
    """
    cam = bpy.data.objects.get("Cam_Catcher")
    if cam is None:
        return None

    bpy.ops.object.text_add(location=(0, 0, 0))
    hud = bpy.context.active_object
    hud.name = "PhaseHUD"
    hud.data.body = "SET\nframe 0\nlock: ankle"
    hud.data.size = 0.10
    hud.data.align_x = 'LEFT'
    hud.data.align_y = 'TOP'

    # Parent to the camera so the HUD lives in screen-space relative
    # to the catcher view. Position is in camera-local coords:
    #   x = -0.55 (left side of frame at 85mm lens)
    #   y = +0.30 (top of frame)
    #   z = -1.5  (1.5m in front of camera; just inside near clip)
    hud.parent = cam
    hud.location = (-0.55, 0.30, -1.5)
    hud.rotation_euler = (0, 0, 0)

    # Bright white emissive material so it pops against any background.
    mat = bpy.data.materials.new("PhaseHUDMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (1, 1, 1, 1)
    if "Emission Color" in bsdf.inputs:
        bsdf.inputs["Emission Color"].default_value = (1, 1, 1, 1)
        bsdf.inputs["Emission Strength"].default_value = 2.0
    elif "Emission" in bsdf.inputs:
        bsdf.inputs["Emission"].default_value = (1, 1, 1, 1)
    hud.data.materials.append(mat)

    return hud


def _install_foot_lock_handler():
    """Register the per-frame foot-lock handler exactly once."""
    handlers = bpy.app.handlers.frame_change_pre
    # Remove any previous instance (keeps re-runs idempotent)
    for h in list(handlers):
        if getattr(h, "__name__", "") == "_foot_lock_handler":
            handlers.remove(h)
    handlers.append(_foot_lock_handler)


def _shape_key_snap_handler(scene, depsgraph=None):
    """Per-frame handler: snap exactly one shape key to value=1.0.

    Bypasses the keyframe interpolation pipeline entirely. The keyframes
    inserted in setup_animation are still used by the GLB exporter (which
    samples per-frame and gets the right pose), but viewport playback in
    Blender 5.1 sometimes ignores the CONSTANT interpolation hint and
    blends multiple shape keys additively — making the body look like
    it's morphing wildly between distinct poses.

    The handler is the most direct fix: at every frame change, we walk
    the shape keys and set exactly the one matching the current frame to
    1.0, all others to 0.0. No interpolation, no additive blending,
    bulletproof.
    """
    target = bpy.data.objects.get("PitcherMesh")
    if target is None or target.type != 'MESH':
        return
    if not target.data.shape_keys:
        return
    keys = target.data.shape_keys.key_blocks
    if len(keys) < 2:
        return

    f = scene.frame_current
    n_keys = len(keys) - 1  # exclude Basis at index 0
    f = max(0, min(f, n_keys - 1))

    # keys[0] is Basis. keys[k] for k in [1..n_keys] are the per-frame poses.
    # Frame f → keys[f + 1].
    target_idx = f + 1
    for i in range(1, len(keys)):
        desired = 1.0 if i == target_idx else 0.0
        if keys[i].value != desired:
            keys[i].value = desired


def _install_shape_key_snap_handler():
    """Install the snap handler idempotently (remove old copies first)."""
    handlers = bpy.app.handlers.frame_change_pre
    handlers[:] = [h for h in handlers if getattr(h, "__name__", "") != "_shape_key_snap_handler"]
    handlers.append(_shape_key_snap_handler)


def setup_animation(mesh_root):
    """
    If the imported GLB has shape keys (morph targets), set up
    animation playback so the team can scrub through the pitch.
    """
    # Find the mesh object with shape keys
    target = None
    if mesh_root.type == 'MESH' and mesh_root.data.shape_keys:
        target = mesh_root
    else:
        for child in mesh_root.children_recursive:
            if child.type == 'MESH' and child.data and child.data.shape_keys:
                target = child
                break

    if target is None:
        print("No shape keys found — single frame mesh, no animation to set up.")
        return

    keys = target.data.shape_keys
    n_keys = len(keys.key_blocks) - 1  # exclude Basis

    if n_keys <= 0:
        return

    print(f"Found {n_keys} shape keys (frames). Setting up animation...")

    # Set scene frame range. FPS comes from the .npz when available
    # (stashed on the object by _import_npz), else falls back to the
    # SOURCE_FPS module default.
    fps = float(target.get("source_fps", SOURCE_FPS))
    # Blender's render.fps is an int + an optional fps_base divisor for
    # fractional rates like 59.94. Use that pattern so timing is exact.
    if abs(fps - round(fps)) < 1e-3:
        bpy.context.scene.render.fps = int(round(fps))
        bpy.context.scene.render.fps_base = 1.0
    else:
        # E.g. 59.94 → 60000 / 1001
        bpy.context.scene.render.fps = 60000
        bpy.context.scene.render.fps_base = 60000.0 / fps
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = n_keys - 1

    # CRITICAL: switch the *default* keyframe interpolation to CONSTANT
    # BEFORE inserting any keyframes. With Blender's default BEZIER, every
    # shape key whose pulse is in the future has a non-zero partial value
    # at intermediate frames, and Blender's additive shape-key formula
    # (result = Basis + Σ (shape - Basis) * value) sums all those partials
    # into a giant blob. With CONSTANT, exactly one shape key is ever at
    # value 1 per frame, so the formula collapses to a single pose.
    #
    # Doing this via user prefs (rather than walking f-curves after the
    # fact) is the only version-portable approach: in Blender 4.4+,
    # keyframe_insert creates "slotted actions" whose F-curves don't appear
    # under action.fcurves at all, so post-hoc interpolation switching
    # silently does nothing.
    prefs = bpy.context.preferences.edit
    prev_interp = prefs.keyframe_new_interpolation_type
    prefs.keyframe_new_interpolation_type = 'CONSTANT'
    try:
        # Sparse keyframes per shape key: a single 1.0 pulse at its target
        # frame, 0.0 immediately before and after. Total keyframes drop
        # from O(N²) to ~3N — also fixes scrub lag on long pitches.
        for k in range(1, len(keys.key_blocks)):  # 0 is Basis, skip it
            kb = keys.key_blocks[k]
            target_frame = k - 1

            if target_frame > 0:
                kb.value = 0.0
                kb.keyframe_insert(data_path="value", frame=0)

            kb.value = 1.0
            kb.keyframe_insert(data_path="value", frame=target_frame)

            if target_frame < n_keys - 1:
                kb.value = 0.0
                kb.keyframe_insert(data_path="value", frame=target_frame + 1)
    finally:
        prefs.keyframe_new_interpolation_type = prev_interp

    # Belt-and-suspenders: also walk every F-curve we can find and force
    # interpolation to CONSTANT in case the user pref didn't take effect
    # for some reason (custom Blender build, addon override, etc.).
    # Handles both legacy actions (action.fcurves) and Blender 4.4+
    # slotted actions (action.layers[*].strips[*].channelbag.fcurves).
    def _force_constant(fcurves):
        for fcurve in fcurves:
            for kp in fcurve.keyframe_points:
                kp.interpolation = 'CONSTANT'

    if keys.animation_data and keys.animation_data.action:
        action = keys.animation_data.action
        _force_constant(getattr(action, "fcurves", []))
        # Slotted-action path (Blender 4.4+).
        for layer in getattr(action, "layers", []):
            for strip in getattr(layer, "strips", []):
                for cb in getattr(strip, "channelbags", []):
                    _force_constant(getattr(cb, "fcurves", []))

    print(f"Animation: {n_keys} frames at {fps:.2f} fps "
          f"({n_keys / fps:.1f}s, CONSTANT interp)")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  SAM 3D Body — MLB Field Viewer")
    print("=" * 60)

    # Clear existing scene
    clear_scene()

    # Build the field — order matters for Z-fighting: outfield grass →
    # warning track → outfield wall → infield dirt → infield grass cutout
    # → mound dirt → mound → bases → chalk markings (highest Z).
    print("Building MLB field geometry...")
    create_ground_plane()
    create_warning_track()
    create_outfield_wall()
    create_foul_poles()
    create_infield_dirt_arc()
    create_infield_grass_diamond()
    create_dirt_circle()
    create_mound()
    create_rubber()
    create_home_plate()
    create_bases()
    create_baselines()
    create_batters_boxes()
    create_catchers_box()
    create_foul_lines()
    create_running_lane()
    create_on_deck_circles()
    create_coaches_boxes()
    create_distance_markers()

    # Cameras
    print("Setting up camera presets...")
    cameras = setup_cameras()
    print(f"  Created {len(cameras)} camera presets:")
    for name in cameras:
        print(f"    - {name}")

    # Phase HUD (parented to catcher cam, updates per frame)
    create_phase_hud()

    # Lighting
    setup_lighting()

    # Import pitcher mesh
    mesh_path = MESH_PATH
    if not mesh_path:
        # Find the project root robustly. Blender 5.1's Text Editor sets
        # __file__ to '/blender_field_viewer.py' (bare filename, leading
        # slash) for scripts opened via Text → Open, so the naive
        # os.path.dirname(__file__) returns '/' and the candidate paths
        # below resolve to '/data/...' which does not exist.
        project_root = None

        # 1. __file__ if it actually points at a real file
        try:
            f = os.path.abspath(__file__)
            if os.path.isfile(f):
                project_root = os.path.dirname(os.path.dirname(f))
        except Exception:
            pass

        # 2. Walk bpy.data.texts to find ANY blender_*.py with a real path
        if project_root is None:
            try:
                for text in bpy.data.texts:
                    fp = text.filepath
                    if not fp:
                        continue
                    resolved = bpy.path.abspath(fp)
                    if os.path.isfile(resolved) and os.path.basename(resolved).startswith("blender_"):
                        project_root = os.path.dirname(os.path.dirname(resolved))
                        break
            except Exception:
                pass

        # 3. Env var override
        if project_root is None:
            env_root = os.environ.get("SAMPLAYS_PROJECT_ROOT", "").strip()
            if env_root and os.path.isdir(env_root):
                project_root = env_root

        # 4. Last-resort hardcoded fallback
        if project_root is None:
            fallback = os.path.expanduser("~/Source/SamPlaysBaseball")
            if os.path.isdir(fallback):
                project_root = fallback

        if project_root:
            print(f"  Project root: {project_root}")
            candidates = [
                # Exported via export_mesh.py
                os.path.join(project_root, "data", "export", "02ec65f0.glb"),
                # Raw NPZ from database
                os.path.join(project_root, "data", "meshes", "813024", "02ec65f0.npz"),
                # Other common paths
                os.path.join(project_root, "data", "output", "pitch.glb"),
                os.path.join(project_root, "data", "output", "813024", "pitch.glb"),
            ]
            for c in candidates:
                if os.path.exists(c):
                    mesh_path = c
                    break

    if mesh_path and os.path.exists(mesh_path):
        print(f"Importing mesh: {mesh_path}")
        mesh_root = import_pitcher_mesh(mesh_path)
        if mesh_root:
            setup_animation(mesh_root)
            _install_foot_lock_handler()
            _install_shape_key_snap_handler()
            # Trigger handlers once so frame 0 is correct without scrubbing
            _foot_lock_handler(bpy.context.scene)
            _shape_key_snap_handler(bpy.context.scene)
            print("Mesh imported and positioned on mound.")
    else:
        print("\nNo mesh file specified or found.")
        print("Set MESH_PATH at the top of this script, then re-run.")
        print("")
        print("Or export from the database first:")
        print("  python scripts/export_mesh.py --play-id 02ec65f0 --format glb")
        print("")
        print("Supported formats: .glb (animated), .obj (single frame), .npz (raw)")
        print("The field is ready — just set MESH_PATH and re-run.\n")

    # Viewport settings — drop the user straight into the catcher cam view
    # so they don't have to hunt for the pitcher. Without this the default
    # viewport position can land inside the mesh and look like garbage.
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'MATERIAL'
                    space.clip_end = 200
                    # Switch the viewport to look through the active scene
                    # camera (Cam_Catcher), the equivalent of pressing
                    # Numpad 0 manually.
                    if bpy.context.scene.camera is not None:
                        space.region_3d.view_perspective = 'CAMERA'
                    break

    print("\n" + "-" * 60)
    print("  CONTROLS:")
    print("  Middle mouse: orbit")
    print("  Scroll: zoom")
    print("  Shift+middle mouse: pan")
    print("  Numpad 0: active camera view")
    print("  Ctrl+Numpad 0: set viewport as camera")
    print("")
    print("  CAMERA PRESETS (select in Outliner or Scene Properties):")
    print("  Cam_Catcher    — behind home plate")
    print("  Cam_Side_1B    — first base side (arm slot)")
    print("  Cam_Side_3B    — third base side")
    print("  Cam_Overhead   — top-down (landing alignment)")
    print("  Cam_Broadcast  — center field (TV angle)")
    print("  Cam_45deg      — 3/4 angle (arm path + stride)")
    print("  Cam_ArmSlot    — close-up release point")
    print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
