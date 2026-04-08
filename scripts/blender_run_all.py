"""
blender_run_all.py — run field viewer + analysis overlays + GLB export
in a single Blender session. Mostly a convenience wrapper for headless
pipeline testing.

Usage:
  blender --background --python scripts/blender_run_all.py
"""

import os
import sys
import runpy

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# Running the scripts via runpy preserves Blender's expected execution model
# (each runs as __main__ with its own globals), the same way Text Editor's
# Alt+P works. Using runpy instead of import lets each script's `if __name__
# == "__main__"` guard fire.

print("\n########## 1/3: FIELD VIEWER ##########")
runpy.run_path(os.path.join(_SCRIPT_DIR, "blender_field_viewer.py"), run_name="__main__")

print("\n########## 2/3: ANALYSIS OVERLAYS ##########")
runpy.run_path(os.path.join(_SCRIPT_DIR, "blender_analysis_overlays.py"), run_name="__main__")

print("\n########## 3/3: GLB EXPORT ##########")
runpy.run_path(os.path.join(_SCRIPT_DIR, "blender_export_glb.py"), run_name="__main__")

print("\n########## DONE ##########")
