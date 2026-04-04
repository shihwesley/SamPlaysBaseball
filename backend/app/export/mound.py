"""MLB regulation mound geometry generator."""
import numpy as np

MOUND_DIAMETER_FT = 18.0
MOUND_RISE_IN = 10.5
MOUND_SLOPE_IN = 6.0
MOUND_SLOPE_FT = 6.0

# Conversion
FT_TO_M = 0.3048
IN_TO_M = 0.0254

MOUND_RADIUS_M = (MOUND_DIAMETER_FT / 2) * FT_TO_M
MOUND_RISE_M = MOUND_RISE_IN * IN_TO_M
MOUND_SLOPE_RISE_M = MOUND_SLOPE_IN * IN_TO_M
MOUND_SLOPE_DIST_M = MOUND_SLOPE_FT * FT_TO_M


class MLBMound:
    """Generates triangulated mound mesh geometry."""

    def generate_mesh(self, rings: int = 24, segments: int = 48) -> dict:
        """
        Generate a triangulated disc mesh representing an MLB pitcher's mound.

        Returns dict with:
          'vertices': np.float32 (N, 3)
          'faces': np.int32 (M, 3)
        """
        vertices = []
        faces = []

        # Center peak
        vertices.append([0.0, MOUND_RISE_M, 0.0])

        # Build concentric rings from center outward
        radii = np.linspace(0, MOUND_RADIUS_M, rings + 1)[1:]  # skip 0 (center)

        for ring_idx, r in enumerate(radii):
            height = self._height_at_radius(r)
            for seg in range(segments):
                angle = 2 * np.pi * seg / segments
                x = r * np.cos(angle)
                z = r * np.sin(angle)
                vertices.append([x, height, z])

        vertices = np.array(vertices, dtype=np.float32)

        # Build faces
        # Fan triangles from center to first ring
        first_ring_start = 1
        for seg in range(segments):
            a = 0  # center
            b = first_ring_start + seg
            c = first_ring_start + (seg + 1) % segments
            faces.append([a, b, c])

        # Quad strips between rings
        for ring_idx in range(rings - 1):
            ring_a_start = 1 + ring_idx * segments
            ring_b_start = 1 + (ring_idx + 1) * segments
            for seg in range(segments):
                a = ring_a_start + seg
                b = ring_a_start + (seg + 1) % segments
                c = ring_b_start + (seg + 1) % segments
                d = ring_b_start + seg
                faces.append([a, b, c])
                faces.append([a, c, d])

        faces = np.array(faces, dtype=np.int32)
        return {"vertices": vertices, "faces": faces}

    def _height_at_radius(self, r: float) -> float:
        """Compute mound height at radial distance r from center."""
        # At center: MOUND_RISE_M
        # Slopes down toward plate side; use dome approximation
        ratio = r / MOUND_RADIUS_M
        # Smooth cosine profile
        return float(MOUND_RISE_M * 0.5 * (1 + np.cos(np.pi * ratio)))

    def generate_plate_direction(self) -> np.ndarray:
        """Unit vector pointing toward home plate (negative Z)."""
        return np.array([0.0, 0.0, -1.0], dtype=np.float32)
