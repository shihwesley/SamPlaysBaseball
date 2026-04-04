"""GLB export for pitcher mesh sequences using morph targets (shape keys)."""
import json
import struct
import os
import numpy as np

try:
    import pygltflib
    from pygltflib import (
        GLTF2, Scene, Node, Mesh, Primitive, Accessor, BufferView, Buffer,
        Asset, Material, MeshPrimitive, Animation,
    )
    HAS_PYGLTFLIB = True
except ImportError:
    HAS_PYGLTFLIB = False


def _require_pygltflib() -> None:
    if not HAS_PYGLTFLIB:
        raise ImportError("pygltflib required: pip install pygltflib")


# meshopt quantization applied via gltfpack post-process
# Target file size: <5MB per pitch with vertex quantization enabled


class GLBExporter:
    """
    Exports mesh frame sequences as animated GLB using morph targets.

    Each pitch is stored as a GLTF2 asset where:
      - Base mesh = first frame positions
      - Remaining frames = morph targets (shape keys)
      - Metadata stored in extras: pitch_type, pitcher_id, frame_timestamps, phase_markers
    """

    def __init__(self, vertex_count: int = 18439, face_count: int = 36874) -> None:
        self.vertex_count = vertex_count
        self.face_count = face_count

    def export_pitch(
        self,
        mesh_frames: list[np.ndarray],
        metadata: dict,
        output_path: str,
    ) -> str:
        """
        Export a sequence of mesh frames as an animated GLB.

        mesh_frames: list of (N, 3) float32 arrays, one per frame
        metadata: dict with pitch_type, pitcher_id, frame_timestamps, phase_markers
        output_path: destination .glb path

        Returns output_path.
        """
        _require_pygltflib()

        if not mesh_frames:
            raise ValueError("mesh_frames must not be empty")

        frames = [np.asarray(f, dtype=np.float32) for f in mesh_frames]
        n_verts = frames[0].shape[0]
        n_frames = len(frames)

        gltf = GLTF2()
        gltf.asset = Asset(version="2.0", generator="SamPlaysBaseball GLBExporter")
        gltf.extras = {
            "pitch_type": metadata.get("pitch_type", ""),
            "pitcher_id": metadata.get("pitcher_id", ""),
            "frame_timestamps": metadata.get("frame_timestamps", []),
            "phase_markers": metadata.get("phase_markers", {}),
        }

        # Flatten all frame data into a single byte buffer
        blob_parts = []

        # Base positions (frame 0)
        base_positions = frames[0].astype(np.float32)
        blob_parts.append(base_positions.tobytes())

        # Morph target deltas (frames 1..N-1)
        morph_deltas = []
        for i in range(1, n_frames):
            delta = (frames[i] - frames[0]).astype(np.float32)
            morph_deltas.append(delta)
            blob_parts.append(delta.tobytes())

        full_blob = b"".join(blob_parts)

        buf = Buffer(byteLength=len(full_blob))
        gltf.buffers.append(buf)

        def add_buffer_view(byte_offset: int, byte_length: int) -> int:
            bv = BufferView(buffer=0, byteOffset=byte_offset, byteLength=byte_length)
            gltf.bufferViews.append(bv)
            return len(gltf.bufferViews) - 1

        def add_accessor(buffer_view_idx: int, count: int, component_type: int = 5126) -> int:
            acc = Accessor(
                bufferView=buffer_view_idx,
                componentType=component_type,
                count=count,
                type="VEC3",
            )
            gltf.accessors.append(acc)
            return len(gltf.accessors) - 1

        stride = n_verts * 3 * 4  # float32 * 3 components * 4 bytes

        # Base positions accessor
        bv_base = add_buffer_view(0, stride)
        acc_base = add_accessor(bv_base, n_verts)

        # Morph target accessors
        morph_targets = []
        for i, _delta in enumerate(morph_deltas):
            offset = stride * (i + 1)
            bv_mt = add_buffer_view(offset, stride)
            acc_mt = add_accessor(bv_mt, n_verts)
            morph_targets.append({"POSITION": acc_mt})

        prim = Primitive(
            attributes=pygltflib.Attributes(POSITION=acc_base),
            targets=morph_targets,
        )
        mesh = Mesh(primitives=[prim])
        if n_frames > 1:
            mesh.weights = [0.0] * len(morph_deltas)
            mesh.weights[0] = 1.0
        gltf.meshes.append(mesh)

        node = Node(mesh=0)
        gltf.nodes.append(node)

        scene = Scene(nodes=[0])
        gltf.scenes.append(scene)
        gltf.scene = 0

        gltf.set_binary_blob(full_blob)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        gltf.save_binary(output_path)
        return output_path

    # Alias
    export_sequence = export_pitch


if __name__ == "__main__":
    # Quick smoke test
    exporter = GLBExporter(vertex_count=100, face_count=50)
    frames = [np.random.randn(100, 3).astype(np.float32) for _ in range(5)]
    meta = {
        "pitch_type": "FF",
        "pitcher_id": "test-001",
        "frame_timestamps": list(range(5)),
        "phase_markers": {"stride": 1, "release": 3},
    }
    try:
        out = exporter.export_pitch(frames, meta, "/tmp/test_pitch.glb")
        print("Exported to", out)
    except ImportError as e:
        print("Skipped (pygltflib not installed):", e)
