"""Multi-pitch comparison GLB builder — two meshes, one file."""
import os
import numpy as np

try:
    import pygltflib
    from pygltflib import (
        GLTF2, Scene, Node, Mesh, Primitive, Accessor, BufferView, Buffer, Asset, Material,
    )
    HAS_GLTF = True
except ImportError:
    HAS_GLTF = False


class ComparisonGLBBuilder:
    """Builds a single GLB containing two phase-aligned pitch meshes."""

    def phase_align(
        self,
        frames1: list[np.ndarray],
        frames2: list[np.ndarray],
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Stretch the shorter sequence to match the longer via linear interpolation.
        Returns (aligned1, aligned2) both with equal length.
        """
        n1, n2 = len(frames1), len(frames2)
        if n1 == n2:
            return frames1, frames2

        target = max(n1, n2)

        def resample(frames: list[np.ndarray], target_len: int) -> list[np.ndarray]:
            src_len = len(frames)
            result = []
            for i in range(target_len):
                t = i * (src_len - 1) / (target_len - 1)
                lo = int(t)
                hi = min(lo + 1, src_len - 1)
                alpha = t - lo
                interp = (1 - alpha) * np.asarray(frames[lo]) + alpha * np.asarray(frames[hi])
                result.append(interp.astype(np.float32))
            return result

        if n1 < target:
            frames1 = resample(frames1, target)
        else:
            frames2 = resample(frames2, target)

        return frames1, frames2

    def build(
        self,
        pitch1_frames: list[np.ndarray],
        pitch2_frames: list[np.ndarray],
        metadata1: dict,
        metadata2: dict,
        output_path: str,
    ) -> str:
        """
        Build a comparison GLB with two meshes.

        Second mesh uses alphaMode=BLEND, alpha=0.3.
        Both metadata dicts stored in extras.
        Returns output_path.
        """
        if not HAS_GLTF:
            raise ImportError("pygltflib required: pip install pygltflib")

        frames1, frames2 = self.phase_align(pitch1_frames, pitch2_frames)

        gltf = GLTF2()
        gltf.asset = Asset(version="2.0", generator="SamPlaysBaseball ComparisonGLBBuilder")
        gltf.extras = {
            "pitch1": metadata1,
            "pitch2": metadata2,
        }

        blob_parts = []
        accessors_start = [0]  # track buffer offsets

        def append_mesh_frames(frames: list[np.ndarray]) -> int:
            """Add a mesh (base + morph targets) to gltf, return mesh index."""
            frames = [np.asarray(f, dtype=np.float32) for f in frames]
            n_verts = frames[0].shape[0]
            stride = n_verts * 3 * 4

            base_offset = sum(len(p) for p in blob_parts)
            blob_parts.append(frames[0].tobytes())

            bv_base = BufferView(buffer=0, byteOffset=base_offset, byteLength=stride)
            gltf.bufferViews.append(bv_base)
            bv_base_idx = len(gltf.bufferViews) - 1

            acc_base = Accessor(
                bufferView=bv_base_idx, componentType=5126,
                count=n_verts, type="VEC3",
            )
            gltf.accessors.append(acc_base)
            acc_base_idx = len(gltf.accessors) - 1

            morph_targets = []
            for frame in frames[1:]:
                delta = (frame - frames[0]).astype(np.float32)
                offset = sum(len(p) for p in blob_parts)
                blob_parts.append(delta.tobytes())
                bv_mt = BufferView(buffer=0, byteOffset=offset, byteLength=stride)
                gltf.bufferViews.append(bv_mt)
                bv_mt_idx = len(gltf.bufferViews) - 1
                acc_mt = Accessor(
                    bufferView=bv_mt_idx, componentType=5126,
                    count=n_verts, type="VEC3",
                )
                gltf.accessors.append(acc_mt)
                morph_targets.append({"POSITION": len(gltf.accessors) - 1})

            prim = Primitive(
                attributes=pygltflib.Attributes(POSITION=acc_base_idx),
                targets=morph_targets,
            )
            mesh = Mesh(primitives=[prim])
            if len(frames) > 1:
                mesh.weights = [0.0] * len(morph_targets)
                mesh.weights[0] = 1.0
            gltf.meshes.append(mesh)
            return len(gltf.meshes) - 1

        mesh1_idx = append_mesh_frames(frames1)
        mesh2_idx = append_mesh_frames(frames2)

        # Ghost material for second mesh
        ghost_mat = Material(
            name="ghost",
            alphaMode="BLEND",
            doubleSided=True,
        )
        if hasattr(ghost_mat, 'pbrMetallicRoughness'):
            ghost_mat.pbrMetallicRoughness = pygltflib.PbrMetallicRoughness(
                baseColorFactor=[0.27, 0.53, 1.0, 0.3],
            )
        gltf.materials.append(ghost_mat)
        gltf.meshes[mesh2_idx].primitives[0].material = len(gltf.materials) - 1

        node1 = Node(mesh=mesh1_idx)
        node2 = Node(mesh=mesh2_idx)
        gltf.nodes.extend([node1, node2])

        scene = Scene(nodes=[0, 1])
        gltf.scenes.append(scene)
        gltf.scene = 0

        full_blob = b"".join(blob_parts)
        gltf.buffers.append(Buffer(byteLength=len(full_blob)))
        gltf.set_binary_blob(full_blob)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        gltf.save_binary(output_path)
        return output_path
