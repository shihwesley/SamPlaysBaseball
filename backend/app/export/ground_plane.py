"""Ground-plane alignment using PCA on ankle trajectory."""
import numpy as np

MHR70_ANKLE_LEFT = 13
MHR70_ANKLE_RIGHT = 14
MOUND_HEIGHT_M = 0.2667  # 10.5 inches in meters


class GroundPlaneAligner:
    """Aligns a joint sequence so the ground plane is Y=0 and up is +Y."""

    def align(self, joint_sequence: list[np.ndarray]) -> np.ndarray:
        """
        Compute a 4x4 rigid transform from ankle trajectory.

        joint_sequence: list of (70, 3) float32 arrays
        Returns 4x4 transform: rotate to Y-up, translate so ground at y=0
        with mound height offset applied.
        """
        # Collect ankle positions across all frames
        ankles = []
        for frame in joint_sequence:
            frame = np.asarray(frame)
            if frame.ndim == 2 and frame.shape[0] > max(MHR70_ANKLE_LEFT, MHR70_ANKLE_RIGHT):
                ankles.append(frame[MHR70_ANKLE_LEFT])
                ankles.append(frame[MHR70_ANKLE_RIGHT])

        if len(ankles) < 3:
            return np.eye(4, dtype=np.float32)

        ankle_pts = np.stack(ankles, axis=0).astype(np.float64)
        centroid = ankle_pts.mean(axis=0)
        centered = ankle_pts - centroid

        # SVD to find ground normal (smallest singular value direction)
        _, s, Vt = np.linalg.svd(centered, full_matrices=False)
        ground_normal = Vt[-1]  # least-variance direction = ground normal
        ground_normal = ground_normal / (np.linalg.norm(ground_normal) + 1e-8)

        # Build rotation so ground_normal maps to +Y
        up = np.array([0.0, 1.0, 0.0])
        v = np.cross(ground_normal, up)
        c = np.dot(ground_normal, up)
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        if abs(c + 1.0) < 1e-8:
            R = -np.eye(3)
        else:
            R = np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))

        # Build 4x4 transform
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R.astype(np.float32)
        # Translate: put centroid at mound height
        T[:3, 3] = (R @ (-centroid)).astype(np.float32)
        T[1, 3] += MOUND_HEIGHT_M

        return T

    def apply_transform(self, joint_sequence: list[np.ndarray], transform: np.ndarray) -> list[np.ndarray]:
        """Apply a 4x4 rigid transform to every frame in the sequence."""
        R = transform[:3, :3]
        t = transform[:3, 3]
        result = []
        for frame in joint_sequence:
            frame = np.asarray(frame, dtype=np.float32)
            transformed = (frame @ R.T) + t
            result.append(transformed.astype(np.float32))
        return result
