"""SAM 3D Body inference wrapper — supports PyTorch/MPS and MLX backends."""

from __future__ import annotations

import math
import sys
from abc import ABC, abstractmethod

import numpy as np

from backend.app.models.pitch import PitchData, PitchMetadata

# SAM 3D Body PyTorch package lives outside the project tree.
_SAM3D_PATH = "/tmp/sam-3d-body"
if _SAM3D_PATH not in sys.path:
    sys.path.insert(0, _SAM3D_PATH)


class _BaseInference(ABC):
    """Common interface for SAM 3D Body backends."""

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> dict | None: ...

    def process_video_frames(
        self,
        frames: list[dict],
        metadata: PitchMetadata,
    ) -> PitchData:
        """Run inference over all frames and return a PitchData.

        Frames where detection fails get zero-filled placeholders so the
        (T, J, 3) tensors stay aligned with frame indices.
        """
        all_joints: list[np.ndarray] = []
        all_joints_mhr70: list[np.ndarray] = []
        first_pose: np.ndarray | None = None
        first_shape: np.ndarray | None = None

        for fd in frames:
            result = self.process_frame(fd["frame"])
            if result is None:
                all_joints.append(np.zeros((127, 3), np.float32))
                all_joints_mhr70.append(np.zeros((70, 3), np.float32))
            else:
                all_joints.append(result["joints"].astype(np.float32).reshape(127, 3))
                j_mhr = result["joints_mhr70"]
                if j_mhr.shape[-1] == 2:
                    j_mhr = np.concatenate(
                        [j_mhr, np.zeros((*j_mhr.shape[:-1], 1), np.float32)], axis=-1
                    )
                all_joints_mhr70.append(j_mhr.astype(np.float32).reshape(70, 3))
                if first_pose is None:
                    first_pose = result["pose_params"].astype(np.float32).reshape(136)
                    first_shape = result["shape_params"].astype(np.float32).reshape(45)

        joints_arr = np.stack(all_joints, axis=0)
        joints_mhr70_arr = np.stack(all_joints_mhr70, axis=0)
        pose_arr = first_pose if first_pose is not None else np.zeros((136,), np.float32)
        shape_arr = first_shape if first_shape is not None else np.zeros((45,), np.float32)

        pose_broadcast = np.tile(pose_arr[np.newaxis, :], (len(frames), 1))

        pitch_data = PitchData.from_numpy(
            metadata=metadata,
            joints=joints_arr,
            pose_params=pose_broadcast,
            shape_params=shape_arr,
            joints_mhr70=joints_mhr70_arr,
        )
        pitch_data.scale_to_height()
        return pitch_data


class SAM3DInferencePyTorch(_BaseInference):
    """PyTorch/MPS backend for SAM 3D Body inference."""

    def __init__(
        self,
        checkpoint_path: str = "/tmp/sam3d-weights/model.ckpt",
        mhr_path: str = "/tmp/sam3d-weights/assets/mhr_model.pt",
        det_threshold: float = 0.5,
        device: str = "mps",
    ) -> None:
        import torch

        self.checkpoint_path = checkpoint_path
        self.mhr_path = mhr_path
        self.det_threshold = det_threshold
        self.device = torch.device(device)

        self._estimator = None
        self._detector = None
        self._loaded = False

    def load(self) -> None:
        import torch
        import torchvision
        from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

        model, cfg = load_sam_3d_body(
            self.checkpoint_path, device=self.device, mhr_path=self.mhr_path,
        )
        self._estimator = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg)

        weights = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        self._detector = (
            torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
            .to(self.device)
            .eval()
        )
        self._loaded = True

    def process_frame(self, frame: np.ndarray) -> dict | None:
        import torch

        if not self._loaded:
            raise RuntimeError("Call load() before process_frame().")

        h, w = frame.shape[:2]

        img_tensor = (
            torch.from_numpy(frame.copy()).permute(2, 0, 1).float().to(self.device) / 255.0
        )
        with torch.no_grad():
            preds = self._detector([img_tensor])[0]

        person_mask = preds["labels"] == 1
        scores = preds["scores"][person_mask]
        boxes = preds["boxes"][person_mask]
        keep = scores > self.det_threshold
        boxes_np = boxes[keep].cpu().numpy()

        if len(boxes_np) == 0:
            bbox = np.array([[0, 0, w, h]], dtype=np.float32)
        else:
            areas = (boxes_np[:, 2] - boxes_np[:, 0]) * (boxes_np[:, 3] - boxes_np[:, 1])
            best = int(np.argmax(areas))
            bbox = boxes_np[best : best + 1]

        outputs = self._estimator.process_one_image(frame, bboxes=bbox, inference_type="body")
        if not outputs:
            return None

        person = outputs[0]
        kp_2d = person["pred_keypoints_2d"]
        if kp_2d.ndim == 2 and kp_2d.shape[1] > 2:
            kp_2d = kp_2d[:, :2]

        return {
            "vertices": person["pred_vertices"],
            "keypoints_2d": kp_2d,
            "joints": person.get("pred_joints", np.zeros((127, 3), np.float32)),
            "joints_mhr70": person.get("pred_keypoints", kp_2d),
            "pose_params": person.get("pred_pose", np.zeros((136,), np.float32)),
            "shape_params": person.get("pred_shape", np.zeros((45,), np.float32)),
            "cam_t": person["pred_cam_t"],
            "focal_length": float(person.get("focal_length", max(h, w))),
            "bbox": bbox[0],
        }


class SAM3DInferenceMLX(_BaseInference):
    """MLX (Apple Silicon) backend for SAM 3D Body inference."""

    def __init__(
        self,
        weights_dir: str = "/tmp/sam3d-mlx-weights/",
        det_threshold: float = 0.5,
    ) -> None:
        self.weights_dir = weights_dir
        self.det_threshold = det_threshold
        self._estimator = None
        self._loaded = False

    def load(self) -> None:
        from sam3d_mlx.estimator import SAM3DBodyEstimator, detect_persons_cached

        self._estimator = SAM3DBodyEstimator(self.weights_dir)
        # Warm up person detector
        detect_persons_cached(np.zeros((100, 100, 3), dtype=np.uint8))
        self._loaded = True

    def process_frame(self, frame: np.ndarray) -> dict | None:
        from sam3d_mlx.estimator import detect_persons_cached

        if not self._loaded:
            raise RuntimeError("Call load() before process_frame().")

        h, w = frame.shape[:2]

        detections = detect_persons_cached(frame, threshold=self.det_threshold)
        bbox = detections[0] if detections else [0, 0, w, h]

        result = self._estimator.predict(frame, bbox, auto_detect=False)

        focal = math.sqrt(h**2 + w**2)
        return {
            "vertices": result["pred_vertices"],
            "keypoints_2d": np.zeros((70, 2), np.float32),  # MLX doesn't project to 2D
            "joints": result["pred_joint_coords"],
            "joints_mhr70": result["pred_keypoints_3d"],
            "pose_params": result["pred_pose"],
            "shape_params": result["pred_shape"],
            "cam_t": result["pred_camera"],
            "focal_length": focal,
            "bbox": np.array(bbox, dtype=np.float32),
        }


# Backward-compatible alias
SAM3DInference = SAM3DInferencePyTorch


def create_inference(backend: str = "mlx", **kwargs) -> _BaseInference:
    """Factory: create an inference backend by name.

    Args:
        backend: "mlx" or "pytorch"
        **kwargs: passed to the backend constructor
    """
    if backend == "mlx":
        return SAM3DInferenceMLX(**kwargs)
    elif backend == "pytorch":
        return SAM3DInferencePyTorch(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend!r}")
