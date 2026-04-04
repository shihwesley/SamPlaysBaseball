"""CLI entry point for SAM 3D Body MLX inference.

Usage:
    python -m sam3d_mlx --image photo.jpg --output mesh.obj
    python -m sam3d_mlx --image photo.jpg --bbox 100,50,400,500 --output mesh.obj
    python -m sam3d_mlx --image photo.jpg --output mesh.obj --weights /path/to/weights/
"""

import argparse
import sys
import time


def main():
    parser = argparse.ArgumentParser(
        description="SAM 3D Body estimation (MLX)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--image", required=True,
        help="Path to input image (RGB, any format supported by PIL/cv2)",
    )
    parser.add_argument(
        "--bbox",
        help="Bounding box as x1,y1,x2,y2 (comma-separated). Defaults to full image.",
    )
    parser.add_argument(
        "--output", "-o", default="output.obj",
        help="Output .obj mesh path (default: output.obj)",
    )
    parser.add_argument(
        "--weights", default="/tmp/sam3d-mlx-weights/",
        help="Path to weights directory (default: /tmp/sam3d-mlx-weights/)",
    )
    parser.add_argument(
        "--save-keypoints", action="store_true",
        help="Also save 3D keypoints to a .npy file alongside the mesh",
    )
    args = parser.parse_args()

    # Load image
    image = _load_image(args.image)
    if image is None:
        print(f"Error: could not load image '{args.image}'", file=sys.stderr)
        sys.exit(1)

    # Parse bbox
    bbox = None
    if args.bbox:
        try:
            bbox = [float(x) for x in args.bbox.split(",")]
            if len(bbox) != 4:
                raise ValueError
        except ValueError:
            print("Error: bbox must be 4 comma-separated numbers (x1,y1,x2,y2)", file=sys.stderr)
            sys.exit(1)

    # Load model
    print(f"Loading model from {args.weights}...")
    t0 = time.time()

    from .estimator import SAM3DBodyEstimator, write_obj
    estimator = SAM3DBodyEstimator(args.weights)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Run inference
    print(f"Running inference on {args.image}...")
    t0 = time.time()
    result = estimator.predict(image, bbox)
    print(f"Inference completed in {time.time() - t0:.1f}s")

    # Get faces from the model
    import numpy as np
    faces = np.array(estimator.model.head_pose.faces)

    # Write mesh
    write_obj(result["pred_vertices"], faces, args.output)
    print(f"Mesh written to {args.output}")
    print(f"  Vertices: {result['pred_vertices'].shape[0]}")
    print(f"  Faces: {faces.shape[0]}")

    # Optionally save keypoints
    if args.save_keypoints:
        kp_path = args.output.rsplit(".", 1)[0] + "_keypoints.npy"
        np.save(kp_path, result["pred_keypoints_3d"])
        print(f"Keypoints saved to {kp_path}")

    # Print camera params
    cam = result["pred_camera"]
    print(f"  Camera: scale={cam[0]:.4f}, tx={cam[1]:.4f}, ty={cam[2]:.4f}")


def _load_image(path: str):
    """Load image as RGB uint8 numpy array. Tries PIL first, then cv2."""
    import numpy as np

    try:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        return np.array(img)
    except ImportError:
        pass

    try:
        import cv2
        img = cv2.imread(path)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except ImportError:
        pass

    print("Error: install Pillow or OpenCV to load images", file=sys.stderr)
    return None


if __name__ == "__main__":
    main()
