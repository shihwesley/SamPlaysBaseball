"""Launch the SamPlaysBaseball demo stack.

Starts the FastAPI backend and Next.js frontend in parallel.
Press Ctrl+C to stop both.

Usage:
    python demo/launcher.py

Environment variables:
    DEMO_DATA_DIR   Path to demo data directory (default: ./demo/data)
    API_PORT        Backend port (default: 8000)
    FRONTEND_PORT   Frontend port (default: 3000)
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DEMO_DATA_DIR = Path(os.environ.get("DEMO_DATA_DIR", REPO_ROOT / "demo" / "data"))
API_PORT = int(os.environ.get("API_PORT", "8000"))
FRONTEND_PORT = int(os.environ.get("FRONTEND_PORT", "3000"))


def check_demo_data() -> None:
    """Warn if synthetic data hasn't been generated yet."""
    if not DEMO_DATA_DIR.exists() or not any(DEMO_DATA_DIR.iterdir()):
        print("WARNING: Demo data directory is empty or missing.")
        print(f"  Expected: {DEMO_DATA_DIR}")
        print("  Run: python demo/generate_synthetic.py")
        print("  Then restart the launcher.")
        print()


def start_api() -> subprocess.Popen:
    """Start the FastAPI server."""
    env = os.environ.copy()
    env["DEMO_DATA_DIR"] = str(DEMO_DATA_DIR)

    cmd = [
        sys.executable, "-m", "uvicorn",
        "backend.app.main:app",
        "--host", "0.0.0.0",
        "--port", str(API_PORT),
        "--reload",
    ]
    print(f"  Starting API on http://localhost:{API_PORT}")
    return subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env)


def start_frontend() -> subprocess.Popen:
    """Start the Next.js dev server."""
    frontend_dir = REPO_ROOT / "frontend"
    if not frontend_dir.exists():
        print("  WARNING: frontend/ directory not found. Skipping frontend.")
        return None

    env = os.environ.copy()
    env["NEXT_PUBLIC_DEMO_MODE"] = "true"
    env["NEXT_PUBLIC_API_URL"] = f"http://localhost:{API_PORT}"
    env["PORT"] = str(FRONTEND_PORT)

    # Prefer pnpm, fall back to npm
    pkg_manager = "pnpm" if _which("pnpm") else "npm"
    cmd = [pkg_manager, "run", "dev"]

    print(f"  Starting frontend on http://localhost:{FRONTEND_PORT}")
    return subprocess.Popen(cmd, cwd=str(frontend_dir), env=env)


def _which(name: str) -> bool:
    import shutil
    return shutil.which(name) is not None


def main() -> None:
    print("SamPlaysBaseball Demo Launcher")
    print("=" * 40)
    check_demo_data()

    processes: list[subprocess.Popen] = []

    def shutdown(sig, frame):
        print("\nShutting down...")
        for proc in processes:
            if proc and proc.poll() is None:
                proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    api_proc = start_api()
    processes.append(api_proc)

    # Give the API a moment to start before launching frontend
    time.sleep(2)

    frontend_proc = start_frontend()
    if frontend_proc:
        processes.append(frontend_proc)

    print()
    print("Demo running:")
    print(f"  API:      http://localhost:{API_PORT}/docs")
    print(f"  Frontend: http://localhost:{FRONTEND_PORT}")
    print()
    print("Press Ctrl+C to stop.")

    # Wait for processes to exit
    try:
        while True:
            for proc in processes:
                if proc and proc.poll() is not None:
                    print(f"Process exited with code {proc.returncode}. Stopping all.")
                    shutdown(None, None)
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown(None, None)


if __name__ == "__main__":
    main()
