#!/usr/bin/env bash
# Unified launcher: builds (if needed) and runs the PyQt6 GUI inside Docker.
# Usage:
#   ./run.sh                # build if missing, run GUI
#   ./run.sh --rebuild       # force rebuild image
#   ./run.sh --gpu           # attempt GPU (NVIDIA) passthrough
#   ./run.sh --headless      # run a smoke test (no X) to validate imports
#   ./run.sh --help          # show help
#   ENV VARS:
#     ML_GUI_IMAGE (default: ml-gui:latest)
#     DISPLAY (override X display)
#
set -euo pipefail

IMAGE_DEFAULT="ml-gui:latest"
IMAGE="${ML_GUI_IMAGE:-$IMAGE_DEFAULT}"
NEED_REBUILD=0
USE_GPU=0
HEADLESS=0
EXTRA_DOCKER_ARGS=()

print_help() {
  cat <<EOF
Machine Learning Model - Docker GUI Runner

Options:
  --rebuild        Force rebuild the GUI image
  --gpu            Enable GPU (NVIDIA) passthrough (--gpus all)
  --headless       Run a headless import/self-test instead of launching the window
  --image <name>   Override image tag (default: ${IMAGE})
  --help           Show this help

Environment:
  ML_GUI_IMAGE     Image tag (default: ${IMAGE_DEFAULT})
  DISPLAY          X11 display (auto-detected if absent)

Examples:
  ./run.sh
  ./run.sh --rebuild --gpu
  ML_GUI_IMAGE=myorg/ml-gui:dev ./run.sh --rebuild
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rebuild) NEED_REBUILD=1; shift ;;
    --gpu) USE_GPU=1; shift ;;
    --headless) HEADLESS=1; shift ;;
    --image) IMAGE="$2"; shift 2 ;;
    --help|-h) print_help; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; print_help; exit 1 ;;
  esac
done

# Detect platform
UNAME_S=$(uname -s || echo unknown)

# Setup DISPLAY if missing (Linux). For macOS user must set DISPLAY via XQuartz.
if [[ "${UNAME_S}" == "Linux" ]]; then
  if [[ -z "${DISPLAY:-}" ]]; then
    export DISPLAY=:0
  fi
fi

# X server access (Linux). Ignore errors if xhost unavailable.
if command -v xhost >/dev/null 2>&1 && [[ $HEADLESS -eq 0 ]]; then
  xhost +SI:localuser:root >/dev/null 2>&1 || true
  xhost +local:docker >/dev/null 2>&1 || true
fi

# Build image if forced or missing
if [[ $NEED_REBUILD -eq 1 ]] || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "[build] Building image: $IMAGE"
  docker build -f Dockerfile.gui -t "$IMAGE" .
else
  echo "[info] Using existing image: $IMAGE"
fi

# GPU support
if [[ $USE_GPU -eq 1 ]]; then
  EXTRA_DOCKER_ARGS+=(--gpus all)
fi

# Headless smoke test
if [[ $HEADLESS -eq 1 ]]; then
  echo "[run] Headless import test..."
  docker run --rm "${EXTRA_DOCKER_ARGS[@]}" "$IMAGE" \
    python -c "from PyQt6.QtWidgets import QApplication; import sys; a=QApplication(sys.argv); print('GUI stack OK')"
  exit 0
fi

# Volume mounts
VOLUMES=(-v "$(pwd)":/app:rw)

# X11 mount if not headless
if [[ $HEADLESS -eq 0 ]]; then
  if [[ -d /tmp/.X11-unix ]]; then
    VOLUMES+=( -v /tmp/.X11-unix:/tmp/.X11-unix:ro )
  fi
fi

# Wayland hint: force xcb if WAYLAND_DISPLAY present
if [[ -n "${WAYLAND_DISPLAY:-}" ]]; then
  EXTRA_DOCKER_ARGS+=( -e QT_QPA_PLATFORM=xcb )
fi

# Shared memory size for more stable Qt (avoid MIT-SHM issues)
EXTRA_DOCKER_ARGS+=( --shm-size=512m )

# Run container
echo "[run] Launching GUI container..."
set -x
docker run --rm \
  --name ml-gui-run \
  -e DISPLAY="$DISPLAY" \
  -e QT_X11_NO_MITSHM=1 \
  "${EXTRA_DOCKER_ARGS[@]}" \
  "${VOLUMES[@]}" \
  "$IMAGE"
set +x

echo "[done] GUI container exited."
