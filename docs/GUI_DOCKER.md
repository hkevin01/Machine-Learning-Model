---
title: Dockerized PyQt6 GUI
---

## Dockerized PyQt6 GUI

Launch the GUI inside a reproducible container with X11 forwarding.

## Prerequisites

- Docker & Docker Compose v2
- Running X server (native Linux desktop, XQuartz on macOS, VcXsrv/Xming on Windows)

## Quick Start (Linux)

```bash
chmod +x scripts/run_gui_docker.sh  # first time
./scripts/run_gui_docker.sh
```

The PyQt6 window from `run_gui.py` should appear.

## macOS (XQuartz)

```bash
open -a XQuartz
# Enable "Allow connections from network clients" in XQuartz Preferences (Security)
export DISPLAY=host.docker.internal:0
xhost + 127.0.0.1
docker compose up --build ml-gui
```

## Windows (VcXsrv)

1. Start VcXsrv allowing public access (or configure access control appropriately).
2. Determine host IP (e.g. 192.168.1.50) then:

```powershell
set DISPLAY=192.168.1.50:0.0
docker compose up --build ml-gui
```

## Makefile Convenience

After adding the `gui` target:

```bash
make gui
```

## Notes

- `QT_X11_NO_MITSHM=1` mitigates shared memory extension issues in some Docker/X11 combos.
- To add Agent Mode in-container, add another service referencing `./run_agent.sh`.
- For GPU acceleration (Linux + NVIDIA), add the appropriate `--gpus all` runtime or compose device reservations.
