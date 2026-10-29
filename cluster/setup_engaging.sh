#!/bin/bash
# One-time setup on the ORCD Engaging login node. Run from the repo root:
#     bash cluster/setup_engaging.sh
set -euo pipefail

# uv gives us the repo's pinned Python 3.11 env without depending on cluster modules
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv sync
uv pip install mplib  # shipped motion planner (Linux wheels only), for demos and start states

# SAPIEN needs a Vulkan device even for state-only simulation. GPU nodes often ship the NVIDIA
# driver without the Vulkan ICD file, so point the loader at the driver ourselves.
mkdir -p "$HOME/.config/vulkan"
cat > "$HOME/.config/vulkan/nvidia_icd.json" <<'EOF'
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version": "1.2.155"
    }
}
EOF

mkdir -p logs results data
echo "Done. Next: sbatch cluster/smoke.sbatch"
