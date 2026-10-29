# Sourced by every job script (submitted from the repo root).
source "$SLURM_SUBMIT_DIR/.venv/bin/activate"
# Use the system ICD if the node has one, else the one setup_engaging.sh wrote.
if [ ! -f /usr/share/vulkan/icd.d/nvidia_icd.json ] && [ ! -f /etc/vulkan/icd.d/nvidia_icd.json ]; then
    export VK_ICD_FILENAMES="$HOME/.config/vulkan/nvidia_icd.json"
fi
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"
