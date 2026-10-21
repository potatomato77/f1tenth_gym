#!/usr/bin/env bash
# Create or update a server checkout and install the SAC training stack.
#
# Usage:
#   bash scripts/setup_server.sh [target-dir]
#
# Environment overrides:
#   REPO_URL          default: https://github.com/potatomato77/f1tenth_gym.git
#   BRANCH            default: study/sac-residual
#   PYTHON            default: python3
#   TORCH_INDEX_URL   optional, e.g. https://download.pytorch.org/whl/cpu
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/potatomato77/f1tenth_gym.git}"
BRANCH="${BRANCH:-study/sac-residual}"
TARGET="${1:-f1tenth-residual-sac}"
PYTHON="${PYTHON:-python3}"

if [ "$#" -eq 0 ] && [ -d .git ] && [ -f SAC_project/train.py ]; then
    TARGET="."
    echo "Using the current checkout at $PWD"
elif [ -d "$TARGET/.git" ]; then
    echo "Updating $TARGET"
    git -C "$TARGET" pull --ff-only
elif [ -e "$TARGET" ]; then
    echo "error: $TARGET exists but is not a git checkout" >&2
    exit 1
else
    echo "Cloning $BRANCH from $REPO_URL"
    git clone --depth 1 --single-branch --branch "$BRANCH" "$REPO_URL" "$TARGET"
fi

cd "$TARGET"

if [ ! -d .venv ]; then
    "$PYTHON" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip

if [ -n "${TORCH_INDEX_URL:-}" ]; then
    python -m pip install "torch>=2.8,<2.11" --index-url "$TORCH_INDEX_URL"
fi

python -m pip install -e .
python -m pip install -r SAC_project/requirements.txt
python -m pip install -r SAC_project/requirements-rl.txt

cat <<'EOF'

Environment ready. From the repository root:

  source .venv/bin/activate
  python -m SAC_project.train --smoke
  python -m SAC_project.train --total-timesteps 500000 --n-envs 8 \
      --vec-env subproc --checkpoint-freq 50000 --device auto

Notes:
- Run commands from the repository root, not from SAC_project/.
- maps/ is not tracked by git. Place the required track files under maps/ or
  let the loader download them on first use.
- CPU-only server:
    TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu bash scripts/setup_server.sh
- CUDA 12.8 server:
    TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 bash scripts/setup_server.sh
EOF
