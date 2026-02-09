#!/usr/bin/env bash
set -euo pipefail

# Patch PaddleX set_optimization_level bug (paddlepaddle-gpu 2.6.x compat)
STATIC_INFER=".venv/lib/python3.12/site-packages/paddlex/inference/models/common/static_infer.py"
if grep -q 'config.set_optimization_level' "$STATIC_INFER" 2>/dev/null; then
    sed -i 's/config.set_optimization_level(3)/pass  # config.set_optimization_level(3)/' "$STATIC_INFER"
    echo "Patched set_optimization_level in $STATIC_INFER"
fi

export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

exec uvicorn main:app --host 0.0.0.0 --port 8000
