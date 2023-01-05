#!/bin/bash --login
# The --login ensures the bash configuration is loaded,
# enabling Conda.

# Enable strict mode.
set -euo pipefail
# ... Run whatever commands ...

# Temporarily disable strict mode and activate conda:
set +euo pipefail
conda activate myenv

# Re-enable strict mode:
set -euo pipefail
exec pip install transformers==4.19.2 scann kornia==0.6.4 torchmetrics==0.6.0
exec pip install git+https://github.com/arogozhnikov/einops.git

# exec the final command:
exec python main.py --base $1 -t --gpus 0, --logdir /storage/orhir/stable_logs/