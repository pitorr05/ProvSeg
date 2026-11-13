#!/usr/bin/env bash
# One-time server setup: install deps, download public datasets, build word-level data and
# dependency graphs. Run from the cloned ProvSeg/ directory:   bash setup_server.sh
# Layout created next to the repo:  ../data/hf  ../data/processed  ../data/feats  (../refs for PROFILE=full)
set -euo pipefail
PROFILE=${PROFILE:-small}
SKIP_PIP=${SKIP_PIP:-0}
cd "$(dirname "$0")"

if [ "$SKIP_PIP" != 1 ]; then
  pip install -r requirements.txt
  python -m spacy download en_core_web_sm
fi
mkdir -p ../data/hf ../data/processed ../data/feats

echo "==> downloading public datasets from Hugging Face"
PROFILE=$PROFILE python - <<'PY'
import os, zipfile
from huggingface_hub import snapshot_download
hf = "../data/hf"
full = os.environ.get("PROFILE") == "full"
snapshot_download("toloka/beemo", repo_type="dataset", local_dir=f"{hf}/toloka__beemo")
snapshot_download("d0rj/SemEval2024-task8", repo_type="dataset", local_dir=f"{hf}/d0rj__SemEval2024-task8",
                  allow_patterns=["subtaskC/*", "README.md"])
p = snapshot_download("linzw/PASTED", repo_type="dataset", local_dir=f"{hf}/linzw__PASTED",
                      allow_patterns=["classification.zip", "eval-OOD.zip", "README.md"])
for z in ("classification", "eval-OOD"):
    zipfile.ZipFile(f"{p}/{z}.zip").extractall(f"{p}/x/{z}")
if full:
    snapshot_download("smksaha/apt-eval", repo_type="dataset", local_dir=f"{hf}/smksaha__apt-eval")
print("datasets ok")
PY

if [ "$PROFILE" = full ]; then
  mkdir -p ../refs
  [ -d ../refs/SeqXGPT ]  || git clone --depth 1 https://github.com/Jihuai-wpy/SeqXGPT.git ../refs/SeqXGPT
  [ -d ../refs/EditLens ] || git clone --depth 1 https://github.com/pangramlabs/EditLens.git ../refs/EditLens
  SETS=""
else
  SETS="--only semeval8c,pasted,beemo"
fi

echo "==> building word-level provenance data"
python scripts/build_data.py --hf ../data/hf --refs ../refs --out ../data/processed $SETS
echo "==> building dependency graphs (CPU)"
python -m provseg.features.syntax_graph --data ../data/processed --out ../data/feats $SETS

python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| GPUs:', torch.cuda.device_count())"
cat <<'EOF'

Setup done. Next:
  LIMIT=50 bash run_all.sh            # ~15 min environment test (results in ../results_test)
  bash run_all.sh                     # round 1: 3 benchmarks x 5 baselines + ProvSeg
  cat ../results_small/table_main.md  # main table
EOF
