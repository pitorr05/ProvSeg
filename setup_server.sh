#!/usr/bin/env bash
# One-time server setup: install deps, download public datasets, build word-level data.
# Run from the cloned ProvSeg/ directory:   bash setup_server.sh
# Layout created next to the repo:  ../data/hf  ../data/processed  ../data/feats  ../refs
set -euo pipefail
PROFILE=${PROFILE:-small}
SKIP_PIP=${SKIP_PIP:-0}
cd "$(dirname "$0")"

if [ "$SKIP_PIP" != 1 ]; then
  pip install -r requirements.txt
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
import urllib.request
cz = f"{hf}/coauthor/coauthor.zip"
if not os.path.exists(f"{hf}/coauthor/x/coauthor-v1.0"):
    os.makedirs(f"{hf}/coauthor", exist_ok=True)
    urllib.request.urlretrieve("https://cs.stanford.edu/~minalee/zip/chi2022-coauthor-v1.0.zip", cz)
    zipfile.ZipFile(cz).extractall(f"{hf}/coauthor/x")
print("datasets ok")
PY

mkdir -p ../refs
# SeqXGPT-Bench ships with the official SeqXGPT repository (base paper of the ladder)
[ -d ../refs/SeqXGPT ] || git clone --depth 1 https://github.com/Jihuai-wpy/SeqXGPT.git ../refs/SeqXGPT
if [ "$PROFILE" = full ]; then
  [ -d ../refs/EditLens ] || git clone --depth 1 https://github.com/pangramlabs/EditLens.git ../refs/EditLens
  SETS=""
else
  SETS="--only seqxgpt,semeval8c,beemo"
fi

echo "==> building word-level provenance data"
python scripts/build_data.py --hf ../data/hf --refs ../refs --out ../data/processed $SETS

python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| GPUs:', torch.cuda.device_count())"
cat <<'EOF'

Setup done. Next:
  LIMIT=50 bash run_seqxgpt.sh && LIMIT=50 bash run_provseg.sh   # quick test (../results_test/)
  bash run_seqxgpt.sh                      # run 1: official SeqXGPT     -> ../results/seqxgpt/
  bash run_provseg.sh                      # run 2: ProvSeg S1..S4       -> ../results/provseg/, ../results/compare/
  cat ../results/compare/table_ladder.md
EOF
