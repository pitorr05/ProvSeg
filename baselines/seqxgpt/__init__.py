"""Official SeqXGPT code (Wang et al., EMNLP 2023), vendored unchanged in ``official/``
from https://github.com/Jihuai-wpy/SeqXGPT (commit da2f264, Apache-2.0).

Importing this package makes ``fastNLP`` importable: the real package is used when installed,
otherwise the behaviour-equivalent shim in ``baselines/compat`` (only CRF Viterbi decoding with
BMES constraints is needed, the CRF transitions are zeroed and never trained by SeqXGPT).
"""
import os
import sys

try:  # pragma: no cover - depends on the environment
    import fastNLP.modules.torch  # noqa: F401
except Exception:  # noqa: BLE001
    for k in [k for k in sys.modules if k == "fastNLP" or k.startswith("fastNLP.")]:
        del sys.modules[k]
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "compat"))

from .official.seqxgpt_model import ModelWiseTransformerClassifier, ConvFeatureExtractionModel  # noqa: E402,F401
from .official.seq_roberta_model import RoBERTClassifier as SeqRoBERTaClassifier  # noqa: E402,F401
from .official.sent_roberta_model import RoBERTClassifier as SentRoBERTaClassifier  # noqa: E402,F401
