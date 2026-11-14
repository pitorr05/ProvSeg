# ProvSeg — word-level Human/AI provenance with certified abstention

Thay vì một điểm "AI-score" toàn văn bản, ProvSeg gán nhãn **từng từ** H / A / **Unknown**.
- **Where**: vùng nào là Human, vùng nào là AI.
- **What survived**: phần nào còn lại sau khi chỉnh sửa (nhãn retained provenance tính qua diff phiên bản).
- **Why**: đặc trưng bằng chứng theo từ (log-prob, entropy, Binoculars x-ppl, Fast-DetectGPT z, GLTR rank, RepreGuard rep-score).
- **Unknown**: từ không đủ chắc chắn. Có bảo đảm rằng rủi ro chọn lọc (selective risk) ≤ α với xác suất ≥ 1−δ (Learn-then-Test).

## Cấu trúc
```
provseg/data/schema.py            Record thống nhất (nhãn theo từ, split theo doc_id)
provseg/data/diff_provenance.py   lan truyền nhãn qua chuỗi phiên bản H→A→H
provseg/data/loaders/             seqxgpt, semeval8c, pasted, beemo, apt_eval, editlens
provseg/features/token_features.py  đặc trưng theo từ từ cặp observer/performer LLM + hướng RepreGuard
provseg/models/provseg.py         encoder + đặc trưng + CRF (marginals cho abstention)
provseg/conformal/selective.py    Learn-then-Test (Hoeffding–Bentkus) → Unknown
provseg/eval/metrics.py           word/span F1, boundary MAE, ratio MAE, AURC, ECE
scripts/build_data.py | train.py | evaluate.py | plots.py
run_all.sh                        toàn bộ pipeline trên GPU
tests/test_core.py                diff chain, CRF (so với brute-force), calibrator, metrics
```

## Dữ liệu (đều công khai, không tự sinh)
| Bộ | Vai trò | Nhãn từ |
|---|---|---|
| SeqXGPT-Bench (`refs/SeqXGPT`) | train/test | human prefix + AI continuation |
| SemEval-24 8C (`d0rj/SemEval2024-task8`) | train/test | boundary |
| Beemo (`toloka/beemo`) | train/test, records | AI output → expert human edit (diff) |
| PASTED (`linzw/PASTED`) | test (+OOD) | câu bị AI paraphrase |
| APT-Eval (`smksaha/apt-eval`) | test theo mức polish | human → LLM polish (diff) |
| EditLens (`refs/EditLens/data`, CC BY-NC-SA) | test | human → AI edit (diff) |

## Chạy trên server (GPU)
```bash
mkdir -p ~/work && cd ~/work
git clone https://github.com/pitorr05/ProvSeg.git && cd ProvSeg
bash setup_server.sh                 # cài thư viện, tải dữ liệu HF, build dữ liệu + graph cú pháp (~30–40 phút CPU)
LIMIT=50 bash run_all.sh             # chạy thử ~15 phút để kiểm tra môi trường
bash run_all.sh                      # vòng 1 đầy đủ
cat ../results_small/table_main.md
```

## Vòng 1 (mặc định): chỉ chạy ProvSeg trên 3 benchmark
```bash
bash run_all.sh                     # PROFILE=small, 1 seed: ProvSeg + ProvSeg (+records)
cat ../results_small/table_main.md  # bảng kết quả
```
- Benchmark: SemEval-24 8C (H→A, boundary MAE), PASTED (AI paraphrase), Beemo (người sửa văn bản AI).
- Metric: word macro-F1, word AUROC, boundary MAE (8C), Cov@0.05 (certified).
- Baseline (Fast-DetectGPT, Binoculars, RepreGuard, SeqXGPT, PASTED) sẽ chạy riêng bằng **repo gốc** của từng paper ở bước sau, không nằm trong script này.
- Vòng 2: `PROFILE=full bash run_all.sh`, thêm APT-Eval/EditLens/PASTED-OOD và các ablation.

## Chạy (đầy đủ)
```bash
pip install -r requirements.txt
python -m pytest -q tests                          # kiểm tra lõi (CPU)
bash run_all.sh                                     # GPU: data → features → train → eval → figs
OBS=tiiuae/falcon-7b PERF=tiiuae/falcon-7b-instruct bash run_all.sh   # cặp LLM gốc của Binoculars
```
Ước lượng trên 1×A100 80GB: đặc trưng mất khoảng 6–10 giờ cho khoảng 170k tài liệu. Mỗi lần train DeBERTa-v3-large mất khoảng 2–3 giờ. Nên chạy `provseg`, `seqroberta` và `abl_nofeat` với 1 seed trước (go/no-go), sau đó mới chạy đủ 3 seed và toàn bộ ablation.

## Baseline (dấu * nghĩa là chuyển từ document-level sang mức từ bằng cửa sổ trượt; được ghi rõ trong bài)
Binoculars*, Fast-DetectGPT*, Log-likelihood*, RepreGuard* (`evaluate.py`), Seq-RoBERTa (CRF, như trong SeqXGPT), SeqXGPT-style trên đặc trưng của chúng ta (`--encoder none`). EditLens có điểm số document-level sẵn trong CSV và được đưa vào phân tích ratio-MAE.
