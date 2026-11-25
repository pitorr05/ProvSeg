# v1-baseline: tái lập SeqXGPT (S0) và bằng chứng tương phản (S1)

**Nhánh:** `v1-baseline`. **Trạng thái code:** đủ. **Lệnh server:** `bash scripts/run_unattended.sh v1`

## Mục tiêu
Khoá baseline gốc. Mỗi bước sau đều được so với S0 này.
- **S0:** model SeqXGPT chính thức (`baselines/seqxgpt/official`), 4 LLM đặc trưng nhỏ.
- **S1:** cùng khung, nhưng bằng chứng là Fast-DetectGPT z, Binoculars x-entropy, entropy, rank (cặp Qwen2.5-0.5B).
- **Benchmark:** SeqXGPT-Bench, SemEval-24 8C, Beemo. Seed 13.

## Task
- **T1 (đã xong):** `run_seqxgpt.sh` và `run_provseg.sh` (S1).
- **T2:** ghi số vào `../results/DECISIONS.md` nếu chưa có. Lần chạy 15/11 cho:
  - S0 = 70.0 / 87.3 (MAE 28.5) / 89.8 word-F1.
  - S1 = 71.9 / 88.0 / 89.1.

## Verify
- **Sanity:** `verify_run.py` không có FAIL.
- **Cổng:** S0 ≥ 85 F1 trên SemEval và SeqXGPT-Bench (đã đạt), nghĩa là baseline được tái lập đúng.

## Nếu FAIL
- **NaN hoặc mô hình "chết":** kiểm tra dtype khi load model (transformers ≥ 5 load checkpoint fp16). `train_seq.py` đã ép `.float()`.
- **S0 thấp bất thường (dưới 80):** kiểm tra `data/feats_small/*_lp4.npz` có tạo bằng đúng 4 LLM không (`S0_LMS`), và split có lệch không.
