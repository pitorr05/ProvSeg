> Cách làm paper và kế hoạch ICLR 2028: xem `SKILL_ICLR.md`.

# ProvSeg: phát hiện từng từ do Người hay AI viết, đi lên từ SeqXGPT (EMNLP 2023)

**Paper gốc (baseline chính): SeqXGPT** (Wang et al., EMNLP 2023). Đây là paper A* đầu tiên làm phát hiện AI ở mức câu/từ trong văn bản trộn Người+AI. Code chính thức được giữ **nguyên văn** trong `baselines/seqxgpt/official/` (Apache-2.0).

ProvSeg cải tiến SeqXGPT **từng bước**. Mỗi bước chỉ thêm một thứ, và bảng kết quả chính là cái thang này:

| Bước | Thêm gì | Giải quyết điểm yếu nào của SeqXGPT |
|---|---|---|
| **S0** | Model SeqXGPT gốc: log-prob từ 4 LLM, CNN, Transformer, BMES-Viterbi | (baseline) |
| **S1** | Bằng chứng *tương phản* theo từ: Fast-DetectGPT z, Binoculars x-entropy, entropy, rank | log-prob tuyệt đối là "điểm số không ý nghĩa" |
| **S2** | Ngữ cảnh ngữ nghĩa: DeBERTa-v3 ghép vào backbone | không hiểu nội dung, yếu khi AI bị người sửa |
| **S2b** | Boundary Curator: head "segment bắt đầu ở đây" hợp nhất vào giải mã BMES | lỗi dồn quanh ranh giới người↔AI |
| **S3** | "Unknown" có bảo đảm: lỗi ≤ α với xác suất ≥ 1−δ (Learn-then-Test) | luôn bị ép trả lời H/A |
| **S4** | Evidence dropout + Shapley chính xác theo từng nguồn bằng chứng = **ProvSeg** | không giải thích được "vì sao từ này là AI" |

## Model và tài nguyên (mặc định: model nhỏ, cùng ngân sách cho cả hai lượt)
| | Lượt 1: SeqXGPT | Lượt 2: ProvSeg |
|---|---|---|
| LLM đặc trưng | gpt2, gpt-neo-125m, pythia-410m, Qwen2.5-0.5B | Qwen2.5-0.5B, Qwen2.5-0.5B-Instruct |
| Encoder | — | DeBERTa-v3-base (bf16) |

Ước lượng: khoảng 5GB ổ đĩa cho model, VRAM đỉnh 6–8GB (mỗi bước có in dòng `peak VRAM`).

**Vì sao paper SeqXGPT dùng 4 LLM (GPT-2-xl, GPT-Neo-2.7B, GPT-J-6B, LLaMA-7B)?**
- SeqXGPT-Bench được sinh ra bởi chính 4 model đó.
- Bài toán 6 lớp của paper là đoán *model nào* viết từng câu, nên cần chính các model ấy làm "máy đo".

Bài của mình chỉ phân biệt Người và AI (2 lớp), nên dùng model nhỏ cho **cả hai lượt** để so sánh công bằng. Muốn tái lập đúng số trong paper thì chạy lại với LLM gốc:
```bash
S0_LMS=gpt2-xl,EleutherAI/gpt-neo-2.7B,EleutherAI/gpt-j-6b,huggyllama/llama-7b bash run_seqxgpt.sh
```
Cách này tốn khoảng 55GB ổ đĩa.

## Cấu trúc repo
```
ProvSeg/
├── run_seqxgpt.sh            LƯỢT 1: chạy lại SeqXGPT gốc  -> ../runs/seqxgpt/,  ../results/seqxgpt/
├── run_provseg.sh            LƯỢT 2: cải tiến S1–S4 của mình -> ../runs/provseg/, ../results/provseg/
│                                     + bảng so sánh SeqXGPT vs ProvSeg -> ../results/compare/
├── run_baselines.sh          (để sau) Seq/Sent-RoBERTa, Fast-DetectGPT, Binoculars -> ../runs/baselines/
├── setup_server.sh           cài đặt + tải dữ liệu (chạy 1 lần)
│
├── baselines/                CODE CỦA NGƯỜI KHÁC (nguyên văn, giữ license)
│   ├── seqxgpt/official/     SeqXGPT, Seq-RoBERTa, Sent-RoBERTa (repo chính thức SeqXGPT)
│   ├── zeroshot/             hàm chấm điểm gốc của Fast-DetectGPT, Binoculars
│   └── compat/               thay thế fastNLP (chỉ Viterbi BMES)
│
├── provseg/                  CODE CỦA MÌNH
│   ├── data/                 đọc 7 bộ dữ liệu → nhãn H/AI theo từng từ, chia train/dev/cal/test
│   ├── features/llm_logprob.py     đặc trưng của SeqXGPT (lượt 1)
│   ├── features/token_features.py  bằng chứng tương phản (lượt 2, S1)
│   ├── models/seqx.py        SeqXNet: kiến trúc SeqXGPT + S1/S2/S4
│   ├── conformal/selective.py      Unknown có bảo đảm (S3)
│   └── eval/metrics.py       metric mức từ / câu / boundary
│
├── scripts/                  train_seq.py (train mọi mô hình), eval_ladder.py (bảng), explain_s4.py (giải thích),
│                             zeroshot_sent.py, build_data.py, motivation.py, _env.sh (cấu hình chung)
├── tests/                    kiểm tra (gồm: SeqXNet cho kết quả y hệt SeqXGPT gốc)
├── paper/                    bản thảo LaTeX
└── extras/                   bản v1 cũ (EFP, GCN cú pháp, RepreGuard), lưu lại, không chạy
```

## Chạy trên server
```bash
cd /data/ProvSeg && git pull
SKIP_PIP=1 bash setup_server.sh      # (1 lần) clone SeqXGPT-Bench, build dữ liệu seqxgpt/semeval8c/beemo

LIMIT=50 bash run_seqxgpt.sh         # thử nhanh lượt 1 (kết quả trong ../results_test/)
LIMIT=50 bash run_provseg.sh         # thử nhanh lượt 2

bash run_seqxgpt.sh                  # LƯỢT 1 thật: SeqXGPT gốc
bash run_provseg.sh                  # LƯỢT 2 thật: cải tiến của mình
cat ../results/compare/table_ladder.md   # SeqXGPT vs S1, S2, S2b, S4 trên cùng tài liệu test
python scripts/diagnose.py --runs "../runs/seqxgpt/S0_*" "../runs/provseg/*" --data ../data/processed --out ../results/diagnose
                                          # đo 3 lỗi của SeqXGPT (motivation), ra figF1_failure_modes.pdf
```
`../results/seqxgpt/repro_6way_*.json` chứa kết quả SeqXGPT 6 lớp. Kết quả này chỉ khớp với số trong paper khi chạy bằng 4 LLM gốc (xem `S0_LMS` ở trên). Với model nhỏ, đây chỉ là một con số tham khảo.

## Khác biệt có chủ đích so với code gốc (đều ghi trong paper)
- Đặc trưng S0 tính bằng HF trực tiếp, không qua server HTTP. Trung bình được tính theo ký tự thay vì theo byte, kết quả như nhau với văn bản ASCII.
- Đặc trưng được chuẩn hoá z-score theo tập train. Giữ checkpoint có dev F1 tốt nhất.
- Không tính đơn vị khoảng trắng: mỗi từ cách nhau bởi khoảng trắng là một bước của chuỗi.
- Seq-RoBERTa giải mã Viterbi trên các từ (lấy sub-token đầu của mỗi từ).
