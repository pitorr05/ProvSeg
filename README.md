# ProvSeg: phát hiện từng từ do Người hay AI viết, đi lên từ SeqXGPT (EMNLP 2023)

**Paper gốc (baseline chính): SeqXGPT** (Wang et al., EMNLP 2023). Đây là paper A* đầu tiên làm phát hiện AI ở mức câu/từ trong văn bản trộn Người+AI. Code chính thức được giữ **nguyên văn** trong `baselines/seqxgpt/official/` (Apache-2.0).

ProvSeg cải tiến SeqXGPT **từng bước**. Mỗi bước chỉ thêm một thứ, và bảng kết quả chính là cái thang này:

| Bước | Thêm gì | Giải quyết điểm yếu nào của SeqXGPT |
|---|---|---|
| **S0** | SeqXGPT gốc: 4 LLM log-prob (GPT-2-xl, GPT-Neo-2.7B, GPT-J-6B, LLaMA-7B), CNN, Transformer, BMES-Viterbi | (tái lập) |
| **S1** | Bằng chứng *tương phản* theo từ: Fast-DetectGPT z, Binoculars x-entropy, entropy, rank (cặp Qwen2.5-1.5B) | log-prob tuyệt đối là "điểm số không ý nghĩa" |
| **S2** | Ngữ cảnh ngữ nghĩa: DeBERTa-v3-large ghép vào backbone | không hiểu nội dung, yếu khi AI bị người sửa |
| **S3** | "Unknown" có bảo đảm: lỗi ≤ α với xác suất ≥ 1−δ (Learn-then-Test) | luôn bị ép trả lời H/A |
| **S4** | Evidence dropout + Shapley chính xác theo từng nguồn bằng chứng = **ProvSeg** | không giải thích được "vì sao từ này là AI" |

## Cấu trúc
```
baselines/seqxgpt/official/   code gốc SeqXGPT, Seq-RoBERTa, Sent-RoBERTa (không sửa)
baselines/zeroshot/           hàm chấm điểm gốc của Fast-DetectGPT và Binoculars (copy nguyên văn)
baselines/compat/             thay thế fastNLP (chỉ phần Viterbi BMES), chỉ dùng khi chưa cài fastNLP
provseg/models/seqx.py        SeqXNet = SeqXGPT tổng quát hoá (test: cho logits giống hệt bản gốc) + S1/S2/S4
provseg/features/llm_logprob.py    đặc trưng S0 (như gen_features.py của SeqXGPT, gọi HF trực tiếp)
provseg/features/token_features.py đặc trưng S1 (bằng chứng tương phản)
provseg/conformal/selective.py     S3 (Learn-then-Test)
scripts/train_seq.py          một trainer cho S0–S4 và các baseline có giám sát
scripts/zeroshot_sent.py      Fast-DetectGPT / Binoculars ở mức câu
scripts/eval_ladder.py        bảng kết quả (cùng tập test cho mọi dòng)
scripts/explain_s4.py         giải thích theo bằng chứng (Shapley) và các ví dụ HTML
scripts/run_ladder.sh         chạy toàn bộ
scripts/motivation.py         phân tích motivation M1/M2 (EditLens, APT, Beemo)
extras/                       bản v1 (EFP, GCN cú pháp, RepreGuard, ...): lưu lại, không chạy mặc định
```

## Benchmark (huấn luyện và kiểm tra trong cùng benchmark) và metric
- **SeqXGPT-Bench**: sân nhà của paper gốc.
- **SemEval-24 Task 8C**: Người viết rồi AI viết tiếp. Metric chính thức là boundary MAE.
- **Beemo**: AI viết, chuyên gia sửa lại.

Metric: word macro-F1, word AUROC, sentence macro-F1 (giao thức của SeqXGPT), boundary MAE, Cov@0.05.

## Chạy trên server
```bash
cd /data/ProvSeg && git pull
bash setup_server.sh                          # (lần đầu) cài đặt, tải dữ liệu, clone SeqXGPT-Bench
LIMIT=50 bash scripts/run_ladder.sh           # chạy thử toàn bộ với 50 tài liệu/split
bash scripts/run_ladder.sh                    # chạy thật
cat ../results_ladder/table_ladder.md
```
Chạy riêng từng phần: `STEPS="feats S0" bash scripts/run_ladder.sh`. Các giá trị STEPS: `feats S0 S1 S2 S4 baselines repro eval`.

**Mốc an toàn:** `repro` huấn luyện SeqXGPT 6 lớp trên SeqXGPT-Bench. Giá trị `multiclass_sent_f1` phải gần với số trong paper SeqXGPT, nếu không thì chưa cải tiến tiếp.

## Khác biệt có chủ đích so với code gốc (đều ghi trong paper)
- Đặc trưng S0 tính bằng HF trực tiếp, không qua server HTTP. Trung bình được tính theo ký tự thay vì theo byte, kết quả như nhau với văn bản ASCII.
- Đặc trưng được chuẩn hoá z-score theo tập train. Giữ checkpoint có dev F1 tốt nhất.
- Không tính đơn vị khoảng trắng: mỗi từ cách nhau bởi khoảng trắng là một bước của chuỗi.
- Seq-RoBERTa giải mã Viterbi trên các từ (lấy sub-token đầu của mỗi từ).
