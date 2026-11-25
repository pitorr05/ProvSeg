# COMMON_RULES: luật chung cho mọi phiên bản (prompt nào cũng bắt đọc file này trước)

## Mục tiêu
Paper A*/Q1, nộp **ICLR 2028**. Phương pháp: **ProvSeg**, gán nhãn H/AI/Unknown cho từng từ trong văn bản người và AI cùng viết.
- **Gốc:** SeqXGPT (EMNLP'23).
- **3 vai trò** (theo mẫu ACE, ICLR'26):
  - **Evidence Reader**: bằng chứng tương phản theo từ cộng ngữ cảnh DeBERTa.
  - **Boundary Curator**: mô hình hoá điểm chuyển người↔AI.
  - **Auditor**: "Unknown" có bảo đảm (Learn-then-Test) cộng giải thích Shapley theo kênh.

Đọc thêm: `README.md`, `SKILL_ICLR.md`, `PROMPT_ICLR.md` (bản đồ SOTA 2025–2026).

## Ràng buộc phần cứng
- GPU H100 dùng chung với job khác, **chỉ còn khoảng 1 tháng**.
- Mỗi job tối đa khoảng 12GB VRAM.
- Dùng model nhỏ: LLM đặc trưng gpt2, gpt-neo-125m, pythia-410m, Qwen2.5-0.5B; encoder DeBERTa-v3-base.
- Mọi lần chạy trên server đều đi qua `bash scripts/run_unattended.sh vN`. Script này tự chờ GPU, ghi log, tự verify và ghi `../results/STATUS_vN.md`.

## Luật bất biến (vi phạm thì dừng)
1. **Không bịa số.** Mọi con số phải truy được về một file trong `../results/` hoặc `../runs/`.
2. **Test chỉ để báo cáo.**
   - Chọn model và siêu tham số trên dev.
   - Calibrator fit trên cal.
   - Không sửa gì sau khi đã nhìn số test của phiên bản đó.
3. **Claim "vượt" phải có kiểm định.**
   - Chạy ≥ 3 seed.
   - `scripts/significance.py` phải cho CI 95% của hiệu số > 0.
   - Không đạt thì chỉ được viết "ngang bằng".
4. **So sánh công bằng.** Cùng tài liệu test, cùng split, cùng ngân sách LLM đặc trưng. Baseline dùng code hoặc hàm chấm điểm chính thức; mọi chuyển đổi (ví dụ từ cấp văn bản xuống cấp từ) phải ghi rõ.
5. **Kiểm thử và verify trước khi push lên server:**
   - `python -m pytest -q tests` pass.
   - Chạy thử trên CPU: `LIMIT=5 SETS=semeval8c` với các bước của phiên bản, chạy hết, không NaN, `skipped_batches 0`.
   - `scripts/verify_run.py` trên bản chạy thử trả PASS (cho phép bỏ qua cổng).
6. **Git:**
   - Làm việc trên nhánh `vN-...` của phiên bản đó.
   - Commit với tác giả `Pham Quang Toan <phamtoan23022005@gmail.com>`, **không** thêm dòng Co-Authored-By hay nhắc tới AI.
   - Chỉ merge vào `main` khi `STATUS_vN.md` là PASS.
   - Sau khi merge, tag `vN-pass`.
7. **Ghi quyết định.** Kết thúc mỗi phiên bản bằng một mục trong `../results/DECISIONS.md`: ngày, phiên bản, số chính, go/no-go, lý do, việc tiếp theo.
8. **Trả lời người dùng bằng tiếng Việt**, gồm:
   - việc đã làm và file đã đổi;
   - lệnh chạy server (một dòng `run_unattended`);
   - bảng số kèm đường dẫn file nguồn;
   - quyết định go/no-go;
   - rủi ro còn lại.

## Bản đồ phiên bản
| Phiên bản | Nhánh | Mục đích | Cổng để đi tiếp |
|---|---|---|---|
| v1 | `v1-baseline` | SeqXGPT (S0) cộng S1, tái lập | verify sạch; S0 ≥ 85 F1 trên SemEval và SeqXGPT-Bench |
| v2 | `v2-core` | Evidence Reader, Boundary Curator, Auditor (S2, S2b, S4) cộng chẩn đoán lỗi | S2b ≥ S0 + 2 F1 trên ≥ 2/3 benchmark |
| v3 | `v3-sota` | Baseline 2024–2026: Seq/Sent-RoBERTa, Fast-DetectGPT, Binoculars, **Markov'26**, **WCP'26** | ProvSeg (S4) ≥ baseline mạnh nhất trên ≥ 2/3 benchmark (1 seed) |
| v4 | `v4-main` | Bảng chính: 3 seed, thêm benchmark (PASTED, CoAuthor), kiểm định bootstrap | CI > 0 so với baseline mạnh nhất trên ≥ 3/5 benchmark |
| v5 | `v5-robust` | Chéo miền, tấn công, chi phí, risk-coverage, kiểm định giải thích | Risk ≤ α trên mọi benchmark; giải thích qua được deletion test |
| v6 | `v6-paper` | Đóng gói: hình, bảng LaTeX, RESULTS_INDEX, bản thảo các mục Experiments | mọi bảng và hình tái tạo được bằng một lệnh |

**Nếu một cổng FAIL:** không đi tiếp phiên bản sau. Ở lại nhánh đó, làm theo mục "NẾU FAIL" trong `PROMPT.md` của phiên bản, rồi chạy lại.
