# v2-core: 3 vai trò của ProvSeg (Evidence Reader, Boundary Curator, Auditor)

**Nhánh:** `v2-core`. **Trạng thái code:** đủ (đã kiểm thử trên CPU). **Lệnh server:** `bash scripts/run_unattended.sh v2`. **Ước tính:** 2–3 giờ, khoảng 8GB VRAM.

## Mục tiêu
Lấy số thật cho phần lõi phương pháp. Gỡ lỗi NaN cũ: DeBERTa-v3 bị load ở fp16, AdamW làm trọng số thành NaN; đã sửa bằng trọng số fp32.

| Bước | Thêm gì | Sửa failure mode |
|---|---|---|
| S2 | DeBERTa-v3-base word states ghép vào khung SeqXGPT | score myopia |
| S2b | Boundary Curator: head "segment bắt đầu" (BCE, pos_weight 5) cộng vào điểm tag B/S so với M/E | boundary blur |
| S4 | S2b + evidence dropout 0.15 + Shapley chính xác theo kênh (lưu cả m_loo cho deletion test) | forced commitment / why |
| S3 | Bonferroni-LTT trên cal, cột Cov/Risk @0.05, @0.10 | forced commitment |

`scripts/diagnose.py` đo 3 failure mode và ra `figF1_failure_modes.pdf`. Đây là motivation cho trang 1 của paper.

## Task
- **T1 (đã xong):** code S2/S2b/S4, calibrator, diagnose, WindowDiff/SpanF1.
- **T2:** sau khi có số, phân tích:
  - Vai trò nào giúp, đo bằng hiệu số so với bước trước trên từng benchmark.
  - Lỗi gần ranh giới (err_dist_0..5) có giảm từ S0 sang S2b không.
  - Cov@0.10 có lớn hơn 0 và Risk có ≤ α không.
- **T3:** ghi `DECISIONS.md`.

## Cổng
`S2b > S0 + 2.0 word-F1` trên ≥ 2/3 benchmark (verify tự kiểm tra).

## Nếu FAIL (thử lần lượt, mỗi lần một thay đổi, chọn trên dev)
1. **S2 không vượt S1:**
   - Tăng số epoch lên 5.
   - Đổi `--lr 5e-5 --enc_lr 3e-5`.
   - Nếu VRAM cho phép, thử `ENC=microsoft/deberta-v3-large` (khoảng 12GB fp32 với bs 8).
2. **S2b không vượt S2:**
   - Quét `--lam_bnd` trong {0.2, 1.0} và `--bnd_pos_weight` trong {2, 10}.
   - Thử thay head bằng lớp MRF mean-field có học, ra nhãn theo từ (khác Markov ICLR'26 ở chỗ có giám sát).
3. **S4 thấp hơn S2b quá 0.5:** giảm evidence dropout xuống 0.05.
4. **Vẫn FAIL:** ghi DECISIONS, chuyển sang câu chuyện dự phòng "đo lường + bảo đảm" (xem `PROMPT_ICLR.md`). Vẫn đi tiếp v3 để có baseline.
