# v5-robust: độ vững, chi phí, bảo đảm và độ trung thực của giải thích

**Nhánh:** `v5-robust`, tạo từ `v4-main`. **Lệnh server:** `bash scripts/run_unattended.sh v5`. **Ước tính:** 4–6 giờ.

## Mục tiêu
Trả lời trước các câu hỏi reviewer chắc chắn sẽ hỏi:
- Có tổng quát hoá sang miền khác không?
- Có chịu được paraphrase không?
- Unknown có thật sự giữ lỗi ≤ α không?
- Giải thích có trung thực không?
- Tốn bao nhiêu tài nguyên?

## Task
### T1 (đã có trong `run.sh`): chéo miền và OOD
- Ma trận chéo miền 3×3 (train A, test B).
- PASTED sang PASTED-OOD (paraphrase).

### T2: `scripts/explain_deletion.py`
- Đọc `shap_<ds>.pkl` của các run S4. Run S4 của v2 trở đi đã lưu `m_full` và `m_loo`.
- Với mỗi từ, so 3 mức giảm margin:
  - Δ_top = m_full − m_loo[argmax |phi|];
  - Δ_rand = m_full − m_loo[kênh ngẫu nhiên];
  - Δ_min = m_full − m_loo[argmin |phi|].
- **Báo:**
  - trung bình |Δ_top| > |Δ_rand| > |Δ_min|, kèm CI bootstrap;
  - tỷ lệ nhãn bị lật khi bỏ kênh top so với kênh ngẫu nhiên;
  - tương quan Spearman giữa |phi| và |Δ|.
- **Đầu ra:** `../results/robust/explain_faithfulness.csv` và hình.
- **Test:** với model tuyến tính giả (cộng tính), phi phải khớp đúng Δ.

### T3: `scripts/plot_risk_coverage.py`
- Với mỗi run và mỗi benchmark, vẽ đường risk–coverage (từ có độ tin cậy cao nhất trước).
- Đánh dấu điểm Bonferroni-LTT ở α = 0.05 và 0.10, có bảo đảm.
- So với ngưỡng max-prob "ngây thơ" (không bảo đảm) chọn trên cal: báo tỷ lệ benchmark mà Risk_test > α.
- **Đầu ra:** `risk_coverage_<ds>.pdf`, `selective_summary.csv`.

### T4: `scripts/parse_cost.py`
- Đọc log trong `../logs/` lấy `peak VRAM`, thời gian mỗi epoch và tổng thời gian.
- Ra `cost.csv`: mỗi phương pháp một dòng (GPU-giờ, VRAM, ms/tài liệu khi dự đoán).

## Cổng
- **Bảo đảm:** Risk_test ≤ α + 0.01 trên mọi benchmark.
- **Giải thích:** |Δ_top| > |Δ_rand| với CI > 0.

## Nếu FAIL
- **Risk vượt α:** kiểm tra giả định exchangeable (cal và test có cùng phân phối không). Nếu không, báo theo từng benchmark và nêu rõ giới hạn.
- **Giải thích không qua deletion test:** tăng evidence dropout lên 0.25. Khi đó S4 phải chạy lại (chọn trên dev).
